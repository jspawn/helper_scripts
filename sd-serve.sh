#!/bin/bash
# ============================================================================
#  sd-serve.sh -- Interactive launcher for sd-server (stable-diffusion.cpp)
#  Hardware: 2x AMD Radeon AI PRO R9700 (64GB total VRAM, gfx1201, RDNA4)
#  Backends: Vulkan (primary; ROCm has known bugs on RDNA4 for diffusion)
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared RDNA4 env + visibility helper
# shellcheck source=lib/rdna4-env.sh
source "$SCRIPT_DIR/lib/rdna4-env.sh"

# -- Paths ------------------------------------------------------------------
MODEL_DIR="${SD_MODEL_DIR:-/srv/models/sd}"
SD_ROCM="${SD_ROCM:-/srv/llama/stable-diffusion.cpp-rocm/build/bin/sd-server}"
SD_VULKAN="${SD_VULKAN:-/srv/llama/stable-diffusion.cpp-vulkan/build/bin/sd-server}"
PRESETS_DIR="${SD_PRESETS_DIR:-/srv/llama/presets-sd}"
OUTPUT_DIR="${SD_OUTPUT_DIR:-/srv/output/sd}"
# Shared web UI install (built by build_tools.sh -> build_sd_frontend)
SD_FRONTEND_HTML="${SD_FRONTEND_HTML:-/srv/llama/stable-diffusion/web/server/frontend/dist/index.html}"

# -- Defaults ---------------------------------------------------------------
# Vulkan is the default for SD: per upstream issue #1213, recent FLUX.2-9B
# crashes on ROCm but works on Vulkan. SDXL/SD1.5 are stable on both.
BACKEND="vulkan"
HOST="0.0.0.0"
PORT="8081"               # llama-server uses 8080, leave room for both
THREADS="-1"

# Model assembly (filled in per-model, see apply_model_preset)
DIFFUSION_MODEL=""        # main flow/diffusion weights
VAE=""                    # autoencoder
LLM=""                    # FLUX.2: Qwen3-based text encoder
CLIP_L=""                 # SD/SDXL: CLIP-L encoder
CLIP_G=""                 # SDXL: CLIP-G encoder
T5XXL=""                  # FLUX.1 / SD3: T5-XXL encoder
LORA=""                   # optional LoRA path
EMBED_DIR=""              # optional embeddings dir

# Generation defaults
STEPS="20"
CFG_SCALE="7.0"
SAMPLER="euler_a"         # euler_a, euler, dpm++2m, dpm++2mkarras, lcm, ...
WIDTH="1024"
HEIGHT="1024"
SEED="-1"                 # -1 = random
BATCH_COUNT="1"

# Performance
DIFFUSION_FA="on"         # flash attention in diffusion model (sd.cpp -DSD_FLASH_ATTN)
OFFLOAD_TO_CPU="off"      # offload weights to CPU when not in use (saves VRAM)
VAE_ON_CPU="off"          # run VAE on CPU only (big VRAM saving for tight fits)
CLIP_ON_CPU="off"         # run text encoder on CPU
TAESD=""                  # tiny autoencoder for fast preview decoding

# Multi-GPU (SD doesn't benefit from multi-GPU for a single image; this is
# really just pin-to-one-card. The other R9700 is free for parallel work.)
VISIBLE_DEVICES="0"

# Web UI: serve the built Vite frontend at "/" via sd-server --serve-html-path.
# Auto-detected from $SD_FRONTEND_HTML; toggle off if you only want the API.
SERVE_WEBUI="auto"        # auto | on | off

EXTRA_ARGS=""

# -- Colors -----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

header() {
    clear
    echo -e "\n  ${CYAN}${BOLD}sd-server launcher${NC} ${DIM}(2x R9700, 64GB VRAM)${NC}\n"
}

divider() {
    echo -e "${DIM}  ---------------------------------------------------------${NC}"
}

update_param() {
    local var_name="$1"
    local prompt_text="$2"
    local current_val="${!var_name:-none}"
    local input
    read -rp "  $prompt_text [$current_val]: " input
    [[ -n "$input" ]] && printf -v "$var_name" "%s" "$input"
}

current_binary() {
    case "$BACKEND" in
        rocm)   echo "$SD_ROCM" ;;
        vulkan) echo "$SD_VULKAN" ;;
        *)      echo "$SD_VULKAN" ;;
    esac
}

# Resolve effective SERVE_WEBUI: auto -> on if file exists, else off
webui_active() {
    case "$SERVE_WEBUI" in
        on)   [[ -f "$SD_FRONTEND_HTML" ]] && return 0 || return 1 ;;
        off)  return 1 ;;
        auto) [[ -f "$SD_FRONTEND_HTML" ]] && return 0 || return 1 ;;
    esac
}

webui_status() {
    if webui_active; then
        echo "on ${DIM}($SD_FRONTEND_HTML)${NC}"
    elif [[ "$SERVE_WEBUI" == "off" ]]; then
        echo "off ${DIM}(API only)${NC}"
    else
        echo "off ${DIM}(build: ./build_tools.sh sd vulkan)${NC}"
    fi
}

# ============================================================================
#  PRESET SAVE / LOAD
# ============================================================================

mkdir -p "$PRESETS_DIR" "$OUTPUT_DIR"

PARAM_LIST=(BACKEND HOST PORT THREADS
            DIFFUSION_MODEL VAE LLM CLIP_L CLIP_G T5XXL LORA EMBED_DIR
            STEPS CFG_SCALE SAMPLER WIDTH HEIGHT SEED BATCH_COUNT
            DIFFUSION_FA OFFLOAD_TO_CPU VAE_ON_CPU CLIP_ON_CPU TAESD
            VISIBLE_DEVICES SERVE_WEBUI EXTRA_ARGS)

save_preset() {
    read -rp "  Preset name: " pname
    [[ -z "$pname" ]] && return
    pname=$(echo "$pname" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    local file="$PRESETS_DIR/${pname}.conf"
    : > "$file"
    for p in "${PARAM_LIST[@]}"; do
        echo "${p}=${!p}" >> "$file"
    done
    echo -e "  ${GREEN}Saved -> ${file}${NC}"
    sleep 0.7
}

load_preset_file() {
    local file="$1"
    while IFS='=' read -r key val; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        printf -v "$key" "%s" "$val" 2>/dev/null || true
    done < "$file"
}

load_preset_menu() {
    local files=()
    shopt -s nullglob
    files=("$PRESETS_DIR"/*.conf)
    shopt -u nullglob

    if [[ ${#files[@]} -eq 0 ]]; then
        echo -e "  ${DIM}No saved presets in $PRESETS_DIR${NC}"
        sleep 0.7
        return
    fi

    echo -e "\n  ${BOLD}Saved Presets${NC}"
    divider
    for i in "${!files[@]}"; do
        local name
        name=$(basename "${files[$i]}" .conf)
        echo -e "  ${GREEN}$((i+1)))${NC} ${name}"
    done
    echo
    read -rp "  Select [1-${#files[@]}] (or Enter to cancel): " choice
    [[ -z "$choice" ]] && return
    if [[ "$choice" -ge 1 && "$choice" -le ${#files[@]} ]] 2>/dev/null; then
        load_preset_file "${files[$((choice-1))]}"
        echo -e "  ${GREEN}Loaded: $(basename "${files[$((choice-1))]}" .conf)${NC}"
        sleep 0.7
    fi
}

# ============================================================================
#  MODEL FAMILY DETECTION + PRESETS
# ----------------------------------------------------------------------------
#  Unlike LLMs, SD models are multi-file assemblies. We treat each model
#  directory under $MODEL_DIR as one "model": it should contain a diffusion
#  weights file plus the family's required companions (VAE + encoders).
#
#  Layout convention:
#    $MODEL_DIR/<model-name>/
#      diffusion.{gguf,safetensors}   <-- main flow/diffusion weights
#      vae.safetensors                <-- autoencoder
#      (one of:)
#      llm.gguf                       <-- FLUX.2 (Qwen3-based encoder)
#      t5xxl.safetensors + clip_l...  <-- FLUX.1 / SD3
#      clip_l + clip_g + (no t5)      <-- SDXL
#      (none)                         <-- SD 1.5 (encoder baked in)
# ============================================================================

# Heuristically detect family from filename and set required companion paths.
# Sets: DIFFUSION_MODEL, VAE, LLM, CLIP_L, CLIP_G, T5XXL + sane defaults.
apply_model_preset() {
    local dir="$SELECTED_MODEL_DIR"
    local fname
    fname=$(basename "$dir")

    # Reset
    DIFFUSION_MODEL=""; VAE=""; LLM=""
    CLIP_L=""; CLIP_G=""; T5XXL=""
    LORA=""; EMBED_DIR=""; TAESD=""
    DIFFUSION_FA="on"; OFFLOAD_TO_CPU="off"
    VAE_ON_CPU="off"; CLIP_ON_CPU="off"
    VISIBLE_DEVICES="0"

    # Find diffusion model: prefer gguf, fall back to safetensors.
    # Exclude obvious encoder/VAE files to avoid misidentifying companions.
    local diff
    diff=$(find "$dir" -maxdepth 1 -type f \
        \( -iname "*flux*klein*.gguf"  -o -iname "*flux*dev*.gguf" \
        -o -iname "*flux*schnell*.gguf" -o -iname "*flux*.gguf" \
        -o -iname "*sdxl*.gguf"         -o -iname "*sd3*.gguf" \
        -o -iname "diffusion*.gguf"     -o -iname "*-Q[0-9]*.gguf" \
        -o -iname "*flux*.safetensors"  -o -iname "*sdxl*.safetensors" \
        -o -iname "*sd3*.safetensors"   -o -iname "diffusion*.safetensors" \) \
        ! -iname "*encoder*" ! -iname "*qwen*" ! -iname "*t5*" \
        ! -iname "*clip*" ! -iname "*vae*" ! -iname "*ae.*" \
        2>/dev/null | head -1) || true
    DIFFUSION_MODEL="$diff"

    # Always look for a VAE
    VAE=$(find "$dir" -maxdepth 1 -type f \
        \( -iname "ae.safetensors" -o -iname "*vae*.safetensors" \
        -o -iname "*_ae.safetensors" \) 2>/dev/null | head -1) || true

    # Companion encoders
    LLM=$(find "$dir" -maxdepth 1 -type f \
        \( -iname "*qwen*.gguf" -o -iname "*qwen*.safetensors" \
        -o -iname "*encoder*.gguf" -o -iname "llm*.gguf" \) \
        ! -iname "*klein-9b-Q*" ! -iname "*klein-4b-Q*" \
        2>/dev/null | head -1) || true
    T5XXL=$(find "$dir" -maxdepth 1 -type f -iname "*t5*xxl*.*" 2>/dev/null | head -1) || true
    CLIP_L=$(find "$dir" -maxdepth 1 -type f -iname "*clip*l*.*" 2>/dev/null | head -1) || true
    CLIP_G=$(find "$dir" -maxdepth 1 -type f -iname "*clip*g*.*" 2>/dev/null | head -1) || true

    # Family-specific defaults
    case "${fname,,}" in
        *flux*2*klein*9b*|*flux2*klein*9b*)
            WIDTH="1024"; HEIGHT="1024"; STEPS="4"
            CFG_SCALE="1.0"; SAMPLER="euler"
            OFFLOAD_TO_CPU="off"  # 32GB per card is enough for 9B + 8B encoder
            echo -e "  ${CYAN}Preset: FLUX.2 klein 9B -- 4 steps, cfg 1.0${NC}"
            echo -e "  ${YELLOW}!${NC} Known issue: 9B may crash on ROCm (issue #1213); Vulkan recommended" ;;

        *flux*2*klein*4b*|*flux2*klein*4b*)
            WIDTH="1024"; HEIGHT="1024"; STEPS="4"
            CFG_SCALE="1.0"; SAMPLER="euler"
            echo -e "  ${CYAN}Preset: FLUX.2 klein 4B -- 4 steps, cfg 1.0${NC}" ;;

        *flux*1*dev*|*flux1*dev*)
            WIDTH="1024"; HEIGHT="1024"; STEPS="20"
            CFG_SCALE="3.5"; SAMPLER="euler"
            echo -e "  ${CYAN}Preset: FLUX.1 dev -- 20 steps, cfg 3.5${NC}" ;;

        *flux*1*schnell*|*flux1*schnell*)
            WIDTH="1024"; HEIGHT="1024"; STEPS="4"
            CFG_SCALE="1.0"; SAMPLER="euler"
            echo -e "  ${CYAN}Preset: FLUX.1 schnell -- 4 steps, cfg 1.0${NC}" ;;

        *sdxl*|*sd_xl*)
            WIDTH="1024"; HEIGHT="1024"; STEPS="30"
            CFG_SCALE="7.0"; SAMPLER="dpm++2mkarras"
            echo -e "  ${CYAN}Preset: SDXL -- 30 steps, cfg 7.0, dpm++2m karras${NC}" ;;

        *sd3*|*sd_3*)
            WIDTH="1024"; HEIGHT="1024"; STEPS="28"
            CFG_SCALE="4.5"; SAMPLER="euler"
            echo -e "  ${CYAN}Preset: SD3 -- 28 steps, cfg 4.5${NC}" ;;

        *sd*1.5*|*sd_v1*|*sd-v1*|*sd15*)
            WIDTH="512"; HEIGHT="512"; STEPS="25"
            CFG_SCALE="7.5"; SAMPLER="euler_a"
            echo -e "  ${CYAN}Preset: SD 1.5 -- 25 steps, cfg 7.5, 512x512${NC}" ;;

        *)
            WIDTH="1024"; HEIGHT="1024"; STEPS="20"
            CFG_SCALE="7.0"; SAMPLER="euler_a"
            echo -e "  ${DIM}No family match -- generic 1024x1024, 20 steps${NC}" ;;
    esac

    # Report what was auto-found
    [[ -n "$DIFFUSION_MODEL" ]] && echo -e "  ${GREEN}OK${NC} diffusion: $(basename "$DIFFUSION_MODEL")"
    [[ -n "$VAE" ]]             && echo -e "  ${GREEN}OK${NC} vae:       $(basename "$VAE")"
    [[ -n "$LLM" ]]             && echo -e "  ${GREEN}OK${NC} llm:       $(basename "$LLM")"
    [[ -n "$T5XXL" ]]           && echo -e "  ${GREEN}OK${NC} t5xxl:     $(basename "$T5XXL")"
    [[ -n "$CLIP_L" ]]          && echo -e "  ${GREEN}OK${NC} clip_l:    $(basename "$CLIP_L")"
    [[ -n "$CLIP_G" ]]          && echo -e "  ${GREEN}OK${NC} clip_g:    $(basename "$CLIP_G")"

    # Warn about missing required pieces
    [[ -z "$DIFFUSION_MODEL" ]] && echo -e "  ${RED}!${NC} no diffusion weights found in $dir"
    return 0
}

# ============================================================================
#  MODEL SELECTION
# ============================================================================

select_model() {
    header
    echo -e "  ${BOLD}Step 1: Select a model${NC}"
    echo -e "  ${DIM}Each model is a directory under $MODEL_DIR${NC}"
    divider
    echo

    MODELS=()
    local display=()

    shopt -s nullglob
    for d in "$MODEL_DIR"/*/; do
        local clean_d="${d%/}"
        MODELS+=("$clean_d")
        local sz cnt
        sz=$(du -sh "$clean_d" 2>/dev/null | cut -f1) || sz="?"
        cnt=$(find "$clean_d" -maxdepth 1 -type f \( -name "*.gguf" -o -name "*.safetensors" \) 2>/dev/null | wc -l)
        display+=("$(basename "$clean_d")/ ${DIM}(${sz}, ${cnt} files)${NC}")
    done
    shopt -u nullglob

    if [[ ${#MODELS[@]} -eq 0 ]]; then
        echo -e "  ${RED}No model directories found in ${MODEL_DIR}${NC}"
        echo -e "  ${DIM}Layout: ${MODEL_DIR}/<name>/{diffusion.gguf,vae.safetensors,...}${NC}"
        exit 1
    fi

    for i in "${!MODELS[@]}"; do
        echo -e "  ${GREEN}$((i+1)))${NC} ${display[$i]}"
    done

    echo
    read -rp "  Select model [1-${#MODELS[@]}] (or 'q' to quit): " choice
    [[ "${choice,,}" == "q" ]] && { echo -e "  ${YELLOW}Bye!${NC}"; exit 0; }

    if [[ "$choice" -ge 1 && "$choice" -le ${#MODELS[@]} ]] 2>/dev/null; then
        SELECTED_MODEL_DIR="${MODELS[$((choice-1))]}"
        echo -e "\n  ${GREEN}OK${NC} $(basename "$SELECTED_MODEL_DIR")"
    else
        echo -e "  ${RED}Invalid selection${NC}"; exit 1
    fi

    apply_model_preset

    # Check for saved preset matching this model
    local model_preset="$PRESETS_DIR/$(basename "$SELECTED_MODEL_DIR").conf"
    if [[ -f "$model_preset" ]]; then
        echo -e "  ${CYAN}Found saved config for this model${NC}"
        read -rp "  Load it? [Y/n]: " load_it
        if [[ "${load_it,,}" != "n" ]]; then
            load_preset_file "$model_preset"
            echo -e "  ${GREEN}OK${NC} Loaded saved config"
        fi
    fi

    sleep 0.7
}

# ============================================================================
#  PARAMETER MENU
# ============================================================================

configure_params() {
    while true; do
        header
        echo -e "  ${BOLD}Step 2: Configure parameters${NC}"
        echo -e "  ${DIM}Model: $(basename "$SELECTED_MODEL_DIR")${NC}"
        echo -e "  ${DIM}Backend: ${BACKEND}  |  Binary: $(current_binary)${NC}"
        divider
        echo
        echo -e "  ${BOLD}Model Files${NC}"
        echo -e "  ${GREEN} 1)${NC} diffusion      ${BOLD}${DIFFUSION_MODEL:-MISSING}${NC}"
        echo -e "  ${GREEN} 2)${NC} vae            ${BOLD}${VAE:-none}${NC}"
        echo -e "  ${GREEN} 3)${NC} llm (FLUX.2)   ${BOLD}${LLM:-none}${NC}"
        echo -e "  ${GREEN} 4)${NC} t5xxl          ${BOLD}${T5XXL:-none}${NC}"
        echo -e "  ${GREEN} 5)${NC} clip_l/clip_g  ${BOLD}${CLIP_L:-none} / ${CLIP_G:-none}${NC}"
        echo -e "  ${GREEN} 6)${NC} lora           ${BOLD}${LORA:-none}${NC}"
        echo
        echo -e "  ${BOLD}Generation${NC}"
        echo -e "  ${GREEN} 7)${NC} Resolution     ${BOLD}${WIDTH}x${HEIGHT}${NC}"
        echo -e "  ${GREEN} 8)${NC} Steps          ${BOLD}${STEPS}${NC}"
        echo -e "  ${GREEN} 9)${NC} CFG scale      ${BOLD}${CFG_SCALE}${NC}"
        echo -e "  ${GREEN}10)${NC} Sampler        ${BOLD}${SAMPLER}${NC} ${DIM}(euler, euler_a, dpm++2m, dpm++2mkarras, lcm)${NC}"
        echo -e "  ${GREEN}11)${NC} Seed           ${BOLD}${SEED}${NC} ${DIM}(-1 = random)${NC}"
        echo
        echo -e "  ${BOLD}Performance${NC}"
        echo -e "  ${GREEN}12)${NC} Diffusion FA   ${BOLD}${DIFFUSION_FA}${NC} ${DIM}(--diffusion-fa)${NC}"
        echo -e "  ${GREEN}13)${NC} Offload to CPU ${BOLD}${OFFLOAD_TO_CPU}${NC} ${DIM}(saves VRAM, slower)${NC}"
        echo -e "  ${GREEN}14)${NC} VAE on CPU     ${BOLD}${VAE_ON_CPU}${NC}"
        echo -e "  ${GREEN}15)${NC} CLIP on CPU    ${BOLD}${CLIP_ON_CPU}${NC}"
        echo
        echo -e "  ${BOLD}Server${NC}"
        echo -e "  ${GREEN}16)${NC} Host/Port      ${BOLD}${HOST}:${PORT}${NC}"
        echo -e "  ${GREEN}17)${NC} Visible devs   ${BOLD}${VISIBLE_DEVICES:-all}${NC}"
        echo -e "  ${GREEN}18)${NC} Backend        ${BOLD}${BACKEND}${NC} ${DIM}(vulkan recommended for FLUX.2)${NC}"
        echo -e "  ${GREEN}19)${NC} Web UI         ${BOLD}$(webui_status)${NC}"
        echo -e "  ${GREEN}20)${NC} Extra args     ${BOLD}${EXTRA_ARGS:-none}${NC}"
        echo
        divider
        echo -e "  ${BOLD}Presets${NC}"
        echo -e "  ${YELLOW} p1)${NC} Fast       ${DIM}-- few steps, cfg 1.0 (FLUX schnell / klein)${NC}"
        echo -e "  ${YELLOW} p2)${NC} Quality    ${DIM}-- 30 steps, dpm++2m karras${NC}"
        echo -e "  ${YELLOW} p3)${NC} Tight VRAM ${DIM}-- VAE+CLIP on CPU, offload on${NC}"
        echo
        echo -e "  ${BOLD}Config${NC}"
        echo -e "  ${YELLOW}  s)${NC} Save current config"
        echo -e "  ${YELLOW}  l)${NC} Load saved config"
        echo -e "  ${YELLOW} sm)${NC} Save as model default"
        echo
        divider
        echo -e "  ${GREEN} r)${NC}  Run server"
        echo -e "  ${GREEN} d)${NC}  Dry run (show command)"
        echo -e "  ${GREEN} q)${NC}  Quit"
        echo
        read -rp "  > " opt

        case "$opt" in
            1)  update_param DIFFUSION_MODEL "Diffusion model path" ;;
            2)  update_param VAE "VAE path" ;;
            3)  update_param LLM "LLM (Qwen3 for FLUX.2) path" ;;
            4)  update_param T5XXL "T5-XXL path" ;;
            5)  update_param CLIP_L "CLIP-L path"; update_param CLIP_G "CLIP-G path" ;;
            6)  update_param LORA "LoRA path" ;;
            7)  update_param WIDTH "Width"; update_param HEIGHT "Height" ;;
            8)  update_param STEPS "Steps" ;;
            9)  update_param CFG_SCALE "CFG scale" ;;
            10) update_param SAMPLER "Sampler" ;;
            11) update_param SEED "Seed (-1=random)" ;;
            12) update_param DIFFUSION_FA "Diffusion flash attn (on/off)" ;;
            13) update_param OFFLOAD_TO_CPU "Offload to CPU (on/off)" ;;
            14) update_param VAE_ON_CPU "VAE on CPU (on/off)" ;;
            15) update_param CLIP_ON_CPU "CLIP on CPU (on/off)" ;;
            16) update_param HOST "Host"; update_param PORT "Port" ;;
            17) update_param VISIBLE_DEVICES "Visible devices (e.g. 0 or 0,1)" ;;
            18) update_param BACKEND "Backend (vulkan/rocm)" ;;
            19) update_param SERVE_WEBUI "Web UI (auto/on/off)" ;;
            20) update_param EXTRA_ARGS "Extra args" ;;

            p1) STEPS="4"; CFG_SCALE="1.0"; SAMPLER="euler"
                echo -e "  ${GREEN}OK${NC} Fast preset"; sleep 0.5 ;;
            p2) STEPS="30"; CFG_SCALE="7.0"; SAMPLER="dpm++2mkarras"
                echo -e "  ${GREEN}OK${NC} Quality preset"; sleep 0.5 ;;
            p3) OFFLOAD_TO_CPU="on"; VAE_ON_CPU="on"; CLIP_ON_CPU="on"
                echo -e "  ${GREEN}OK${NC} Tight VRAM preset"; sleep 0.5 ;;

            s)  save_preset ;;
            l)  load_preset_menu ;;
            sm) local mfile="$PRESETS_DIR/$(basename "$SELECTED_MODEL_DIR").conf"
                : > "$mfile"
                for p in "${PARAM_LIST[@]}"; do echo "${p}=${!p}" >> "$mfile"; done
                echo -e "  ${GREEN}Saved model default${NC}"; sleep 0.7 ;;

            r)  run_server; break ;;
            d)  dry_run ;;
            q)  echo -e "  ${YELLOW}Bye!${NC}"; exit 0 ;;
            *)  echo -e "  ${RED}Invalid${NC}"; sleep 0.3 ;;
        esac
    done
}

# ============================================================================
#  BUILD COMMAND
# ============================================================================

build_command() {
    local bin
    bin="$(current_binary)"

    CMD=("$bin")
    CMD+=(--listen-ip "$HOST" --listen-port "$PORT")

    # Web UI: only add the flag if the built dist/ exists, so a missing
    # frontend gracefully degrades to the API-only placeholder page.
    if webui_active; then
        CMD+=(--serve-html-path "$SD_FRONTEND_HTML")
    fi

    [[ -n "$DIFFUSION_MODEL" ]] && CMD+=(--diffusion-model "$DIFFUSION_MODEL")
    [[ -n "$VAE" ]]             && CMD+=(--vae "$VAE")
    [[ -n "$LLM" ]]             && CMD+=(--llm "$LLM")
    [[ -n "$T5XXL" ]]           && CMD+=(--t5xxl "$T5XXL")
    [[ -n "$CLIP_L" ]]          && CMD+=(--clip_l "$CLIP_L")
    [[ -n "$CLIP_G" ]]          && CMD+=(--clip_g "$CLIP_G")
    [[ -n "$LORA" ]]            && CMD+=(--lora-model-dir "$(dirname "$LORA")")
    [[ -n "$EMBED_DIR" ]]       && CMD+=(--embd-dir "$EMBED_DIR")
    [[ -n "$TAESD" ]]           && CMD+=(--taesd "$TAESD")

    CMD+=(--steps "$STEPS" --cfg-scale "$CFG_SCALE")
    CMD+=(--sampling-method "$SAMPLER")
    CMD+=(-W "$WIDTH" -H "$HEIGHT")
    CMD+=(--seed "$SEED")

    [[ "$THREADS" != "-1" ]]        && CMD+=(-t "$THREADS")
    [[ "$DIFFUSION_FA" == "on" ]]   && CMD+=(--diffusion-fa)
    [[ "$OFFLOAD_TO_CPU" == "on" ]] && CMD+=(--offload-to-cpu)
    [[ "$VAE_ON_CPU" == "on" ]]     && CMD+=(--vae-on-cpu)
    [[ "$CLIP_ON_CPU" == "on" ]]    && CMD+=(--clip-on-cpu)

    if [[ -n "$EXTRA_ARGS" && "$EXTRA_ARGS" != "none" ]]; then
        # shellcheck disable=SC2206
        CMD+=($EXTRA_ARGS)
    fi
}

# ============================================================================
#  DRY RUN / RUN
# ============================================================================

dry_run() {
    build_command
    apply_gpu_visibility "$BACKEND" "$VISIBLE_DEVICES"
    echo
    divider
    echo -e "  ${BOLD}Environment:${NC}"
    echo -e "    GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES}"
    echo -e "    RADV_DEBUG=${RADV_DEBUG}"
    [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]      && echo -e "    HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    [[ -n "${GGML_VK_VISIBLE_DEVICES:-}" ]]  && echo -e "    GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES}"
    echo
    if webui_active; then
        echo -e "  ${BOLD}Web UI:${NC} ${SD_FRONTEND_HTML}"
    else
        echo -e "  ${BOLD}Web UI:${NC} ${DIM}disabled (placeholder page at /)${NC}"
    fi
    echo
    echo -e "  ${BOLD}Command:${NC}\n"
    echo -n "  "
    for i in "${!CMD[@]}"; do
        if [[ $i -eq 0 ]]; then
            echo -n "${CMD[$i]}"
        elif [[ "${CMD[$i]}" == -* ]]; then
            printf " \\\\\n    %s" "${CMD[$i]}"
        else
            echo -n " ${CMD[$i]}"
        fi
    done
    echo -e "\n"
    divider
    read -rp "  Press Enter to continue..."
}

run_server() {
    build_command
    apply_gpu_visibility "$BACKEND" "$VISIBLE_DEVICES"
    echo
    divider
    echo -e "  ${GREEN}${BOLD}Starting sd-server (${BACKEND})...${NC}"
    if webui_active; then
        echo -e "  ${DIM}Web UI:  http://${HOST}:${PORT}/${NC}"
    else
        echo -e "  ${DIM}API:     http://${HOST}:${PORT}/  ${YELLOW}(placeholder, no UI built)${NC}"
        echo -e "  ${DIM}         Build UI: ./build_tools.sh sd ${BACKEND}${NC}"
    fi
    echo -e "  ${DIM}Outputs: ${OUTPUT_DIR}${NC}"
    [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]     && \
        echo -e "  ${DIM}HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}${NC}"
    [[ -n "${GGML_VK_VISIBLE_DEVICES:-}" ]] && \
        echo -e "  ${DIM}GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES}${NC}"
    echo -e "  ${DIM}Ctrl+C to stop${NC}"
    divider
    echo
    exec "${CMD[@]}"
}

# ============================================================================
#  MAIN
# ============================================================================

main() {
    local bin
    bin="$(current_binary)"
    if [[ ! -x "$bin" ]]; then
        echo -e "  ${RED}Error: sd-server (${BACKEND}) not found at ${bin}${NC}"
        echo
        echo "  Build with: ./build_tools.sh sd ${BACKEND}"
        exit 1
    fi
    [[ ! -d "$MODEL_DIR" ]] && {
        echo -e "  ${RED}Error: ${MODEL_DIR} not found${NC}"
        echo -e "  ${DIM}Create it and put model dirs inside, or set SD_MODEL_DIR${NC}"
        exit 1
    }

    # Warn (don't fail) if SERVE_WEBUI=on but the file is missing
    if [[ "$SERVE_WEBUI" == "on" && ! -f "$SD_FRONTEND_HTML" ]]; then
        echo -e "  ${YELLOW}!${NC} Web UI explicitly enabled but ${SD_FRONTEND_HTML} not found"
        echo -e "  ${DIM}  Build it with: ./build_tools.sh sd ${BACKEND}${NC}"
        echo -e "  ${DIM}  Continuing without Web UI...${NC}"
        sleep 1.5
    fi

    select_model
    configure_params
}

main "$@"
