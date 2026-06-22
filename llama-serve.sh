#!/bin/bash
# ============================================================================
#  llama-serve.sh -- Interactive launcher for llama-server
#  Hardware: 2x AMD Radeon AI PRO R9700 (64GB total VRAM, gfx1201, RDNA4)
#            AMD 7950X (32 threads) / 64GB DDR5
#  Backends: ROCm/HIP (primary) + Vulkan (fallback)
#
#  Presets are self-contained: they store the MODEL_PATH (and optional vision
#  MMPROJ) alongside every runtime flag, so a preset fully describes one
#  servable model. Switch models by simply picking a preset.
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared RDNA4 env + visibility helper
# shellcheck source=lib/rdna4-env.sh
source "$SCRIPT_DIR/lib/rdna4-env.sh"

# -- Paths ------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-/srv/models}"
LLAMA_ROCM="${LLAMA_ROCM:-/srv/llama/llama.cpp-rocm/build/bin/llama-server}"
LLAMA_VULKAN="${LLAMA_VULKAN:-/srv/llama/llama.cpp-vulkan/build/bin/llama-server}"
PRESETS_DIR="${PRESETS_DIR:-/srv/llama/presets}"

# -- Model (lives in the preset) -------------------------------------------
SELECTED_MODEL=""        # working var used everywhere
MODEL_PATH=""            # persisted form of SELECTED_MODEL (kept in sync)
ALIAS=""                 # --alias served name (what LiteLLM / the orchestrator call)

# -- Defaults ---------------------------------------------------------------
BACKEND="rocm"           # rocm | vulkan
HOST="0.0.0.0"
PORT="8080"
GPU_LAYERS="99"
CTX_SIZE="8192"
PREDICT="-1"
TEMP="0.6"
TOP_K="20"
TOP_P="0.95"
MIN_P="0"
PRESENCE_PENALTY="1.5"
REPEAT_PENALTY="1.0"
THREADS="-1"
BATCH_SIZE="2048"
UBATCH_SIZE="512"        # physical micro-batch; MUST be >=512 for vision
                         # (an image tokenizes to several hundred tokens; a
                         #  smaller ubatch makes llama.cpp assert mid-inference)
CACHE_TYPE_K="f16"       # -ctk : keep K/V symmetric so HIP fused flash-attn
CACHE_TYPE_V="f16"       # -ctv : engages. f16 = lossless; we have the VRAM.
FLASH_ATTN="on"          # Caveat: ROCm flash-attn is improving but still
                         # slower than CUDA's; try "off" if you see issues.
JINJA="yes"
REASONING_FORMAT=""
MMPROJ=""                # optional vision projector (*mmproj*.gguf)
MMPROJ_OFFLOAD="on"      # on = vision encoder on GPU (llama.cpp default, fast).
                         # off = --no-mmproj-offload -> encoder on CPU. Slower
                         # image encoding, but a known workaround for the gfx1201
                         # HIP bug where mmproj pins the GPU at full clock / never
                         # idles. Only affects image prompt-processing, not TG.
MTP="off"                # on = --spec-type draft-mtp : self-speculative decoding
                         # via the model's built-in MTP head (NO separate draft
                         # model). Needs an MTP-head GGUF + llama.cpp >= b9180.
                         # Forces f16 KV (quantized KV tanks draft acceptance).
SPEC_DRAFT_N_MAX="2"     # --spec-draft-n-max : draft tokens/step. 2 is safest for
                         # the 35B-A3B MoE; n-max 3 can change committed output.
EXTRA_ARGS=""
SYSTEM_PROMPT=""

# -- Multi-GPU --------------------------------------------------------------
# split-mode: layer (default, lowest overhead, sequential layer placement)
#             row   (tensor-parallel over rows; better with fast PCIe Gen5)
#             none  (single-GPU only; combine with HIP_VISIBLE_DEVICES)
SPLIT_MODE="layer"
TENSOR_SPLIT=""          # e.g. "0.5,0.5" or "1,1" for explicit ratios
MAIN_GPU="0"             # which device gets the small/host tensors
VISIBLE_DEVICES=""       # e.g. "0" or "1" to restrict to one R9700

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
    echo -e "\n  ${CYAN}${BOLD}llama-server launcher${NC} ${DIM}(2x R9700, 64GB VRAM)${NC}\n"
}

divider() {
    echo -e "${DIM}  ---------------------------------------------------------${NC}"
}

# -- Input helper -----------------------------------------------------------
update_param() {
    local var_name="$1"
    local prompt_text="$2"
    local current_val="${!var_name:-none}"
    local input
    read -rp "  $prompt_text [$current_val]: " input
    [[ -n "$input" ]] && printf -v "$var_name" "%s" "$input"
}

# -- Resolve current binary based on backend --------------------------------
current_binary() {
    case "$BACKEND" in
        rocm)   echo "$LLAMA_ROCM" ;;
        vulkan) echo "$LLAMA_VULKAN" ;;
        *)      echo "$LLAMA_ROCM" ;;
    esac
}

# Derive a default alias from the model filename if none set.
default_alias() {
    [[ -n "$ALIAS" ]] && { echo "$ALIAS"; return; }
    [[ -n "$SELECTED_MODEL" ]] && basename "$SELECTED_MODEL" .gguf || echo ""
}

# ============================================================================
#  PRESET SAVE / LOAD
#  MODEL_PATH + MMPROJ + ALIAS are persisted so a preset is self-contained.
# ============================================================================

mkdir -p "$PRESETS_DIR"

PARAM_LIST=(MODEL_PATH ALIAS MMPROJ MMPROJ_OFFLOAD MTP SPEC_DRAFT_N_MAX
            BACKEND HOST PORT GPU_LAYERS CTX_SIZE PREDICT TEMP TOP_K TOP_P MIN_P
            PRESENCE_PENALTY REPEAT_PENALTY THREADS BATCH_SIZE UBATCH_SIZE
            CACHE_TYPE_K CACHE_TYPE_V
            FLASH_ATTN JINJA REASONING_FORMAT EXTRA_ARGS SYSTEM_PROMPT
            SPLIT_MODE TENSOR_SPLIT MAIN_GPU VISIBLE_DEVICES)

# Write every PARAM to a .conf (after syncing model path + alias).
write_preset() {
    local file="$1"
    MODEL_PATH="$SELECTED_MODEL"
    [[ -z "$ALIAS" ]] && ALIAS="$(default_alias)"
    : > "$file"
    for p in "${PARAM_LIST[@]}"; do
        echo "${p}=${!p}" >> "$file"
    done
}

save_preset() {
    read -rp "  Preset name: " pname
    [[ -z "$pname" ]] && return
    pname=$(echo "$pname" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    local file="$PRESETS_DIR/${pname}.conf"
    write_preset "$file"
    echo -e "  ${GREEN}Saved -> ${file}${NC}"
    sleep 0.7
}

load_preset_file() {
    local file="$1"
    while IFS='=' read -r key val; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        printf -v "$key" "%s" "$val" 2>/dev/null || true
    done < "$file"
    # The model path travels with the preset.
    [[ -n "${MODEL_PATH:-}" ]] && SELECTED_MODEL="$MODEL_PATH"
}

# List presets with the model each one carries (used by the start menu).
preset_model_label() {
    local file="$1" mp=""
    mp=$(grep -m1 '^MODEL_PATH=' "$file" 2>/dev/null | cut -d= -f2-) || true
    [[ -n "$mp" ]] && basename "$mp" || echo "?"
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
        echo -e "  ${GREEN}$((i+1)))${NC} ${name}  ${DIM}[$(preset_model_label "${files[$i]}")]${NC}"
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
#  MODEL-SPECIFIC PRESETS
#  Tuned for: 2x R9700 (64GB total VRAM) / 7950X / 64GB DDR5
#  Strategy: most models comfortably single-GPU; only spread across both
#            GPUs when weights+KV+activations exceed ~30GB on one card.
# ============================================================================

apply_model_preset() {
    local model_file
    model_file=$(basename "$SELECTED_MODEL")

    HOST="0.0.0.0"; PORT="8080"; GPU_LAYERS="99"; PREDICT="-1"
    THREADS="-1"; BATCH_SIZE="2048"; UBATCH_SIZE="512"
    CACHE_TYPE_K="f16"; CACHE_TYPE_V="f16"
    FLASH_ATTN="on"; JINJA="yes"; MMPROJ=""; MMPROJ_OFFLOAD="on"; EXTRA_ARGS=""
    MTP="off"; SPEC_DRAFT_N_MAX="2"
    SYSTEM_PROMPT=""; REASONING_FORMAT=""
    SPLIT_MODE="layer"; TENSOR_SPLIT=""; MAIN_GPU="0"; VISIBLE_DEVICES=""

    case "$model_file" in
        *[Bb]onsai*8[Bb]*)
            CTX_SIZE="32768"; TEMP="0.7"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT="none"
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            echo -e "  ${CYAN}Preset: Bonsai 8B -- single GPU0, 32K ctx${NC}" ;;

        *[Gg]emma*4*31[Bb]*)
            CTX_SIZE="32768"; TEMP="0.7"; TOP_K="40"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT="none"
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            echo -e "  ${CYAN}Preset: Gemma 4 31B -- single GPU0, 32K ctx${NC}" ;;

        *[Gg]emma*[Pp][Rr][Ii][Ss][Mm]*)
            CTX_SIZE="16384"; TEMP="0.7"; TOP_K="40"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT="none"
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            local prism_dir prism_mmproj
            prism_dir=$(dirname "$SELECTED_MODEL")
            prism_mmproj=$(find "$prism_dir" -maxdepth 1 -name "*mmproj*" -type f 2>/dev/null | head -1) || true
            [[ -n "$prism_mmproj" ]] && MMPROJ="$prism_mmproj" && UBATCH_SIZE="512"
            echo -e "  ${CYAN}Preset: Gemma PRISM PRO -- single GPU0, vision${NC}" ;;

        *[Pp]hi*4*[Qq]8*)
            CTX_SIZE="32768"; TEMP="0.6"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT="none"
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            echo -e "  ${CYAN}Preset: Phi-4 Q8 -- single GPU0, 32K ctx${NC}" ;;

        *[Qq]wen3.5*27[Bb]*[Uu]ncensored*)
            CTX_SIZE="32768"; TEMP="0.7"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="1.5"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT="none"
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            SYSTEM_PROMPT="You are a helpful assistant. /no_think"
            echo -e "  ${CYAN}Preset: Qwen3.5-27B Uncensored -- single GPU, no_think${NC}" ;;

        *[Qq]wopus*27[Bb]*)
            CTX_SIZE="32768"; TEMP="0.6"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="1.0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT=""
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            echo -e "  ${CYAN}Preset: Qwopus3.5-27B -- single GPU0, 32K ctx${NC}" ;;

        *[Qq]wen3.5*27[Bb]*)
            CTX_SIZE="32768"; TEMP="0.6"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="1.5"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT=""
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            echo -e "  ${CYAN}Preset: Qwen3.5-27B -- single GPU0, 32K ctx${NC}" ;;

        *[Qq]wen3*35[Bb]*[Aa]3[Bb]*)
            CTX_SIZE="65536"; TEMP="0.6"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="1.5"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT=""
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            echo -e "  ${CYAN}Preset: Qwen 35B-A3B MoE -- single GPU0, 64K ctx${NC}" ;;

        *[Qq]wen3*[Cc]oder*30[Bb]*)
            CTX_SIZE="65536"; TEMP="0.2"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.1"
            REASONING_FORMAT="none"
            VISIBLE_DEVICES="0"; SPLIT_MODE="none"
            SYSTEM_PROMPT="You are an expert programmer. Write concise, correct code."
            echo -e "  ${CYAN}Preset: Qwen3-Coder-30B -- single GPU0, 64K ctx${NC}" ;;

        *[Ll]lama*70[Bb]*|*[Ll]3*70[Bb]*)
            CTX_SIZE="16384"; TEMP="0.7"; TOP_K="40"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT="none"
            SPLIT_MODE="layer"; TENSOR_SPLIT="1,1"
            echo -e "  ${CYAN}Preset: Llama 70B -- both GPUs, layer split, 16K ctx${NC}" ;;

        *[Qq]wen*72[Bb]*)
            CTX_SIZE="32768"; TEMP="0.6"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="1.0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT=""
            SPLIT_MODE="layer"; TENSOR_SPLIT="1,1"
            echo -e "  ${CYAN}Preset: Qwen 72B -- both GPUs, 32K ctx${NC}" ;;

        *[Mm]istral*[Ll]arge*|*123[Bb]*)
            CTX_SIZE="8192"; TEMP="0.7"; TOP_K="40"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT="none"
            SPLIT_MODE="layer"; TENSOR_SPLIT="1,1"
            echo -e "  ${CYAN}Preset: Mistral Large 123B -- both GPUs, 8K ctx (tight)${NC}" ;;

        *[Dd]eep[Ss]eek*|*100[Bb]*|*[Mm]ix*[Mm]oE*)
            CTX_SIZE="16384"; TEMP="0.6"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT=""
            SPLIT_MODE="row"; TENSOR_SPLIT="1,1"
            echo -e "  ${CYAN}Preset: Large MoE -- both GPUs, row split, 16K ctx${NC}" ;;

        *)
            CTX_SIZE="16384"; TEMP="0.7"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
            PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"
            REASONING_FORMAT=""
            echo -e "  ${DIM}No specific preset, using defaults${NC}" ;;
    esac

    # MTP-head variants (…-MTP-…gguf) carry a built-in draft head: enable
    # self-speculative decoding by default and pin KV to f16 for acceptance.
    if [[ "$model_file" == *MTP* || "$model_file" == *mtp* ]]; then
        MTP="on"; SPEC_DRAFT_N_MAX="2"; CACHE_TYPE_K="f16"; CACHE_TYPE_V="f16"
        echo -e "  ${CYAN}  + MTP head detected -- self-speculative draft on (n-max 2, f16 KV)${NC}"
    fi
}

# ============================================================================
#  VISION MODEL (mmproj) SELECTION
#  Optional. Lists *mmproj*.gguf candidates near the model and across the
#  model tree, lets you pick one (or none). Stored in the preset as MMPROJ.
# ============================================================================

select_vision_model() {
    local mdir
    mdir=$(dirname "$SELECTED_MODEL")

    local cands=()
    shopt -s nullglob
    local f
    for f in "$mdir"/*mmproj*.gguf "$MODEL_DIR"/*mmproj*.gguf "$MODEL_DIR"/*/*mmproj*.gguf; do
        cands+=("$f")
    done
    shopt -u nullglob

    # de-duplicate while preserving order
    local uniq=() seen c x dup
    for c in "${cands[@]}"; do
        dup=0
        for x in "${uniq[@]}"; do [[ "$x" == "$c" ]] && dup=1 && break; done
        [[ $dup -eq 0 ]] && uniq+=("$c")
    done
    cands=("${uniq[@]}")

    if [[ ${#cands[@]} -eq 0 ]]; then
        [[ -n "$MMPROJ" ]] && echo -e "  ${DIM}Vision projector (from preset): $(basename "$MMPROJ")${NC}" \
                           || echo -e "  ${DIM}No vision projector (*mmproj*.gguf) found near this model${NC}"
        return
    fi

    echo -e "\n  ${BOLD}Vision projector (optional)${NC} ${DIM}-- enables image input${NC}"
    divider
    echo -e "  ${GREEN}0)${NC} none ${DIM}(text only)${NC}"
    local i
    for i in "${!cands[@]}"; do
        local tag=""
        [[ "${cands[$i]}" == "$MMPROJ" ]] && tag=" ${CYAN}(current)${NC}"
        [[ "$(dirname "${cands[$i]}")" == "$mdir" ]] && tag="$tag ${DIM}(same dir)${NC}"
        echo -e "  ${GREEN}$((i+1)))${NC} $(basename "${cands[$i]}")${tag}"
    done
    echo
    # default: keep current MMPROJ if set, else the same-dir candidate, else none
    local def="0"
    for i in "${!cands[@]}"; do
        if [[ -n "$MMPROJ" && "${cands[$i]}" == "$MMPROJ" ]]; then def="$((i+1))"; break; fi
    done
    if [[ "$def" == "0" ]]; then
        for i in "${!cands[@]}"; do
            if [[ "$(dirname "${cands[$i]}")" == "$mdir" ]]; then def="$((i+1))"; break; fi
        done
    fi

    local choice
    read -rp "  Select vision projector [0-${#cands[@]}] [${def}]: " choice
    [[ -z "$choice" ]] && choice="$def"

    if [[ "$choice" == "0" ]]; then
        MMPROJ=""
        echo -e "  ${DIM}Vision disabled${NC}"
    elif [[ "$choice" -ge 1 && "$choice" -le ${#cands[@]} ]] 2>/dev/null; then
        MMPROJ="${cands[$((choice-1))]}"
        # vision correctness: image token batch needs ubatch >= 512
        if [[ "$UBATCH_SIZE" -lt 512 ]] 2>/dev/null; then
            UBATCH_SIZE="512"
            echo -e "  ${DIM}(ubatch raised to 512 for image tokens)${NC}"
        fi
        echo -e "  ${GREEN}OK${NC} Vision enabled: $(basename "$MMPROJ")"
    else
        echo -e "  ${RED}Invalid -- leaving vision unchanged${NC}"
    fi
    sleep 0.4
}

# ============================================================================
#  MODEL SELECTION (browse)
# ============================================================================

# Populate MODELS[] / display[] with every model under $MODEL_DIR, at any nesting
# depth (HF repos land in $MODEL_DIR/<org>/<repo>/…). Rules:
#  - each individual .gguf is its own selectable entry (a repo folder with several
#    quants/variants lists all of them)
#  - a genuine multi-shard model (…-00001-of-00005.gguf, …) collapses to ONE entry
#    at shard 1 (llama.cpp loads the rest automatically)
#  - *mmproj* projectors are skipped here (offered later as a vision model)
scan_models() {
    local f name base total sz parent loc
    while IFS= read -r f; do
        name=$(basename "$f")
        [[ "$name" == *mmproj* ]] && continue
        parent=$(dirname "$f")
        if [[ "$parent" == "$MODEL_DIR" ]]; then
            loc=""
        else
            local pb; pb=$(basename "$parent")
            [[ ${#pb} -gt 34 ]] && pb="${pb:0:31}…"
            loc=" ${DIM}[${pb}]${NC}"
        fi
        if [[ "$name" =~ ^(.+)-[0-9][0-9][0-9][0-9][0-9]-of-([0-9][0-9][0-9][0-9][0-9])\.gguf$ ]]; then
            base="${BASH_REMATCH[1]}"; total="${BASH_REMATCH[2]}"
            [[ "$name" == *-00001-of-* ]] || continue     # only surface shard 1
            sz=$(du -ch "$parent/${base}"-*-of-*.gguf 2>/dev/null | tail -1 | cut -f1) || sz="?"
            MODELS+=("$f")
            display+=("${base}.gguf ${DIM}(${sz}, $((10#$total)) shards)${NC}${loc}")
        else
            sz=$(du -h "$f" 2>/dev/null | cut -f1) || sz="?"
            MODELS+=("$f")
            display+=("${name} ${DIM}(${sz})${NC}${loc}")
        fi
    done < <(find -L "$MODEL_DIR" -maxdepth 4 -type f -name '*.gguf' 2>/dev/null | sort)
}

select_model() {
    header
    echo -e "  ${BOLD}Step 1: Select a model${NC}"
    divider
    echo

    MODELS=()
    local display=()
    scan_models

    if [[ ${#MODELS[@]} -eq 0 ]]; then
        echo -e "  ${RED}No .gguf models found in ${MODEL_DIR}${NC}"
        exit 1
    fi

    for i in "${!MODELS[@]}"; do
        echo -e "  ${GREEN}$((i+1)))${NC} ${display[$i]}"
    done

    echo
    read -rp "  Select model [1-${#MODELS[@]}] (or 'q' to quit): " choice
    [[ "${choice,,}" == "q" ]] && { echo -e "  ${YELLOW}Bye!${NC}"; exit 0; }

    if [[ "$choice" -ge 1 && "$choice" -le ${#MODELS[@]} ]] 2>/dev/null; then
        SELECTED_MODEL="${MODELS[$((choice-1))]}"
        MODEL_PATH="$SELECTED_MODEL"
        ALIAS="$(basename "$SELECTED_MODEL" .gguf)"
        echo -e "\n  ${GREEN}OK${NC} $(basename "$SELECTED_MODEL")"
    else
        echo -e "  ${RED}Invalid selection${NC}"; exit 1
    fi

    # Heuristic defaults first, then offer to pick a vision projector,
    # then offer the saved per-model default (which overrides both).
    apply_model_preset
    select_vision_model

    local model_preset="$PRESETS_DIR/$(basename "$SELECTED_MODEL" .gguf).conf"
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
#  START MENU -- launch from a preset (carries its model) or browse models
# ============================================================================

choose_start() {
    local files=()
    shopt -s nullglob
    files=("$PRESETS_DIR"/*.conf)
    shopt -u nullglob

    header
    echo -e "  ${BOLD}Step 0: Start${NC}"
    divider
    echo
    if [[ ${#files[@]} -gt 0 ]]; then
        echo -e "  ${BOLD}Launch from a preset${NC} ${DIM}(model + flags travel together)${NC}"
        local i
        for i in "${!files[@]}"; do
            echo -e "  ${GREEN}$((i+1)))${NC} $(basename "${files[$i]}" .conf)  ${DIM}[$(preset_model_label "${files[$i]}")]${NC}"
        done
        echo
    fi
    echo -e "  ${GREEN}b)${NC} Browse models in ${MODEL_DIR}"
    echo -e "  ${GREEN}q)${NC} Quit"
    echo
    read -rp "  > " choice

    case "${choice,,}" in
        q) echo -e "  ${YELLOW}Bye!${NC}"; exit 0 ;;
        b|"") return ;;     # fall through to browse
        *)
            if [[ "$choice" -ge 1 && "$choice" -le ${#files[@]} ]] 2>/dev/null; then
                load_preset_file "${files[$((choice-1))]}"
                echo -e "  ${GREEN}Loaded preset:${NC} $(basename "${files[$((choice-1))]}" .conf)"
                if [[ -z "$SELECTED_MODEL" || ! -f "$SELECTED_MODEL" ]]; then
                    echo -e "  ${YELLOW}Preset has no usable MODEL_PATH -- browse instead${NC}"
                    SELECTED_MODEL=""; sleep 0.8; return
                fi
                sleep 0.6
            else
                echo -e "  ${RED}Invalid${NC}"; sleep 0.4; choose_start
            fi ;;
    esac
}

# ============================================================================
#  PARAMETER MENU
# ============================================================================

configure_params() {
    while true; do
        header
        echo -e "  ${BOLD}Step 2: Configure parameters${NC}"
        echo -e "  ${DIM}Model: $(basename "$SELECTED_MODEL")${NC}"
        echo -e "  ${DIM}Backend: ${BACKEND}  |  Binary: $(current_binary)${NC}"
        divider
        echo
        echo -e "  ${BOLD}Model & Network${NC}"
        echo -e "  ${GREEN} 1)${NC} Host/Port      ${BOLD}${HOST}:${PORT}${NC}"
        echo -e "  ${GREEN} 2)${NC} GPU layers     ${BOLD}${GPU_LAYERS}${NC} ${DIM}(-ngl)${NC}"
        echo -e "  ${GREEN} 3)${NC} Context size   ${BOLD}${CTX_SIZE}${NC} ${DIM}(-c)${NC}"
        echo -e "  ${GREEN} 4)${NC} Max predict    ${BOLD}${PREDICT}${NC} ${DIM}(-n, -1=unlimited)${NC}"
        echo -e "  ${GREEN} 5)${NC} CPU threads    ${BOLD}${THREADS}${NC} ${DIM}(-t, -1=auto)${NC}"
        echo -e "  ${GREEN} 6)${NC} Batch / ubatch ${BOLD}${BATCH_SIZE} / ${UBATCH_SIZE}${NC} ${DIM}(-b/-ub; ub>=512 for vision)${NC}"
        echo -e "  ${GREEN} 7)${NC} Flash attn     ${BOLD}${FLASH_ATTN}${NC} ${DIM}(on/off; ROCm: try off if slow)${NC}"
        echo -e "  ${GREEN}18)${NC} KV cache type  ${BOLD}${CACHE_TYPE_K}/${CACHE_TYPE_V}${NC} ${DIM}(-ctk/-ctv; symmetric=fused FA)${NC}"
        echo -e "  ${GREEN}19)${NC} Alias          ${BOLD}${ALIAS:-auto}${NC} ${DIM}(--alias; served model name)${NC}"
        echo
        echo -e "  ${BOLD}Multi-GPU (2x R9700)${NC}"
        echo -e "  ${GREEN}m1)${NC} Backend        ${BOLD}${BACKEND}${NC} ${DIM}(rocm/vulkan)${NC}"
        echo -e "  ${GREEN}m2)${NC} Split mode     ${BOLD}${SPLIT_MODE}${NC} ${DIM}(layer/row/none)${NC}"
        echo -e "  ${GREEN}m3)${NC} Tensor split   ${BOLD}${TENSOR_SPLIT:-auto}${NC} ${DIM}(e.g. 1,1)${NC}"
        echo -e "  ${GREEN}m4)${NC} Main GPU       ${BOLD}${MAIN_GPU}${NC} ${DIM}(-mg)${NC}"
        echo -e "  ${GREEN}m5)${NC} Visible devs   ${BOLD}${VISIBLE_DEVICES:-all}${NC} ${DIM}(HIP_VISIBLE_DEVICES)${NC}"
        echo
        echo -e "  ${BOLD}Sampling${NC}"
        echo -e "  ${GREEN} 8)${NC} Temperature    ${BOLD}${TEMP}${NC}"
        echo -e "  ${GREEN} 9)${NC} Top-K / Top-P  ${BOLD}${TOP_K} / ${TOP_P}${NC}"
        echo -e "  ${GREEN}10)${NC} Min-P          ${BOLD}${MIN_P}${NC}"
        echo -e "  ${GREEN}11)${NC} Pres. penalty  ${BOLD}${PRESENCE_PENALTY}${NC}"
        echo -e "  ${GREEN}12)${NC} Repeat penalty ${BOLD}${REPEAT_PENALTY}${NC}"
        echo
        echo -e "  ${BOLD}Features${NC}"
        echo -e "  ${GREEN}13)${NC} Jinja          ${BOLD}${JINJA}${NC}"
        echo -e "  ${GREEN}14)${NC} Reasoning fmt  ${BOLD}${REASONING_FORMAT:-auto}${NC}"
        echo -e "  ${GREEN}15)${NC} System prompt  ${BOLD}${SYSTEM_PROMPT:-none}${NC}"
        echo -e "  ${GREEN}16)${NC} Vision mmproj  ${BOLD}${MMPROJ:-none}${NC} ${DIM}(--mmproj)${NC}"
        echo -e "  ${GREEN}20)${NC} mmproj offload ${BOLD}${MMPROJ_OFFLOAD}${NC} ${DIM}($([[ "$MMPROJ_OFFLOAD" == off ]] && echo "CPU, --no-mmproj-offload" || echo "GPU"))${NC}"
        echo -e "  ${GREEN}17)${NC} Extra args     ${BOLD}${EXTRA_ARGS:-none}${NC}"
        echo
        echo -e "  ${BOLD}Speculative decoding (MTP)${NC}"
        echo -e "  ${GREEN}21)${NC} MTP draft      ${BOLD}${MTP}${NC} ${DIM}(--spec-type draft-mtp; needs MTP-head GGUF; forces f16 KV)${NC}"
        echo -e "  ${GREEN}22)${NC} draft n-max    ${BOLD}${SPEC_DRAFT_N_MAX}${NC} ${DIM}(--spec-draft-n-max; 2 safest for MoE)${NC}"
        echo
        divider
        echo -e "  ${BOLD}Presets${NC}"
        echo -e "  ${YELLOW} p1)${NC} Chat       ${DIM}-- temp 0.7, no penalties, no_think${NC}"
        echo -e "  ${YELLOW} p2)${NC} Reasoning  ${DIM}-- temp 0.6, ctx 32K, thinking on${NC}"
        echo -e "  ${YELLOW} p3)${NC} Coding     ${DIM}-- temp 0.2, repeat 1.1, no_think${NC}"
        echo -e "  ${YELLOW} p4)${NC} Creative   ${DIM}-- temp 0.9, top-k 40, thinking on${NC}"
        echo
        echo -e "  ${BOLD}Config${NC}"
        echo -e "  ${YELLOW}  s)${NC} Save current config ${DIM}(includes model + vision)${NC}"
        echo -e "  ${YELLOW}  l)${NC} Load saved config"
        echo -e "  ${YELLOW} sm)${NC} Save as model default ${DIM}(auto-loads next time)${NC}"
        echo -e "  ${YELLOW}  v)${NC} List devices ${DIM}(llama-server --list-devices)${NC}"
        echo
        divider
        echo -e "  ${GREEN} r)${NC}  Run server"
        echo -e "  ${GREEN} d)${NC}  Dry run (show command)"
        echo -e "  ${GREEN} q)${NC}  Quit"
        echo
        read -rp "  > " opt

        case "$opt" in
            1)  update_param HOST "Host"; update_param PORT "Port" ;;
            2)  update_param GPU_LAYERS "GPU layers (-ngl)" ;;
            3)  update_param CTX_SIZE "Context size (-c)" ;;
            4)  update_param PREDICT "Max predict (-n)" ;;
            5)  update_param THREADS "CPU threads (-t)" ;;
            6)  update_param BATCH_SIZE "Batch size (-b)"; update_param UBATCH_SIZE "Micro-batch (-ub)" ;;
            7)  update_param FLASH_ATTN "Flash attention (on/off)" ;;
            18) update_param CACHE_TYPE_K "KV cache K type (f16/q8_0/q4_0)"
                update_param CACHE_TYPE_V "KV cache V type (f16/q8_0/q4_0)" ;;
            19) update_param ALIAS "Served alias (--alias)" ;;
            8)  update_param TEMP "Temperature" ;;
            9)  update_param TOP_K "Top-K"; update_param TOP_P "Top-P" ;;
            10) update_param MIN_P "Min-P" ;;
            11) update_param PRESENCE_PENALTY "Presence penalty" ;;
            12) update_param REPEAT_PENALTY "Repeat penalty" ;;
            13) update_param JINJA "Jinja (yes/no)" ;;
            14) update_param REASONING_FORMAT "Reasoning format (auto/none/deepseek)" ;;
            15) read -rp "  System prompt: " SYSTEM_PROMPT ;;
            16) select_vision_model ;;
            20) [[ "$MMPROJ_OFFLOAD" == "on" ]] && MMPROJ_OFFLOAD="off" || MMPROJ_OFFLOAD="on"
                echo -e "  ${GREEN}OK${NC} mmproj offload: ${MMPROJ_OFFLOAD} ($([[ "$MMPROJ_OFFLOAD" == off ]] && echo CPU || echo GPU))"
                sleep 0.4 ;;
            17) update_param EXTRA_ARGS "Extra args" ;;
            21) [[ "$MTP" == "on" ]] && MTP="off" || MTP="on"
                if [[ "$MTP" == "on" ]]; then
                    CACHE_TYPE_K="f16"; CACHE_TYPE_V="f16"
                    echo -e "  ${GREEN}OK${NC} MTP on (--spec-type draft-mtp) -- KV forced to f16"
                    [[ "$(basename "$SELECTED_MODEL")" != *MTP* && "$(basename "$SELECTED_MODEL")" != *mtp* ]] && \
                      echo -e "  ${YELLOW}note:${NC} this model's name has no 'MTP' tag -- it may have no draft head; llama-server will error if so"
                else
                    echo -e "  ${GREEN}OK${NC} MTP off"
                fi
                sleep 0.7 ;;
            22) update_param SPEC_DRAFT_N_MAX "MTP draft n-max (2 safest; 3 can change output)" ;;

            m1) update_param BACKEND "Backend (rocm/vulkan)" ;;
            m2) update_param SPLIT_MODE "Split mode (layer/row/none)" ;;
            m3) update_param TENSOR_SPLIT "Tensor split (e.g. 1,1 or 0.6,0.4)" ;;
            m4) update_param MAIN_GPU "Main GPU index" ;;
            m5) update_param VISIBLE_DEVICES "Visible devices (e.g. 0 or 0,1)" ;;

            p1) TEMP="0.7"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
                PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.0"; CTX_SIZE="16384"
                PREDICT="-1"; REASONING_FORMAT="none"
                SYSTEM_PROMPT="You are a helpful assistant. /no_think"
                echo -e "  ${GREEN}OK${NC} Chat preset"; sleep 0.5 ;;
            p2) TEMP="0.6"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
                PRESENCE_PENALTY="1.5"; REPEAT_PENALTY="1.0"; CTX_SIZE="32768"
                PREDICT="-1"; REASONING_FORMAT=""; SYSTEM_PROMPT=""
                echo -e "  ${GREEN}OK${NC} Reasoning preset"; sleep 0.5 ;;
            p3) TEMP="0.2"; TOP_K="20"; TOP_P="0.95"; MIN_P="0"
                PRESENCE_PENALTY="0"; REPEAT_PENALTY="1.1"; CTX_SIZE="32768"
                PREDICT="-1"; REASONING_FORMAT="none"
                SYSTEM_PROMPT="You are an expert programmer. Write concise, correct code. /no_think"
                echo -e "  ${GREEN}OK${NC} Coding preset"; sleep 0.5 ;;
            p4) TEMP="0.9"; TOP_K="40"; TOP_P="0.95"; MIN_P="0"
                PRESENCE_PENALTY="0.5"; REPEAT_PENALTY="1.0"; CTX_SIZE="16384"
                PREDICT="-1"; REASONING_FORMAT=""; SYSTEM_PROMPT=""
                echo -e "  ${GREEN}OK${NC} Creative preset"; sleep 0.5 ;;

            s)  save_preset ;;
            l)  load_preset_menu ;;
            sm) local mfile="$PRESETS_DIR/$(basename "$SELECTED_MODEL" .gguf).conf"
                write_preset "$mfile"
                echo -e "  ${GREEN}Saved model default -> $(basename "$mfile")${NC}"
                sleep 0.7 ;;
            v)  echo
                "$(current_binary)" --list-devices 2>&1 || true
                echo
                read -rp "  Press Enter to continue..." ;;

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

    # Vision correctness guard: image token batch needs ub >= 512.
    if [[ -n "$MMPROJ" && "$MMPROJ" != "none" ]]; then
        [[ "$UBATCH_SIZE" -lt 512 ]] 2>/dev/null && UBATCH_SIZE="512"
    fi

    # MTP guard: quantized KV collapses draft acceptance -> force f16 both sides.
    if [[ "$MTP" == "on" ]]; then
        CACHE_TYPE_K="f16"; CACHE_TYPE_V="f16"
    fi

    CMD=("$bin")
    CMD+=(-m "$SELECTED_MODEL")
    [[ -n "$ALIAS" ]] && CMD+=(--alias "$ALIAS")
    CMD+=(--host "$HOST" --port "$PORT")
    CMD+=(-ngl "$GPU_LAYERS" -c "$CTX_SIZE" -n "$PREDICT")
    CMD+=(--temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P")
    CMD+=(--presence-penalty "$PRESENCE_PENALTY" --repeat-penalty "$REPEAT_PENALTY")
    CMD+=(-b "$BATCH_SIZE" -ub "$UBATCH_SIZE")
    [[ -n "$CACHE_TYPE_K" ]] && CMD+=(-ctk "$CACHE_TYPE_K")
    [[ -n "$CACHE_TYPE_V" ]] && CMD+=(-ctv "$CACHE_TYPE_V")
    CMD+=(-fa "$FLASH_ATTN")

    if [[ "$MTP" == "on" ]]; then
        CMD+=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_DRAFT_N_MAX")
    fi

    [[ -n "$SPLIT_MODE" ]]   && CMD+=(-sm "$SPLIT_MODE")
    [[ -n "$TENSOR_SPLIT" ]] && CMD+=(-ts "$TENSOR_SPLIT")
    [[ -n "$MAIN_GPU" ]]     && CMD+=(-mg "$MAIN_GPU")

    [[ "$THREADS" != "-1" ]] && CMD+=(-t "$THREADS")
    [[ "$JINJA" == "yes" ]]  && CMD+=(--jinja)

    [[ -n "$REASONING_FORMAT" && "$REASONING_FORMAT" != "auto" ]] && \
        CMD+=(--reasoning-format "$REASONING_FORMAT")
    if [[ -n "$MMPROJ" && "$MMPROJ" != "none" ]]; then
        CMD+=(--mmproj "$MMPROJ")
        [[ "$MMPROJ_OFFLOAD" == "off" ]] && CMD+=(--no-mmproj-offload)
    fi
    [[ -n "$SYSTEM_PROMPT" ]] && \
        CMD+=(--system-prompt "$SYSTEM_PROMPT")

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
    [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]     && echo -e "    HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    [[ -n "${GGML_VK_VISIBLE_DEVICES:-}" ]] && echo -e "    GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES}"
    [[ -n "$MMPROJ" && "$MMPROJ" != "none" ]] && echo -e "    ${CYAN}vision: enabled (${MMPROJ##*/}) -- encoder on $([[ "$MMPROJ_OFFLOAD" == off ]] && echo CPU || echo GPU)${NC}"
    [[ "$MTP" == "on" ]] && echo -e "    ${CYAN}MTP: draft-mtp, n-max=${SPEC_DRAFT_N_MAX} (KV forced f16)${NC}"
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
    echo -e "  ${GREEN}${BOLD}Starting llama-server (${BACKEND})...${NC}"
    echo -e "  ${DIM}Model:   $(basename "$SELECTED_MODEL")${NC}"
    [[ -n "$ALIAS" ]] && echo -e "  ${DIM}Alias:   ${ALIAS}${NC}"
    [[ -n "$MMPROJ" && "$MMPROJ" != "none" ]] && echo -e "  ${DIM}Vision:  ${MMPROJ##*/} (encoder on $([[ "$MMPROJ_OFFLOAD" == off ]] && echo CPU || echo GPU))${NC}"
    [[ "$MTP" == "on" ]] && echo -e "  ${DIM}MTP:     draft-mtp, n-max=${SPEC_DRAFT_N_MAX}${NC}"
    echo -e "  ${DIM}API:     http://${HOST}:${PORT}/v1 (OpenAI-compatible)${NC}"
    [[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && \
        echo -e "  ${DIM}HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}${NC}"
    echo -e "  ${DIM}Ctrl+C to stop${NC}"
    divider
    echo
    exec "${CMD[@]}"
}

# ============================================================================
#  MAIN
# ============================================================================

print_usage() {
    cat <<EOF
Usage:
  llama-serve.sh                      Interactive launcher (browse / pick a preset)
  llama-serve.sh --preset FILE        Load a preset and start the server (headless)
  llama-serve.sh --preset FILE --dry-run   Print the resolved command and exit
  llama-serve.sh -h | --help          This help

Headless mode is for systemd units, e.g.:
  ExecStart=/srv/llama/llama-serve.sh --preset \${PRESETS}
EOF
}

# Non-interactive launch straight from a preset file (model + flags travel in it).
run_preset_headless() {
    local preset_file="$1" action="$2"
    [[ -f "$preset_file" ]] || { echo "llama-serve: preset not found: $preset_file" >&2; exit 1; }
    load_preset_file "$preset_file"
    if [[ -z "${SELECTED_MODEL:-}" || ! -f "$SELECTED_MODEL" ]]; then
        echo "llama-serve: preset has no usable MODEL_PATH (got: '${SELECTED_MODEL:-}')" >&2; exit 1
    fi
    local bin; bin="$(current_binary)"
    [[ -x "$bin" ]] || { echo "llama-serve: llama-server ($BACKEND) not found at $bin" >&2; exit 1; }
    if [[ "$action" == "dry" ]]; then
        build_command
        apply_gpu_visibility "$BACKEND" "$VISIBLE_DEVICES"
        printf '%q ' "${CMD[@]}"; echo
        exit 0
    fi
    run_server   # build_command + apply_gpu_visibility + exec llama-server
}

main() {
    # ---- non-interactive arg parsing ----
    local preset_file="" action="run"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --preset)   [[ $# -ge 2 ]] || { echo "--preset needs a FILE" >&2; exit 2; }
                        preset_file="$2"; shift 2 ;;
            --preset=*) preset_file="${1#*=}"; shift ;;
            --dry-run|-d) action="dry"; shift ;;
            --run)      action="run"; shift ;;
            -h|--help)  print_usage; exit 0 ;;
            *)          echo "llama-serve: unknown argument: $1" >&2; print_usage >&2; exit 2 ;;
        esac
    done
    if [[ -n "$preset_file" ]]; then
        run_preset_headless "$preset_file" "$action"
        return
    fi

    # ---- interactive flow ----
    [[ ! -d "$MODEL_DIR" ]] && { echo -e "  ${RED}Error: ${MODEL_DIR} not found${NC}"; exit 1; }

    choose_start                                   # may load a preset (with its model)
    [[ -z "${SELECTED_MODEL:-}" ]] && select_model # otherwise browse

    local bin
    bin="$(current_binary)"
    if [[ ! -x "$bin" ]]; then
        echo -e "  ${RED}Error: llama-server (${BACKEND}) not found at ${bin}${NC}"
        echo
        echo "  Build with: ./build_tools.sh llama ${BACKEND}"
        exit 1
    fi

    configure_params
}

# Allow sourcing for tests without running the interactive flow.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
