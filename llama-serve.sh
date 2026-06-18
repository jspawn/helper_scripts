#!/bin/bash
# ============================================================================
#  llama-serve.sh -- Interactive launcher for llama-server
#  Hardware: 2x AMD Radeon AI PRO R9700 (64GB total VRAM, gfx1201, RDNA4)
#            AMD 7950X (32 threads) / 64GB DDR5
#  Backends: ROCm/HIP (primary) + Vulkan (fallback)
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
UBATCH_SIZE="512"
FLASH_ATTN="on"          # Caveat: ROCm flash-attn is improving but still
                         # slower than CUDA's; try "off" if you see issues.
JINJA="yes"
REASONING_FORMAT=""
MMPROJ=""
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

# ============================================================================
#  PRESET SAVE / LOAD
# ============================================================================

mkdir -p "$PRESETS_DIR"

PARAM_LIST=(BACKEND HOST PORT GPU_LAYERS CTX_SIZE PREDICT TEMP TOP_K TOP_P MIN_P
            PRESENCE_PENALTY REPEAT_PENALTY THREADS BATCH_SIZE UBATCH_SIZE
            FLASH_ATTN JINJA REASONING_FORMAT MMPROJ EXTRA_ARGS SYSTEM_PROMPT
            SPLIT_MODE TENSOR_SPLIT MAIN_GPU VISIBLE_DEVICES)

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
    FLASH_ATTN="on"; JINJA="yes"; MMPROJ=""; EXTRA_ARGS=""
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
            local prism_dir
            prism_dir=$(dirname "$SELECTED_MODEL")
            local prism_mmproj
            prism_mmproj=$(find "$prism_dir" -maxdepth 1 -name "*mmproj*" -type f 2>/dev/null | head -1) || true
            [[ -n "$prism_mmproj" ]] && MMPROJ="$prism_mmproj"
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
}

# ============================================================================
#  MODEL SELECTION
# ============================================================================

select_model() {
    header
    echo -e "  ${BOLD}Step 1: Select a model${NC}"
    divider
    echo

    MODELS=()
    local display=()

    shopt -s nullglob
    for f in "$MODEL_DIR"/*.gguf; do
        MODELS+=("$f")
        local sz
        sz=$(du -h "$f" 2>/dev/null | cut -f1) || sz="?"
        display+=("$(basename "$f") ${DIM}(${sz})${NC}")
    done
    for d in "$MODEL_DIR"/*/; do
        local clean_d="${d%/}"
        local first
        first=$(ls -1 "$clean_d"/*.gguf 2>/dev/null | sort | head -n 1) || true
        if [[ -n "$first" ]]; then
            MODELS+=("$first")
            local dsz
            dsz=$(du -sh "$clean_d" 2>/dev/null | cut -f1) || dsz="?"
            local cnt
            cnt=$(ls -1 "$clean_d"/*.gguf 2>/dev/null | wc -l) || cnt=0
            display+=("$(basename "$clean_d")/ ${DIM}(${dsz}, ${cnt} shards)${NC}")
        fi
    done
    shopt -u nullglob

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
        echo -e "\n  ${GREEN}OK${NC} $(basename "$SELECTED_MODEL")"
    else
        echo -e "  ${RED}Invalid selection${NC}"; exit 1
    fi

    local mdir
    mdir=$(dirname "$SELECTED_MODEL")
    local mp
    mp=$(find "$mdir" -maxdepth 1 -name "*mmproj*.gguf" -type f 2>/dev/null | head -1) || true
    if [[ -n "$mp" ]]; then
        echo -e "  ${CYAN}Found vision projector:${NC} $(basename "$mp")"
        read -rp "  Enable vision? [Y/n]: " vis
        [[ "${vis,,}" != "n" ]] && MMPROJ="$mp" && echo -e "  ${GREEN}OK${NC} Vision enabled"
    fi

    apply_model_preset

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
        echo -e "  ${BOLD}Network & Compute${NC}"
        echo -e "  ${GREEN} 1)${NC} Host/Port      ${BOLD}${HOST}:${PORT}${NC}"
        echo -e "  ${GREEN} 2)${NC} GPU layers     ${BOLD}${GPU_LAYERS}${NC} ${DIM}(-ngl)${NC}"
        echo -e "  ${GREEN} 3)${NC} Context size   ${BOLD}${CTX_SIZE}${NC} ${DIM}(-c)${NC}"
        echo -e "  ${GREEN} 4)${NC} Max predict    ${BOLD}${PREDICT}${NC} ${DIM}(-n, -1=unlimited)${NC}"
        echo -e "  ${GREEN} 5)${NC} CPU threads    ${BOLD}${THREADS}${NC} ${DIM}(-t, -1=auto)${NC}"
        echo -e "  ${GREEN} 6)${NC} Batch size     ${BOLD}${BATCH_SIZE}${NC} ${DIM}(-b)${NC}"
        echo -e "  ${GREEN} 7)${NC} Flash attn     ${BOLD}${FLASH_ATTN}${NC} ${DIM}(on/off; ROCm: try off if slow)${NC}"
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
        echo -e "  ${GREEN}16)${NC} mmproj         ${BOLD}${MMPROJ:-none}${NC}"
        echo -e "  ${GREEN}17)${NC} Extra args     ${BOLD}${EXTRA_ARGS:-none}${NC}"
        echo
        divider
        echo -e "  ${BOLD}Presets${NC}"
        echo -e "  ${YELLOW} p1)${NC} Chat       ${DIM}-- temp 0.7, no penalties, no_think${NC}"
        echo -e "  ${YELLOW} p2)${NC} Reasoning  ${DIM}-- temp 0.6, ctx 32K, thinking on${NC}"
        echo -e "  ${YELLOW} p3)${NC} Coding     ${DIM}-- temp 0.2, repeat 1.1, no_think${NC}"
        echo -e "  ${YELLOW} p4)${NC} Creative   ${DIM}-- temp 0.9, top-k 40, thinking on${NC}"
        echo
        echo -e "  ${BOLD}Config${NC}"
        echo -e "  ${YELLOW}  s)${NC} Save current config"
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
            6)  update_param BATCH_SIZE "Batch size (-b)" ;;
            7)  update_param FLASH_ATTN "Flash attention (on/off)" ;;
            8)  update_param TEMP "Temperature" ;;
            9)  update_param TOP_K "Top-K"; update_param TOP_P "Top-P" ;;
            10) update_param MIN_P "Min-P" ;;
            11) update_param PRESENCE_PENALTY "Presence penalty" ;;
            12) update_param REPEAT_PENALTY "Repeat penalty" ;;
            13) update_param JINJA "Jinja (yes/no)" ;;
            14) update_param REASONING_FORMAT "Reasoning format (auto/none/deepseek)" ;;
            15) read -rp "  System prompt: " SYSTEM_PROMPT ;;
            16) update_param MMPROJ "mmproj path" ;;
            17) update_param EXTRA_ARGS "Extra args" ;;

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
                : > "$mfile"
                for p in "${PARAM_LIST[@]}"; do echo "${p}=${!p}" >> "$mfile"; done
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

    CMD=("$bin")
    CMD+=(-m "$SELECTED_MODEL")
    CMD+=(--host "$HOST" --port "$PORT")
    CMD+=(-ngl "$GPU_LAYERS" -c "$CTX_SIZE" -n "$PREDICT")
    CMD+=(--temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P")
    CMD+=(--presence-penalty "$PRESENCE_PENALTY" --repeat-penalty "$REPEAT_PENALTY")
    CMD+=(-b "$BATCH_SIZE" -ub "$UBATCH_SIZE")
    CMD+=(-fa "$FLASH_ATTN")

    [[ -n "$SPLIT_MODE" ]]   && CMD+=(-sm "$SPLIT_MODE")
    [[ -n "$TENSOR_SPLIT" ]] && CMD+=(-ts "$TENSOR_SPLIT")
    [[ -n "$MAIN_GPU" ]]     && CMD+=(-mg "$MAIN_GPU")

    [[ "$THREADS" != "-1" ]] && CMD+=(-t "$THREADS")
    [[ "$JINJA" == "yes" ]]  && CMD+=(--jinja)

    [[ -n "$REASONING_FORMAT" && "$REASONING_FORMAT" != "auto" ]] && \
        CMD+=(--reasoning-format "$REASONING_FORMAT")
    [[ -n "$MMPROJ" && "$MMPROJ" != "none" ]] && \
        CMD+=(--mmproj "$MMPROJ")
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
    echo -e "  ${DIM}Web UI:  http://${HOST}:${PORT}${NC}"
    echo -e "  ${DIM}API:     http://${HOST}:${PORT}/v1${NC}"
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

main() {
    local bin
    bin="$(current_binary)"
    if [[ ! -x "$bin" ]]; then
        echo -e "  ${RED}Error: llama-server (${BACKEND}) not found at ${bin}${NC}"
        echo
        echo "  Build with: ./build_tools.sh llama ${BACKEND}"
        exit 1
    fi
    [[ ! -d "$MODEL_DIR" ]] && { echo -e "  ${RED}Error: ${MODEL_DIR} not found${NC}"; exit 1; }

    select_model
    configure_params
}

main "$@"
