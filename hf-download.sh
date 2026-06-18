#!/bin/bash
# ============================================================================
#  hf-download.sh — Interactive HuggingFace model downloader
# ============================================================================

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/srv/models}"
VENV="${VENV:-/srv/llama/llm-tools/bin/activate}"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

header() {
    clear
    echo -e "\n  ${CYAN}${BOLD}HuggingFace Model Downloader${NC}\n"
}

divider() {
    echo -e "${DIM}  ---------------------------------------------------------${NC}"
}

# ── Ensure venv is active and hf available ──
ensure_hf() {
    if ! command -v hf &>/dev/null; then
        if [[ -f "$VENV" ]]; then
            # shellcheck source=/dev/null
            source "$VENV"
        fi
    fi
    if ! command -v hf &>/dev/null; then
        echo -e "  ${RED}Error: 'hf' command not found. Activate your venv first:${NC}"
        echo "  source $VENV"
        exit 1
    fi
}

# ── List local models ──
list_local() {
    echo -e "  ${BOLD}Local models in ${MODEL_DIR}:${NC}"
    divider

    if [[ ! -d "$MODEL_DIR" ]]; then
        echo -e "  ${DIM}Directory not found${NC}"
        return
    fi

    local count=0
    shopt -s nullglob

    # 1. Standalone files
    for f in "$MODEL_DIR"/*.gguf; do
        local size
        size=$(du -h "$f" 2>/dev/null | cut -f1) || size="???"
        local name="${f#$MODEL_DIR/}"
        echo -e "  ${GREEN}•${NC} ${name} ${DIM}(${size})${NC}"
        count=$((count + 1))
    done

    # 2. Directories (split models)
    for d in "$MODEL_DIR"/*/; do
        local clean_d="${d%/}"
        if ls "$clean_d"/*.gguf &>/dev/null; then
            local dir_size
            dir_size=$(du -sh "$clean_d" 2>/dev/null | cut -f1) || dir_size="???"
            local dir_name="${clean_d#$MODEL_DIR/}"
            local gguf_count
            gguf_count=$(ls -1 "$clean_d"/*.gguf | wc -l)
            echo -e "  ${GREEN}•${NC} ${dir_name}/ ${DIM}(${dir_size}, ${gguf_count} shards)${NC}"
            count=$((count + 1))
        fi
    done

    shopt -u nullglob

    if [[ $count -eq 0 ]]; then
        echo -e "  ${DIM}No .gguf models found${NC}"
    fi
    echo
}

# ── List files in a repo ──
list_repo_files() {
    local repo="$1"
    echo -e "\n  ${BOLD}Files in ${repo}:${NC}"
    divider
    python3 -c "
from huggingface_hub import list_repo_files
try:
    files = list_repo_files('$repo')
    gguf_files = [f for f in files if f.endswith('.gguf')]
    other_files = [f for f in files if not f.endswith('.gguf')]

    if gguf_files:
        print('  GGUF files:')
        for f in sorted(gguf_files):
            print(f'    {f}')
    else:
        print('  No GGUF files found.')
        print('  Other files:')
        for f in sorted(other_files)[:20]:
            print(f'    {f}')
        if len(other_files) > 20:
            print(f'    ... and {len(other_files) - 20} more')
except Exception as e:
    print(f'  Error: {e}')
"
    echo
}

# ── Download a model ──
download_model() {
    local repo="$1"
    local file="$2"

    echo -e "\n  ${BOLD}Downloading:${NC}"
    echo -e "  Repo: ${CYAN}${repo}${NC}"

    if [[ -n "$file" ]]; then
        if [[ "$file" == */ ]]; then
            echo -e "  Pattern: ${CYAN}${file}*${NC}"
            echo -e "  Dest:    ${MODEL_DIR}/"
            divider
            hf download "$repo" --include "${file}*" --local-dir "$MODEL_DIR"
        else
            echo -e "  File:    ${CYAN}${file}${NC}"
            echo -e "  Dest:    ${MODEL_DIR}/"
            divider
            hf download "$repo" "$file" --local-dir "$MODEL_DIR"
        fi
    else
        echo -e "  ${YELLOW}Downloading entire repo${NC}"
        echo -e "  Dest:    ${MODEL_DIR}/${repo}"
        divider
        hf download "$repo" --local-dir "$MODEL_DIR/$repo"
    fi

    if [[ $? -eq 0 ]]; then
        echo -e "\n  ${GREEN}✓ Download complete!${NC}"
    else
        echo -e "\n  ${RED}✗ Download failed${NC}"
    fi
}

# ── Delete a local model ──
delete_model() {
    echo -e "  ${BOLD}Delete a local model${NC}"
    divider

    local TARGETS=()
    local display_names=()
    local display_sizes=()

    shopt -s nullglob

    for f in "$MODEL_DIR"/*.gguf; do
        TARGETS+=("$f")
        local size
        size=$(du -h "$f" 2>/dev/null | cut -f1) || size="???"
        display_sizes+=("$size")
        display_names+=("${f#$MODEL_DIR/}")
    done

    for d in "$MODEL_DIR"/*/; do
        local clean_d="${d%/}"
        if ls "$clean_d"/*.gguf &>/dev/null; then
            TARGETS+=("$clean_d")
            local dir_size
            dir_size=$(du -sh "$clean_d" 2>/dev/null | cut -f1) || dir_size="???"
            display_sizes+=("$dir_size")
            local dir_name="${clean_d#$MODEL_DIR/}"
            display_names+=("${dir_name}/ (split model)")
        fi
    done

    shopt -u nullglob

    if [[ ${#TARGETS[@]} -eq 0 ]]; then
        echo -e "  ${DIM}No models found${NC}\n"
        return
    fi

    for i in "${!TARGETS[@]}"; do
        echo -e "  ${GREEN}$((i+1)))${NC} ${display_names[$i]} ${DIM}(${display_sizes[$i]})${NC}"
    done

    echo
    read -rp "  Select model [1-${#TARGETS[@]}] (or 'c' to cancel): " choice

    if [[ "${choice,,}" == "c" || -z "$choice" ]]; then
        return
    fi

    if [[ "$choice" -ge 1 && "$choice" -le ${#TARGETS[@]} ]]; then
        local target="${TARGETS[$((choice-1))]}"
        read -rp "  Delete ${display_names[$((choice-1))]}? [y/N]: " confirm
        if [[ "${confirm,,}" == "y" ]]; then
            rm -rf "$target"
            echo -e "  ${GREEN}✓ Deleted${NC}\n"
        fi
    fi
}

# ── Manage HF cache ──
manage_cache() {
    echo -e "  ${BOLD}HuggingFace Cache${NC}"
    divider
    hf cache scan 2>/dev/null || echo -e "  ${DIM}Cache scan not available${NC}"
    echo
    read -rp "  Clean up cache? [y/N] (or 'c' to cancel): " confirm
    if [[ "${confirm,,}" == "y" ]]; then
        hf cache delete 2>/dev/null || echo -e "  ${DIM}Interactive cache cleanup not available${NC}"
    fi
}

# ── Quick download presets ──
quick_download() {
    header
    echo -e "  ${BOLD}Quick Download — Popular Models${NC}"
    divider
    echo -e "  ${BOLD}General Chat${NC}"
    echo -e "  ${GREEN} 1)${NC} Qwen3.5-27B Q4_K_M           ${DIM}~16GB — best dense model for 4090${NC}"
    echo -e "  ${GREEN} 2)${NC} Qwen3.5-35B-A3B Q4_K_M       ${DIM}~20GB — fast MoE, 3B active${NC}"
    echo -e "  ${GREEN} 3)${NC} Qwen3-8B Q8_0                ${DIM}~8.5GB — lightweight, near-lossless${NC}"
    echo
    echo -e "  ${BOLD}Coding${NC}"
    echo -e "  ${GREEN} 4)${NC} Qwen2.5-Coder-32B Q4_K_M     ${DIM}~18GB — FIM/autocomplete king${NC}"
    echo -e "  ${GREEN} 5)${NC} Qwen3-Coder-30B-A3B Q4_K_M   ${DIM}~17GB — agentic coding MoE${NC}"
    echo
    echo -e "  ${BOLD}Uncensored${NC}"
    echo -e "  ${GREEN} 6)${NC} Qwen3.5-27B Uncensored Q5_K_M ${DIM}~19GB — HauhauCS aggressive${NC}"
    echo
    divider
    read -rp "  Select model (or 'c' to cancel): " choice

    case "${choice,,}" in
        1) download_model "unsloth/Qwen3.5-27B-GGUF" "Qwen3.5-27B-Q4_K_M.gguf" ;;
        2) download_model "unsloth/Qwen3.5-35B-A3B-GGUF" "Qwen3.5-35B-A3B-Q4_K_M.gguf" ;;
        3) download_model "Qwen/Qwen3-8B-GGUF" "qwen3-8b-q8_0.gguf" ;;
        4) download_model "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF" "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf" ;;
        5) download_model "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF" "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" ;;
        6) download_model "HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive" "Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf" ;;
        c|"") return ;;
        *) echo -e "  ${RED}Invalid option${NC}"; sleep 0.5; return ;;
    esac

    read -rp "  Press Enter to continue..."
}

# ── Main menu ──
main() {
    ensure_hf
    mkdir -p "$MODEL_DIR"

    while true; do
        header
        echo -e "  ${BOLD}Main Menu${NC}"
        divider
        echo -e "  ${GREEN} 1)${NC} Quick download (popular models)"
        echo -e "  ${GREEN} 2)${NC} Download by repo + filename"
        echo -e "  ${GREEN} 3)${NC} Download split model (multi-shard)"
        echo -e "  ${GREEN} 4)${NC} List files in a HF repo"
        echo -e "  ${GREEN} 5)${NC} List local models"
        echo -e "  ${GREEN} 6)${NC} Delete a local model"
        echo -e "  ${GREEN} 7)${NC} Manage HF cache"
        echo -e "  ${GREEN} q)${NC} Quit"
        echo
        read -rp "  Select: " opt

        case "${opt,,}" in
            1) quick_download ;;
            2)
                header
                echo -e "  ${BOLD}Download Model${NC}"
                divider
                echo -e "  ${DIM}Example repo:  unsloth/Qwen3.5-27B-GGUF${NC}"
                echo -e "  ${DIM}Example file:  Qwen3.5-27B-Q4_K_M.gguf${NC}\n"

                read -rp "  HF repo (org/name) ['c' to cancel]: " repo
                if [[ "${repo,,}" == "c" || -z "$repo" ]]; then continue; fi

                echo -e "\n  ${DIM}Fetching file list...${NC}"
                list_repo_files "$repo"

                read -rp "  Filename (Enter for entire repo, 'c' to cancel): " file
                if [[ "${file,,}" == "c" ]]; then continue; fi

                download_model "$repo" "$file"
                read -rp "  Press Enter to continue..."
                ;;
            3)
                header
                echo -e "  ${BOLD}Download Split Model (Multi-Shard)${NC}"
                divider
                echo -e "  ${DIM}Example repo:    Qwen/Qwen3-Coder-Next-GGUF${NC}"
                echo -e "  ${DIM}Example prefix: Qwen3-Coder-Next-Q4_K_M/${NC}\n"

                read -rp "  HF repo (org/name) ['c' to cancel]: " repo
                if [[ "${repo,,}" == "c" || -z "$repo" ]]; then continue; fi

                echo -e "\n  ${DIM}Fetching file list...${NC}"
                list_repo_files "$repo"

                read -rp "  Directory prefix ['c' to cancel]: " prefix
                if [[ "${prefix,,}" == "c" || -z "$prefix" ]]; then continue; fi

                [[ "$prefix" != */ ]] && prefix="${prefix}/"
                echo -e "\n  Downloading all files matching: ${CYAN}${prefix}*${NC}"
                hf download "$repo" --include "${prefix}*" --local-dir "$MODEL_DIR"
                echo -e "\n  ${GREEN}✓ Download complete!${NC}"

                read -rp "  Press Enter to continue..."
                ;;
            4)
                header
                echo -e "  ${BOLD}List Files in HF Repo${NC}"
                divider
                read -rp "  HF repo (org/name) ['c' to cancel]: " repo
                if [[ "${repo,,}" != "c" && -n "$repo" ]]; then
                    list_repo_files "$repo"
                    read -rp "  Press Enter to continue..."
                fi
                ;;
            5)
                header
                list_local
                read -rp "  Press Enter to continue..."
                ;;
            6)
                header
                delete_model
                read -rp "  Press Enter to continue..."
                ;;
            7)
                header
                manage_cache
                read -rp "  Press Enter to continue..."
                ;;
            q)
                echo -e "  ${YELLOW}Bye!${NC}\n"
                exit 0
                ;;
            *)
                echo -e "  ${RED}Invalid option${NC}"
                sleep 0.5
                ;;
        esac
    done
}

main "$@"
