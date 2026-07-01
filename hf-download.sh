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

# ── Collect local models into parallel arrays (shared by list + delete) ──
#   MODEL_PATHS[i]  filesystem path (file or directory)
#   MODEL_LABELS[i] display name
#   MODEL_MSIZES[i] human-readable size
# Handles loose *.gguf files, legacy one-level dirs with direct shards, and
# org/name repo folders (two levels) holding .gguf anywhere beneath them.
collect_models() {
    MODEL_PATHS=(); MODEL_LABELS=(); MODEL_MSIZES=()
    [[ -d "$MODEL_DIR" ]] || return 0
    shopt -s nullglob

    # 1. Standalone files in the root
    for f in "$MODEL_DIR"/*.gguf; do
        MODEL_PATHS+=("$f")
        MODEL_LABELS+=("${f#"$MODEL_DIR"/}")
        MODEL_MSIZES+=("$(du -h "$f" 2>/dev/null | cut -f1 || echo '???')")
    done

    # 2. Legacy one-level dirs that directly contain shards
    for d in "$MODEL_DIR"/*/; do
        local cd="${d%/}"
        local shards=("$cd"/*.gguf)          # nullglob -> empty array if none
        if [[ ${#shards[@]} -gt 0 ]]; then
            MODEL_PATHS+=("$cd")
            MODEL_LABELS+=("${cd#"$MODEL_DIR"/}/ (${#shards[@]} shards)")
            MODEL_MSIZES+=("$(du -sh "$cd" 2>/dev/null | cut -f1 || echo '???')")
        fi
    done

    # 3. org/name repo folders (two levels) holding .gguf anywhere beneath
    for d in "$MODEL_DIR"/*/*/; do
        local cd="${d%/}"
        if find -L "$cd" -maxdepth 3 -name '*.gguf' -print -quit 2>/dev/null | grep -q .; then
            local n; n=$(find -L "$cd" -maxdepth 3 -name '*.gguf' 2>/dev/null | wc -l | tr -d ' ')
            local sfx=""; if [[ "$n" -ne 1 ]]; then sfx="s"; fi
            MODEL_PATHS+=("$cd")
            MODEL_LABELS+=("${cd#"$MODEL_DIR"/}/ (${n} file${sfx})")
            MODEL_MSIZES+=("$(du -sh "$cd" 2>/dev/null | cut -f1 || echo '???')")
        fi
    done

    shopt -u nullglob
}

# ── List local models ──
list_local() {
    echo -e "  ${BOLD}Local models in ${MODEL_DIR}:${NC}"
    divider

    if [[ ! -d "$MODEL_DIR" ]]; then
        echo -e "  ${DIM}Directory not found${NC}"
        return
    fi

    collect_models
    if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
        echo -e "  ${DIM}No .gguf models found${NC}"
    else
        local i
        for i in "${!MODEL_PATHS[@]}"; do
            echo -e "  ${GREEN}•${NC} ${MODEL_LABELS[$i]} ${DIM}(${MODEL_MSIZES[$i]})${NC}"
        done
    fi
    echo
}

# ── List files in a repo ──
# ── Fetch a repo's file list (raw: gguf first, then everything else) ──
# Prints one filename per line on stdout; errors go to stderr (exit 1).
fetch_repo_files() {
    local repo="$1"
    python3 -c "
from huggingface_hub import list_repo_files
import sys
try:
    files = list_repo_files('$repo')
except Exception as e:
    sys.stderr.write(f'  Error: {e}\n'); sys.exit(1)
gguf  = sorted(f for f in files if f.endswith('.gguf'))
other = sorted(f for f in files if not f.endswith('.gguf'))
for f in gguf + other:
    print(f)
"
}

# ── List files in a repo (viewer; shows ALL files, gguf highlighted) ──
list_repo_files() {
    local repo="$1"
    echo -e "\n  ${BOLD}Files in ${repo}:${NC}"
    divider
    local files=()
    mapfile -t files < <(fetch_repo_files "$repo")
    if [[ ${#files[@]} -eq 0 ]]; then
        echo -e "  ${DIM}No files found (or repo error)${NC}"
        echo; return
    fi
    local f
    for f in "${files[@]}"; do
        if [[ "$f" == *.gguf ]]; then
            echo -e "    ${GREEN}${f}${NC}"
        else
            echo -e "    ${DIM}${f}${NC}"
        fi
    done
    echo
}

# ── Pick one/several files from a repo (multi-select) and download them ──
# Selection syntax: "1 5 6" or "1,5,6" or ranges "1-3", 'a'=all, Enter=whole
# repo, 'c'=cancel. Everything lands in $MODEL_DIR/<repo>/.
download_selected() {
    local repo="$1"
    echo -e "\n  ${DIM}Fetching file list...${NC}"

    local files=()
    mapfile -t files < <(fetch_repo_files "$repo")
    if [[ ${#files[@]} -eq 0 ]]; then
        echo -e "  ${RED}No files found (or repo error).${NC}"
        return
    fi

    echo -e "\n  ${BOLD}Files in ${repo}:${NC}"
    divider
    local i
    for i in "${!files[@]}"; do
        local f="${files[$i]}" tag=""
        [[ "$f" == *.gguf ]] && tag=" ${DIM}(gguf)${NC}"
        printf "  ${GREEN}%2d)${NC} %s%b\n" "$((i+1))" "$f" "$tag"
    done
    echo
    echo -e "  ${DIM}Pick: e.g. '1 5 6' or '1-3', 'a'=all, Enter=whole repo, 'c'=cancel${NC}"
    read -rp "  > " sel

    [[ "${sel,,}" == "c" ]] && return
    if [[ -z "$sel" ]]; then
        download_model "$repo" ""          # whole repo
        return
    fi

    local chosen=()
    if [[ "${sel,,}" == "a" ]]; then
        chosen=("${files[@]}")
    else
        local tok
        for tok in ${sel//,/ }; do          # commas -> spaces, then split
            if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                local lo="${BASH_REMATCH[1]}" hi="${BASH_REMATCH[2]}" n
                for ((n = lo; n <= hi; n++)); do
                    [[ "$n" -ge 1 && "$n" -le ${#files[@]} ]] && chosen+=("${files[$((n-1))]}")
                done
            elif [[ "$tok" =~ ^[0-9]+$ ]]; then
                if [[ "$tok" -ge 1 && "$tok" -le ${#files[@]} ]]; then
                    chosen+=("${files[$((tok-1))]}")
                else
                    echo -e "  ${YELLOW}Out of range: $tok${NC}"
                fi
            else
                echo -e "  ${YELLOW}Ignoring: $tok${NC}"
            fi
        done
    fi

    # de-duplicate while preserving order
    local uniq=() c x dup
    for c in "${chosen[@]}"; do
        dup=0; for x in "${uniq[@]}"; do [[ "$x" == "$c" ]] && { dup=1; break; }; done
        [[ $dup -eq 0 ]] && uniq+=("$c")
    done
    chosen=("${uniq[@]}")

    if [[ ${#chosen[@]} -eq 0 ]]; then
        echo -e "  ${RED}Nothing selected.${NC}"
        return
    fi

    local dest="$MODEL_DIR/$repo"
    echo -e "\n  ${BOLD}Downloading ${#chosen[@]} file(s)${NC} -> ${CYAN}${dest}/${NC}"
    divider
    local file ok=0 fail=0
    for file in "${chosen[@]}"; do
        echo -e "  ${CYAN}|${NC} $file"
        if hf download "$repo" "$file" --local-dir "$dest"; then
            ok=$((ok + 1))
        else
            echo -e "  ${RED}x failed: $file${NC}"; fail=$((fail + 1))
        fi
    done
    echo -e "\n  ${GREEN}OK ${ok} downloaded${NC}$([[ $fail -gt 0 ]] && echo -e " / ${RED}${fail} failed${NC}")"
}

# ── Download a model ──
download_model() {
    local repo="$1"
    local file="$2"

    echo -e "\n  ${BOLD}Downloading:${NC}"
    echo -e "  Repo: ${CYAN}${repo}${NC}"

    # Always download into an org/name repo folder so files stay organised,
    # whether it's the whole repo, a single file, or a shard pattern.
    local dest="$MODEL_DIR/$repo"

    if [[ -n "$file" ]]; then
        if [[ "$file" == */ ]]; then
            echo -e "  Pattern: ${CYAN}${file}*${NC}"
            echo -e "  Dest:    ${dest}/"
            divider
            hf download "$repo" --include "${file}*" --local-dir "$dest"
        else
            echo -e "  File:    ${CYAN}${file}${NC}"
            echo -e "  Dest:    ${dest}/"
            divider
            hf download "$repo" "$file" --local-dir "$dest"
        fi
    else
        echo -e "  ${YELLOW}Downloading entire repo${NC}"
        echo -e "  Dest:    ${dest}/"
        divider
        hf download "$repo" --local-dir "$dest"
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

    collect_models

    if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
        echo -e "  ${DIM}No models found${NC}\n"
        return
    fi

    local i
    for i in "${!MODEL_PATHS[@]}"; do
        echo -e "  ${GREEN}$((i+1)))${NC} ${MODEL_LABELS[$i]} ${DIM}(${MODEL_MSIZES[$i]})${NC}"
    done

    echo
    read -rp "  Select model [1-${#MODEL_PATHS[@]}] (or 'c' to cancel): " choice

    if [[ "${choice,,}" == "c" || -z "$choice" ]]; then
        return
    fi

    if [[ "$choice" =~ ^[0-9]+$ ]] && [[ "$choice" -ge 1 && "$choice" -le ${#MODEL_PATHS[@]} ]]; then
        local target="${MODEL_PATHS[$((choice-1))]}"
        read -rp "  Delete ${MODEL_LABELS[$((choice-1))]}? [y/N]: " confirm
        if [[ "${confirm,,}" == "y" ]]; then
            rm -rf "$target"
            echo -e "  ${GREEN}✓ Deleted${NC}"
            # If that emptied the org/ parent (org/name layout), prune it too.
            local parent; parent="$(dirname "$target")"
            if [[ "$parent" != "$MODEL_DIR" && -d "$parent" && -z "$(ls -A "$parent" 2>/dev/null)" ]]; then
                rmdir "$parent" 2>/dev/null && \
                    echo -e "  ${DIM}removed empty ${parent#"$MODEL_DIR"/}/${NC}"
            fi
            echo
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
        echo -e "  ${GREEN} 2)${NC} Download from a repo (pick files)"
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
                echo -e "  ${BOLD}Download Model${NC} ${DIM}(pick one, several, or all files)${NC}"
                divider
                echo -e "  ${DIM}Example repo:  unsloth/Qwen3.5-27B-GGUF${NC}\n"

                read -rp "  HF repo (org/name) ['c' to cancel]: " repo
                if [[ "${repo,,}" == "c" || -z "$repo" ]]; then continue; fi

                download_selected "$repo"
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
                echo -e "  Dest: ${CYAN}${MODEL_DIR}/${repo}/${NC}"
                hf download "$repo" --include "${prefix}*" --local-dir "$MODEL_DIR/$repo"
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
