#!/bin/bash
# ============================================================================
#  serve -- Dispatcher for the AI serve scripts
#  Usage:
#    ./serve llm     # text generation (llama-server)
#    ./serve image   # image generation (sd-server)
#    ./serve         # interactive picker
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    llm|llama|chat|text)
        exec "$SCRIPT_DIR/llama-serve.sh" "${@:2}"
        ;;
    image|img|sd|diffusion)
        exec "$SCRIPT_DIR/sd-serve.sh" "${@:2}"
        ;;
    "")
        echo "  Choose a server:"
        echo "    1) LLM     (llama-server: chat, completion, embeddings)"
        echo "    2) Image   (sd-server: text-to-image, image-to-image)"
        echo
        read -rp "  > " choice
        case "$choice" in
            1|llm|llama)        exec "$SCRIPT_DIR/llama-serve.sh" ;;
            2|image|img|sd)     exec "$SCRIPT_DIR/sd-serve.sh" ;;
            *) echo "  Cancelled"; exit 1 ;;
        esac
        ;;
    -h|--help)
        sed -n '3,8p' "$0"
        ;;
    *)
        echo "Unknown target: $1" >&2
        echo "Run with: llm | image" >&2
        exit 1
        ;;
esac
