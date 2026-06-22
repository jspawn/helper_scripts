#!/bin/bash
# ============================================================================
#  runner.sh -- fzf-driven launcher for the AI stack
#  Hardware: 2x AMD Radeon AI PRO R9700 (64GB total VRAM, gfx1201, RDNA4)
#            AMD 7950X / 64GB DDR5
#  Backends: ROCm/HIP (default) or Vulkan (BACKEND=vulkan ./runner.sh)
# ============================================================================

# --- PATHS & BINARIES ---
MODEL_DIR="/srv/models"
AI_TOOLS_DIR="/srv/llama/ai-tools"
MCP_CONFIG="$AI_TOOLS_DIR/mcp-config.json"
WEBUI_CONFIG="$AI_TOOLS_DIR/webui-settings.json"
SEARXNG_DIR="/srv/podman/searxng"

BACKEND="${BACKEND:-rocm}"
LLAMA_ROCM="${LLAMA_ROCM:-/srv/llama/llama.cpp-rocm/build/bin/llama-server}"
LLAMA_VULKAN="${LLAMA_VULKAN:-/srv/llama/llama.cpp-vulkan/build/bin/llama-server}"

case "$BACKEND" in
    rocm)   LLAMA_BIN="$LLAMA_ROCM" ;;
    vulkan) LLAMA_BIN="$LLAMA_VULKAN" ;;
    *)      echo "Error: BACKEND must be 'rocm' or 'vulkan' (got: $BACKEND)"; exit 1 ;;
esac

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "Error: llama-server ($BACKEND) not found at $LLAMA_BIN"
    echo "Build hint:"
    if [[ "$BACKEND" == "rocm" ]]; then
        echo "  cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release"
    else
        echo "  cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release"
    fi
    echo "  cmake --build build -j32 --target llama-server"
    exit 1
fi

# --- RDNA4 ENVIRONMENT WORKAROUNDS ---
# gfx1201 stays pegged at 100% util after idle under HIP (ROCm/ROCm#5706).
# These mitigate it. Override by exporting before invocation.
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-1}"
export RADV_DEBUG="${RADV_DEBUG:-nocompute}"

# --- OPTIONAL: list devices and exit ---
if [[ "${LIST_DEVICES:-0}" == "1" ]]; then
    "$LLAMA_BIN" --list-devices
    exit 0
fi

# --- INTERACTIVE SELECTOR ---
if ! command -v fzf &> /dev/null; then
    echo "Error: fzf is not installed. Run: sudo pacman -S fzf"
    exit 1
fi

echo "[*] Scanning $MODEL_DIR..."
SELECTED_MODEL=$(find "$MODEL_DIR" -maxdepth 2 \( -name "*.gguf" -o -type d -name "*GGUF" \) \
    | fzf --prompt="Select Model > " --height=15% --layout=reverse --border --info=inline)
[[ -z "$SELECTED_MODEL" ]] && exit 1

# If a directory was picked (split GGUF), resolve to its first shard
if [[ -d "$SELECTED_MODEL" ]]; then
    FIRST_SHARD=$(ls -1 "$SELECTED_MODEL"/*.gguf 2>/dev/null | sort | head -n 1)
    [[ -n "$FIRST_SHARD" ]] && SELECTED_MODEL="$FIRST_SHARD"
fi

# --- HARDWARE TUNING (2x 32GB VRAM = 64GB total / 64GB RAM) ---
# Defaults: full GPU offload, 32K ctx, KV cache q8_0 to stretch context further.
# Models >30GB get spread across both GPUs via -sm layer + -ts 1,1.
GPU_LAYERS="999"
CTX=32768
KV_CACHE="--cache-type-k q8_0 --cache-type-v q8_0"
SPLIT_MODE="layer"
TENSOR_SPLIT=""
VISIBLE_DEVICES="0"    # default: pin to single GPU; big models override

case "$SELECTED_MODEL" in
    *Bonsai-8B* | *bonsai*8B*)
        echo "[!] Bonsai 8B: tiny model, GPU0 only, big ctx, no KV quant..."
        CTX=32768
        KV_CACHE=""    # ~5GB model, KV quant overhead not worth it
        VISIBLE_DEVICES="0"
        ;;
    *phi-4* | *Phi-4*)
        echo "[!] Phi-4: GPU0 only, 32K ctx..."
        CTX=32768
        VISIBLE_DEVICES="0"
        ;;
    *gemma-4-31B* | *PRISM*)
        echo "[!] Gemma 4 31B: full offload now possible (~18GB), GPU0 only..."
        CTX=32768
        VISIBLE_DEVICES="0"
        ;;
    *Qwen3.5-35B-A3B* | *qwen3-35B-A3B*)
        echo "[!] Qwen 35B-A3B MoE: GPU0 only, 64K ctx..."
        CTX=65536
        VISIBLE_DEVICES="0"
        ;;
    *Qwen3-Coder-30B* | *qwen3-coder-30B*)
        echo "[!] Qwen3-Coder-30B: GPU0 only, 64K ctx for big repos..."
        CTX=65536
        VISIBLE_DEVICES="0"
        ;;
    *Qwen3.5-27B* | *Qwopus*27B*)
        echo "[!] Qwen 27B: GPU0 only, 32K ctx..."
        CTX=32768
        VISIBLE_DEVICES="0"
        ;;
    *Llama*70B* | *llama*70B* | *L3*70B*)
        echo "[!] Llama 70B: spreading across both GPUs (layer split, 1:1)..."
        CTX=16384
        VISIBLE_DEVICES=""    # both GPUs visible
        TENSOR_SPLIT="1,1"
        ;;
    *Qwen*72B* | *qwen*72B*)
        echo "[!] Qwen 72B: spreading across both GPUs..."
        CTX=32768
        VISIBLE_DEVICES=""
        TENSOR_SPLIT="1,1"
        ;;
    *Mistral*Large* | *123B*)
        echo "[!] Mistral Large 123B: both GPUs, tight 8K ctx..."
        CTX=8192
        VISIBLE_DEVICES=""
        TENSOR_SPLIT="1,1"
        ;;
    *DeepSeek* | *deepseek* | *100B*)
        echo "[!] Large MoE: both GPUs, row split for parallelism..."
        CTX=16384
        VISIBLE_DEVICES=""
        SPLIT_MODE="row"
        TENSOR_SPLIT="1,1"
        ;;
    *)
        echo "[!] No specific tuning, using 32K ctx defaults on GPU0..."
        ;;
esac

# Apply GPU visibility for ROCm (ignored on Vulkan, where it's harmless)
if [[ -n "$VISIBLE_DEVICES" && "$BACKEND" == "rocm" ]]; then
    export HIP_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    export ROCR_VISIBLE_DEVICES="$VISIBLE_DEVICES"
fi
if [[ -n "$VISIBLE_DEVICES" && "$BACKEND" == "vulkan" ]]; then
    export GGML_VK_VISIBLE_DEVICES="$VISIBLE_DEVICES"
fi

EXTRA_FLAGS="--flash-attn on --mlock --no-mmap $KV_CACHE"

# Multi-GPU flags (only meaningful when both GPUs are visible)
MULTI_GPU_FLAGS=""
if [[ -z "$VISIBLE_DEVICES" ]]; then
    MULTI_GPU_FLAGS="-sm $SPLIT_MODE"
    [[ -n "$TENSOR_SPLIT" ]] && MULTI_GPU_FLAGS="$MULTI_GPU_FLAGS -ts $TENSOR_SPLIT"
fi

# --- INFRASTRUCTURE AUTOMATION ---
cleanup() {
    echo -e "\n\n[!] Tearing down AI stack..."
    kill "$PROXY_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start SearXNG via Podman
if ! podman ps --format "{{.Names}}" | grep -q "searxng"; then
    echo "[>] Podman: Starting SearXNG..."
    (cd "$SEARXNG_DIR" && podman-compose up -d)
fi

# Start the MCP Proxy with CORS bypass for PuTTY
echo "[>] MCP: Launching bridge with CORS bypass..."
uvx mcp-proxy --named-server-config "$MCP_CONFIG" --allow-origin "*" > /tmp/mcp-proxy.log 2>&1 &
PROXY_PID=$!
sleep 2

# --- EXECUTION ---
echo -e "\n-------------------------------------------------------"
echo "  MODEL:   $(basename "$SELECTED_MODEL")"
echo "  BACKEND: $BACKEND  ($LLAMA_BIN)"
echo "  CONTEXT: $CTX tokens"
echo "  GPU:     --n-gpu-layers $GPU_LAYERS"
if [[ -z "$VISIBLE_DEVICES" ]]; then
    echo "  DEVICES: both GPUs ($SPLIT_MODE split${TENSOR_SPLIT:+, ts=$TENSOR_SPLIT})"
else
    echo "  DEVICES: GPU $VISIBLE_DEVICES only"
fi
echo "  CONFIG:  Server-side WebUI tools enabled"
echo "-------------------------------------------------------"
echo

"$LLAMA_BIN" \
    --model "$SELECTED_MODEL" \
    --ctx-size "$CTX" \
    --n-gpu-layers "$GPU_LAYERS" \
    --jinja \
    --webui-config-file "$WEBUI_CONFIG" \
    --webui-mcp-proxy \
    $MULTI_GPU_FLAGS \
    $EXTRA_FLAGS \
    --host 0.0.0.0 \
    --port 8080
