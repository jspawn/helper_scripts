#!/bin/bash
# ============================================================================
#  build_tools.sh -- Build llama.cpp, stable-diffusion.cpp, whisper.cpp,
#                     piper TTS, and the llm-tools venv
#  Hardware: 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) / 7950X (znver4)
#            CUDA (NVIDIA) and SYCL (Intel oneAPI) backends also supported
#
#  Usage:
#    ./build_tools.sh                    # show this help (builds nothing)
#    ./build_tools.sh all                # all projects, default backends (rocm+vulkan)
#    ./build_tools.sh llama rocm         # just llama.cpp ROCm
#    ./build_tools.sh sd vulkan          # just stable-diffusion.cpp Vulkan
#    ./build_tools.sh whisper cuda       # just whisper.cpp CUDA (NVIDIA)
#    ./build_tools.sh llama sycl         # just llama.cpp SYCL (Intel oneAPI)
#    ./build_tools.sh rocm               # all projects, ROCm only
#    ./build_tools.sh piper              # piper TTS (venv install, no GPU backend)
#    ./build_tools.sh tools              # llm-tools venv (hf download tooling)
#    ./build_tools.sh --clean llama cuda # nuke build dir first, then build
#
#  Projects:  llama, sd, whisper, piper, tools, all
#  Backends:  rocm, vulkan, cuda, sycl  (default if omitted: rocm vulkan)
#  Env vars:  JOBS=16        parallelism (default 32)
#             CUDA_ARCHS=89   CUDA arch list (default "native" = auto-detect)
# ============================================================================

set -euo pipefail

# -- Paths ------------------------------------------------------------------
ROCM_DIR="/srv/llama/llama.cpp-rocm"
VULKAN_DIR="/srv/llama/llama.cpp-vulkan"
CUDA_DIR="/srv/llama/llama.cpp-cuda"
SYCL_DIR="/srv/llama/llama.cpp-sycl"
SD_ROCM_DIR="/srv/llama/stable-diffusion.cpp-rocm"
SD_VULKAN_DIR="/srv/llama/stable-diffusion.cpp-vulkan"
SD_CUDA_DIR="/srv/llama/stable-diffusion.cpp-cuda"
SD_SYCL_DIR="/srv/llama/stable-diffusion.cpp-sycl"
WHISPER_ROCM_DIR="/srv/llama/whisper.cpp-rocm"
WHISPER_VULKAN_DIR="/srv/llama/whisper.cpp-vulkan"
WHISPER_CUDA_DIR="/srv/llama/whisper.cpp-cuda"
WHISPER_SYCL_DIR="/srv/llama/whisper.cpp-sycl"
# CUDA target architectures for llama/sd/whisper CUDA builds. "native" lets
# nvcc detect the installed GPU(s); set e.g. CUDA_ARCHS=89 for a fixed list.
CUDA_ARCHS="${CUDA_ARCHS:-native}"
# Piper TTS (piper1-gpl) is a Python package -- no GPU backend, no CMake.
# Installed into a dedicated venv; voices (onnx + onnx.json) live with the
# other models so the orchestrator's voice.tts.command can reference them.
PIPER_VENV="/srv/llama/piper/.venv"
PIPER_VOICE_DIR="/srv/models/piper"
PIPER_VOICE="en_US-lessac-high"
PIPER_VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high"
# llm-tools venv: Python tooling for the scripts here (hf-download.sh needs
# the `hf` CLI). Kept minimal on purpose -- the finetuning stack lives in
# /srv/finetuning (own requirements.txt, own venv).
TOOLS_VENV="/srv/llama/llm-tools"
TOOLS_REQUIREMENTS="/srv/llama/requirements.txt"
# Shared install location for the SD web UI frontend. The Vite project lives
# inside each SD source tree at examples/server/frontend, but the built
# artifacts are copied here so both backends use the same canonical path
# via --serve-html-path.
SD_FRONTEND_INSTALL_DIR="/srv/llama/stable-diffusion/web/server/frontend"
SD_FRONTEND_HTML="${SD_FRONTEND_INSTALL_DIR}/dist/index.html"
JOBS="${JOBS:-32}"

# -- Args -------------------------------------------------------------------
# Parse projects (llama, sd, whisper, piper, tools, all) and backends
# (rocm, vulkan, cuda, sycl) independently. If only backends specified, all
# projects are built. If only projects specified, the default backends
# (rocm+vulkan) are built. No args at all shows the usage header and exits.
# piper and tools are backend-less: each produces a single job.
usage() { sed -n '/^#  Usage:/,/====/p' "$0"; }

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

PROJECTS=()
BACKENDS=()
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        llama|sd|whisper|piper|tools) PROJECTS+=("$arg") ;;
        all) PROJECTS+=(llama sd whisper piper tools) ;;
        rocm|vulkan|cuda|sycl) BACKENDS+=("$arg") ;;
        --clean|-c)  CLEAN=1 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) echo "Unknown arg: $arg (try --help)"; exit 1 ;;
    esac
done
[[ ${#PROJECTS[@]} -eq 0 ]] && PROJECTS=(llama sd whisper piper tools)
[[ ${#BACKENDS[@]} -eq 0 ]] && BACKENDS=(rocm vulkan)

# Build a flat list of (project, backend) jobs to run
JOBS_LIST=()
for p in "${PROJECTS[@]}"; do
    if [[ "$p" == "piper" || "$p" == "tools" ]]; then
        JOBS_LIST+=("$p")
        continue
    fi
    for b in "${BACKENDS[@]}"; do
        JOBS_LIST+=("${p}-${b}")
    done
done

# -- Colors -----------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

step() { echo -e "\n${CYAN}${BOLD}==>${NC} ${BOLD}$*${NC}"; }
ok()   { echo -e "${GREEN}OK${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC}  $*"; }
fail() { echo -e "${RED}FAIL${NC} $*"; exit 1; }

# -- Pre-flight checks ------------------------------------------------------
check_clone() {
    local dir="$1"
    local url="$2"
    local recursive="${3:-}"
    if [[ ! -d "$dir/.git" ]]; then
        warn "$dir not found, cloning..."
        mkdir -p "$(dirname "$dir")"
        if [[ "$recursive" == "recursive" ]]; then
            git clone --recursive "$url" "$dir"
        else
            git clone "$url" "$dir"
        fi
    fi
}

sync_repo() {
    local dir="$1"
    local recursive="${2:-}"
    step "Sync ${dir}"
    cd "$dir"
    git fetch --all --tags --quiet
    git pull --ff-only
    if [[ "$recursive" == "recursive" ]]; then
        git submodule update --init --recursive --quiet
    fi
    ok "at $(git describe --tags --always)"
}

# -- SD web UI frontend build + install -------------------------------------
# sd-server only ships a placeholder at "/". The actual web UI lives in
# examples/server/frontend as a Vite project and must be built separately.
# We build it from the freshly-synced SD source tree, then copy the dist/
# output to a shared install location so both backends use the same path
# via --serve-html-path ${SD_FRONTEND_INSTALL_DIR}/dist/index.html
build_sd_frontend() {
    local src_dir="$1/examples/server/frontend"
    local src_label="$(basename "$1")"

    if [[ ! -d "$src_dir" ]]; then
        warn "Frontend source not found at $src_dir, skipping web UI"
        return 0
    fi

    step "Build SD web UI frontend (from $src_label)"

    # Prefer pnpm (lockfile is pnpm-lock.yaml), fall back to npm
    local pm=""
    if command -v pnpm >/dev/null 2>&1; then
        pm="pnpm"
    elif command -v npm >/dev/null 2>&1; then
        warn "pnpm not found, falling back to npm (slower, dep tree may differ)"
        pm="npm"
    else
        warn "Neither pnpm nor npm found -- skipping web UI build"
        warn "  Install with: sudo pacman -S pnpm  (or  nodejs npm)"
        return 0
    fi

    # Skip rebuild if dist/index.html in the source tree is newer than
    # package.json, vite.config.js, and everything under src/
    local need_build=1
    if [[ -f "$src_dir/dist/index.html" ]] \
       && [[ "$src_dir/dist/index.html" -nt "$src_dir/package.json" ]] \
       && [[ "$src_dir/dist/index.html" -nt "$src_dir/vite.config.js" ]]; then
        local newest_src
        newest_src=$(find "$src_dir/src" -type f -newer "$src_dir/dist/index.html" 2>/dev/null | head -1)
        if [[ -z "$newest_src" ]]; then
            ok "Web UI dist/ in source tree is up to date, skipping rebuild"
            need_build=0
        fi
    fi

    if [[ $need_build -eq 1 ]]; then
        pushd "$src_dir" >/dev/null
        if [[ "$pm" == "pnpm" ]]; then
            pnpm install --frozen-lockfile
            pnpm run build
        else
            npm install
            npm run build
        fi
        popd >/dev/null

        if [[ ! -f "$src_dir/dist/index.html" ]]; then
            warn "Web UI build finished but dist/index.html not found"
            return 0
        fi
    fi

    # Install (copy) dist/ to the shared canonical location. We copy rather
    # than symlink so an `--clean` on the source tree won't break the install.
    step "Install web UI -> ${SD_FRONTEND_INSTALL_DIR}"
    mkdir -p "$SD_FRONTEND_INSTALL_DIR"
    rm -rf "${SD_FRONTEND_INSTALL_DIR}/dist"
    cp -r "$src_dir/dist" "${SD_FRONTEND_INSTALL_DIR}/dist"
    ok "Web UI installed: ${SD_FRONTEND_HTML}"
}

# -- llama.cpp ROCm build ---------------------------------------------------
build_llama_rocm() {
    check_clone "$ROCM_DIR" "https://github.com/ggml-org/llama.cpp"
    sync_repo "$ROCM_DIR"
    cd "$ROCM_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean ROCm build dir"
        rm -rf build
    fi

    step "Configure ROCm (gfx1201 = R9700)"
    # Note: hipcc lives in /opt/rocm on Arch via the `rocm-hip-sdk` package

    # rocWMMA enables HW-accelerated flash attention on RDNA, but the headers
    # are in a separate Arch package (`rocwmma`). Auto-detect.
    local rocwmma_flag="OFF"
    if [[ -f "/opt/rocm/include/rocwmma/rocwmma-version.hpp" ]] \
       || [[ -f "/usr/include/rocwmma/rocwmma-version.hpp" ]]; then
        rocwmma_flag="ON"
        ok "rocWMMA headers found -- enabling HW flash-attn"
    else
        warn "rocWMMA headers NOT found -- building without HW flash-attn"
        warn "  Install with: sudo pacman -S rocwmma"
    fi

    # Web UI is intentionally disabled: we run llama-server headless behind
    # LiteLLM. Both flags are currently required -- LLAMA_BUILD_UI=OFF alone
    # still triggers asset provisioning (npm build / HF bucket download / stale
    # dist embed), which is exactly what was breaking the build after the
    # Svelte+PWA web UI rework landed. Drop both flags + `pacman -S nodejs npm`
    # if you ever want the embedded browser UI back.
    cmake -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1201 \
        -DGGML_HIP_ROCWMMA_FATTN="$rocwmma_flag" \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DCMAKE_C_FLAGS="-march=znver4 -O3" \
        -DCMAKE_CXX_FLAGS="-march=znver4 -O3"

    step "Build ROCm (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

    if [[ -x "$ROCM_DIR/build/bin/llama-server" ]]; then
        ok "ROCm build at $ROCM_DIR/build/bin/llama-server"
    else
        fail "ROCm build produced no llama-server binary"
    fi
}

# -- llama.cpp Vulkan build -------------------------------------------------
build_llama_vulkan() {
    check_clone "$VULKAN_DIR" "https://github.com/ggml-org/llama.cpp"
    sync_repo "$VULKAN_DIR"
    cd "$VULKAN_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean Vulkan build dir"
        rm -rf build
    fi

    step "Configure Vulkan"
    # Web UI disabled here too -- see note in build_llama_rocm().
    cmake -B build \
        -DGGML_VULKAN=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=znver4 -O3" \
        -DCMAKE_CXX_FLAGS="-march=znver4 -O3"

    step "Build Vulkan (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

    if [[ -x "$VULKAN_DIR/build/bin/llama-server" ]]; then
        ok "Vulkan build at $VULKAN_DIR/build/bin/llama-server"
    else
        fail "Vulkan build produced no llama-server binary"
    fi
}

# -- Intel oneAPI environment (for SYCL builds) -------------------------------
# SYCL builds need icx/icpx plus the oneAPI libs. Source the standard
# setvars.sh if the compiler isn't already on PATH.
ensure_oneapi() {
    if ! command -v icpx >/dev/null 2>&1; then
        if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
            step "Source Intel oneAPI environment"
            # setvars.sh trips on unbound vars under `set -u`
            set +u
            source /opt/intel/oneapi/setvars.sh >/dev/null
            set -u
        else
            fail "icpx not found -- install Intel oneAPI (AUR: intel-oneapi-compiler-dpcpp-cpp)"
        fi
    fi
}

# -- llama.cpp CUDA build (NVIDIA) --------------------------------------------
build_llama_cuda() {
    check_clone "$CUDA_DIR" "https://github.com/ggml-org/llama.cpp"
    sync_repo "$CUDA_DIR"
    cd "$CUDA_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean CUDA build dir"
        rm -rf build
    fi

    step "Configure CUDA (archs: ${CUDA_ARCHS})"
    # Web UI disabled here too -- see note in build_llama_rocm().
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=znver4 -O3" \
        -DCMAKE_CXX_FLAGS="-march=znver4 -O3"

    step "Build CUDA (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

    if [[ -x "$CUDA_DIR/build/bin/llama-server" ]]; then
        ok "CUDA build at $CUDA_DIR/build/bin/llama-server"
    else
        fail "CUDA build produced no llama-server binary"
    fi
}

# -- llama.cpp SYCL build (Intel oneAPI) ----------------------------------------
build_llama_sycl() {
    check_clone "$SYCL_DIR" "https://github.com/ggml-org/llama.cpp"
    sync_repo "$SYCL_DIR"
    ensure_oneapi
    cd "$SYCL_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean SYCL build dir"
        rm -rf build
    fi

    step "Configure SYCL (Intel oneAPI)"
    # Web UI disabled here too -- see note in build_llama_rocm().
    # No -march flags here: keep the oneAPI defaults; add -fsycl-targets via
    # CMAKE_CXX_FLAGS if you need to target something other than the default.
    cmake -B build \
        -DGGML_SYCL=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx

    step "Build SYCL (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

    if [[ -x "$SYCL_DIR/build/bin/llama-server" ]]; then
        ok "SYCL build at $SYCL_DIR/build/bin/llama-server"
    else
        fail "SYCL build produced no llama-server binary"
    fi
}

# -- whisper.cpp ROCm build ---------------------------------------------------
build_whisper_rocm() {
    check_clone "$WHISPER_ROCM_DIR" "https://github.com/ggml-org/whisper.cpp"
    sync_repo "$WHISPER_ROCM_DIR"
    cd "$WHISPER_ROCM_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean whisper ROCm build dir"
        rm -rf build
    fi

    step "Configure whisper.cpp ROCm (gfx1201 = R9700)"
    # No ffmpeg: the orchestrator web UI sends 16 kHz mono WAV, which
    # whisper-server decodes natively. Add -DWHISPER_FFMPEG=ON if you ever
    # need to feed it arbitrary audio formats.
    cmake -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1201 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DCMAKE_C_FLAGS="-march=znver4 -O3" \
        -DCMAKE_CXX_FLAGS="-march=znver4 -O3"

    step "Build whisper ROCm (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target whisper-server whisper-cli

    if [[ -x "$WHISPER_ROCM_DIR/build/bin/whisper-server" ]]; then
        ok "whisper ROCm build at $WHISPER_ROCM_DIR/build/bin/whisper-server"
    else
        fail "whisper ROCm build produced no whisper-server binary"
    fi
}

# -- whisper.cpp Vulkan build -------------------------------------------------
build_whisper_vulkan() {
    check_clone "$WHISPER_VULKAN_DIR" "https://github.com/ggml-org/whisper.cpp"
    sync_repo "$WHISPER_VULKAN_DIR"
    cd "$WHISPER_VULKAN_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean whisper Vulkan build dir"
        rm -rf build
    fi

    step "Configure whisper.cpp Vulkan"
    cmake -B build \
        -DGGML_VULKAN=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=znver4 -O3" \
        -DCMAKE_CXX_FLAGS="-march=znver4 -O3"

    step "Build whisper Vulkan (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target whisper-server whisper-cli

    if [[ -x "$WHISPER_VULKAN_DIR/build/bin/whisper-server" ]]; then
        ok "whisper Vulkan build at $WHISPER_VULKAN_DIR/build/bin/whisper-server"
    else
        fail "whisper Vulkan build produced no whisper-server binary"
    fi
}

# -- whisper.cpp CUDA build (NVIDIA) --------------------------------------------
build_whisper_cuda() {
    check_clone "$WHISPER_CUDA_DIR" "https://github.com/ggml-org/whisper.cpp"
    sync_repo "$WHISPER_CUDA_DIR"
    cd "$WHISPER_CUDA_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean whisper CUDA build dir"
        rm -rf build
    fi

    step "Configure whisper.cpp CUDA (archs: ${CUDA_ARCHS})"
    # No ffmpeg -- see note in build_whisper_rocm().
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=znver4 -O3" \
        -DCMAKE_CXX_FLAGS="-march=znver4 -O3"

    step "Build whisper CUDA (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target whisper-server whisper-cli

    if [[ -x "$WHISPER_CUDA_DIR/build/bin/whisper-server" ]]; then
        ok "whisper CUDA build at $WHISPER_CUDA_DIR/build/bin/whisper-server"
    else
        fail "whisper CUDA build produced no whisper-server binary"
    fi
}

# -- whisper.cpp SYCL build (Intel oneAPI) ---------------------------------------
build_whisper_sycl() {
    check_clone "$WHISPER_SYCL_DIR" "https://github.com/ggml-org/whisper.cpp"
    sync_repo "$WHISPER_SYCL_DIR"
    ensure_oneapi
    cd "$WHISPER_SYCL_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean whisper SYCL build dir"
        rm -rf build
    fi

    step "Configure whisper.cpp SYCL (Intel oneAPI)"
    # No ffmpeg -- see note in build_whisper_rocm().
    cmake -B build \
        -DGGML_SYCL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx

    step "Build whisper SYCL (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target whisper-server whisper-cli

    if [[ -x "$WHISPER_SYCL_DIR/build/bin/whisper-server" ]]; then
        ok "whisper SYCL build at $WHISPER_SYCL_DIR/build/bin/whisper-server"
    else
        fail "whisper SYCL build produced no whisper-server binary"
    fi
}

# -- piper TTS install (venv, CPU only) ---------------------------------------
build_piper() {
    step "Install piper TTS -> ${PIPER_VENV}"

    command -v python3 >/dev/null 2>&1 || fail "python3 not found"
    command -v curl    >/dev/null 2>&1 || fail "curl not found (needed for voice download)"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean piper venv"
        rm -rf "$PIPER_VENV"
    fi

    if [[ ! -x "$PIPER_VENV/bin/pip" ]]; then
        python3 -m venv "$PIPER_VENV"
    fi
    "$PIPER_VENV/bin/pip" install --quiet --upgrade pip
    "$PIPER_VENV/bin/pip" install --quiet --upgrade piper-tts

    # Default voice model (onnx + config); skip if already present
    mkdir -p "$PIPER_VOICE_DIR"
    local onnx="${PIPER_VOICE_DIR}/${PIPER_VOICE}.onnx"
    local json="${PIPER_VOICE_DIR}/${PIPER_VOICE}.onnx.json"
    if [[ ! -f "$onnx" ]]; then
        step "Download voice ${PIPER_VOICE}"
        curl -fL --progress-bar -o "$onnx" "${PIPER_VOICE_BASE}/${PIPER_VOICE}.onnx"
        curl -fL --progress-bar -o "$json" "${PIPER_VOICE_BASE}/${PIPER_VOICE}.onnx.json"
    else
        ok "Voice ${PIPER_VOICE} already present"
    fi

    if [[ -x "$PIPER_VENV/bin/piper" ]]; then
        ok "piper at $PIPER_VENV/bin/piper (voice: $onnx)"
    else
        fail "piper install produced no piper binary in $PIPER_VENV/bin"
    fi
}

# -- llm-tools venv (hf download tooling) -------------------------------------
build_llmtools() {
    step "Install llm-tools venv -> ${TOOLS_VENV}"

    command -v python3 >/dev/null 2>&1 || fail "python3 not found"
    [[ -f "$TOOLS_REQUIREMENTS" ]] || fail "requirements not found: $TOOLS_REQUIREMENTS"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean llm-tools venv"
        rm -rf "$TOOLS_VENV"
    fi

    if [[ ! -x "$TOOLS_VENV/bin/pip" ]]; then
        python3 -m venv "$TOOLS_VENV"
    fi
    "$TOOLS_VENV/bin/pip" install --quiet --upgrade pip
    "$TOOLS_VENV/bin/pip" install --quiet --upgrade -r "$TOOLS_REQUIREMENTS"

    if [[ -x "$TOOLS_VENV/bin/hf" ]]; then
        ok "hf CLI at $TOOLS_VENV/bin/hf"
    else
        fail "llm-tools install produced no hf binary in $TOOLS_VENV/bin"
    fi
}

# -- stable-diffusion.cpp ROCm build ----------------------------------------
build_sd_rocm() {
    check_clone "$SD_ROCM_DIR" "https://github.com/leejet/stable-diffusion.cpp" recursive
    sync_repo "$SD_ROCM_DIR" recursive
    cd "$SD_ROCM_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean SD ROCm build dir"
        rm -rf build
    fi

    step "Configure stable-diffusion.cpp ROCm (gfx1201)"
    # Note: SD uses -DSD_HIPBLAS=ON (not GGML_HIP) and wants both GPU_TARGETS
    # and AMDGPU_TARGETS depending on ROCm version. Ninja is recommended.
    # rocWMMA is kept OFF for SD -- per AUR notes it regressed on ROCm 7+.
    cmake -B build \
        -G Ninja \
        -DSD_HIPBLAS=ON \
        -DGPU_TARGETS=gfx1201 \
        -DAMDGPU_TARGETS=gfx1201 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON

    step "Build SD ROCm (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS"

    if [[ -x "$SD_ROCM_DIR/build/bin/sd" ]] \
       || [[ -x "$SD_ROCM_DIR/build/bin/sd-cli" ]]; then
        ok "SD ROCm build at $SD_ROCM_DIR/build/bin/"
    else
        fail "SD ROCm build produced no sd/sd-cli binary"
    fi

    # Build + install web UI from this freshly-synced source tree.
    # Both backends do this; whichever runs second just refreshes the install.
    build_sd_frontend "$SD_ROCM_DIR"
}

# -- stable-diffusion.cpp Vulkan build --------------------------------------
build_sd_vulkan() {
    check_clone "$SD_VULKAN_DIR" "https://github.com/leejet/stable-diffusion.cpp" recursive
    sync_repo "$SD_VULKAN_DIR" recursive
    cd "$SD_VULKAN_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean SD Vulkan build dir"
        rm -rf build
    fi

    step "Configure stable-diffusion.cpp Vulkan"
    cmake -B build \
        -DSD_VULKAN=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=znver4 -O3" \
        -DCMAKE_CXX_FLAGS="-march=znver4 -O3"

    step "Build SD Vulkan (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS"

    if [[ -x "$SD_VULKAN_DIR/build/bin/sd" ]] \
       || [[ -x "$SD_VULKAN_DIR/build/bin/sd-cli" ]]; then
        ok "SD Vulkan build at $SD_VULKAN_DIR/build/bin/"
    else
        fail "SD Vulkan build produced no sd/sd-cli binary"
    fi

    # Build + install web UI from this freshly-synced source tree.
    # Both backends do this; whichever runs second just refreshes the install.
    build_sd_frontend "$SD_VULKAN_DIR"
}

# -- stable-diffusion.cpp CUDA build (NVIDIA) -----------------------------------
build_sd_cuda() {
    check_clone "$SD_CUDA_DIR" "https://github.com/leejet/stable-diffusion.cpp" recursive
    sync_repo "$SD_CUDA_DIR" recursive
    cd "$SD_CUDA_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean SD CUDA build dir"
        rm -rf build
    fi

    step "Configure stable-diffusion.cpp CUDA (archs: ${CUDA_ARCHS})"
    cmake -B build \
        -DSD_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=znver4 -O3" \
        -DCMAKE_CXX_FLAGS="-march=znver4 -O3"

    step "Build SD CUDA (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS"

    if [[ -x "$SD_CUDA_DIR/build/bin/sd" ]] \
       || [[ -x "$SD_CUDA_DIR/build/bin/sd-cli" ]]; then
        ok "SD CUDA build at $SD_CUDA_DIR/build/bin/"
    else
        fail "SD CUDA build produced no sd/sd-cli binary"
    fi

    build_sd_frontend "$SD_CUDA_DIR"
}

# -- stable-diffusion.cpp SYCL build (Intel oneAPI) ------------------------------
build_sd_sycl() {
    check_clone "$SD_SYCL_DIR" "https://github.com/leejet/stable-diffusion.cpp" recursive
    sync_repo "$SD_SYCL_DIR" recursive
    ensure_oneapi
    cd "$SD_SYCL_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean SD SYCL build dir"
        rm -rf build
    fi

    step "Configure stable-diffusion.cpp SYCL (Intel oneAPI)"
    cmake -B build \
        -DSD_SYCL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx

    step "Build SD SYCL (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS"

    if [[ -x "$SD_SYCL_DIR/build/bin/sd" ]] \
       || [[ -x "$SD_SYCL_DIR/build/bin/sd-cli" ]]; then
        ok "SD SYCL build at $SD_SYCL_DIR/build/bin/"
    else
        fail "SD SYCL build produced no sd/sd-cli binary"
    fi

    build_sd_frontend "$SD_SYCL_DIR"
}

# -- Sanity: toolchain ------------------------------------------------------
preflight() {
    step "Pre-flight checks"

    command -v cmake >/dev/null 2>&1 || fail "cmake not found"
    command -v git   >/dev/null 2>&1 || fail "git not found"

    # Ninja required for SD ROCm build
    if [[ " ${JOBS_LIST[*]} " == *" sd-rocm "* ]]; then
        command -v ninja >/dev/null 2>&1 \
            || fail "ninja not found (required for SD ROCm build) -- pacman -S ninja"
    fi

    # piper + tools need python3 (venvs); piper also needs curl (voice download)
    if [[ " ${JOBS_LIST[*]} " == *" piper "* || " ${JOBS_LIST[*]} " == *" tools "* ]]; then
        command -v python3 >/dev/null 2>&1 || fail "python3 not found (required for piper/tools)"
        python3 -m venv --help >/dev/null 2>&1 \
            || fail "python3 venv module missing -- pacman -S python"
    fi
    if [[ " ${JOBS_LIST[*]} " == *" piper "* ]]; then
        command -v curl >/dev/null 2>&1 \
            || fail "curl not found (required for piper voice download)"
    fi

    # pnpm/npm required for SD web UI -- warn once if any SD backend is built
    if [[ " ${JOBS_LIST[*]} " == *" sd-"* ]]; then
        if command -v pnpm >/dev/null 2>&1; then
            ok "pnpm present for web UI build"
        elif command -v npm >/dev/null 2>&1; then
            warn "pnpm not found -- web UI will build with npm (slower)"
            warn "  Recommended: sudo pacman -S pnpm"
        else
            warn "Neither pnpm nor npm found -- web UI will be skipped"
            warn "  Install with: sudo pacman -S pnpm  (or  nodejs npm)"
        fi
    fi

    # Dedupe backends across projects so we don't print the same warning twice
    local checked_rocm=0 checked_vulkan=0 checked_cuda=0 checked_sycl=0
    for job in "${JOBS_LIST[@]}"; do
        local backend="${job#*-}"

        if [[ "$backend" == "rocm" && $checked_rocm -eq 0 ]]; then
            checked_rocm=1
            command -v hipcc >/dev/null 2>&1 || fail "hipcc not found -- pacman -S rocm-hip-sdk"
            if ! command -v rocminfo >/dev/null 2>&1; then
                warn "rocminfo not found -- pacman -S rocminfo (skipping device check)"
            elif rocminfo 2>/dev/null | grep -q "gfx1201"; then
                local n
                n=$(rocminfo 2>/dev/null | grep -c "Name:.*gfx1201" || true)
                ok "rocminfo sees ${n} gfx1201 device(s)"
            else
                warn "rocminfo doesn't list gfx1201. Check:"
                warn "  1. ROCm >= 6.4.1 required for RDNA4 (pacman -Qi rocm-hip-runtime)"
                warn "  2. User in render+video groups: groups | grep -E 'render|video'"
                warn "  3. amdgpu kernel module loaded: lsmod | grep amdgpu"
                warn "  4. AUR fallback if needed: paru -S rocm-gfx120x-bin"
            fi
            # rocWMMA only matters for llama.cpp (SD has it off intentionally)
            if [[ " ${JOBS_LIST[*]} " == *" llama-rocm "* ]]; then
                if [[ ! -f /opt/rocm/include/rocwmma/rocwmma-version.hpp \
                   && ! -f /usr/include/rocwmma/rocwmma-version.hpp ]]; then
                    warn "rocwmma headers missing -- pacman -S rocwmma (optional but recommended for llama.cpp)"
                fi
            fi
        fi

        if [[ "$backend" == "vulkan" && $checked_vulkan -eq 0 ]]; then
            checked_vulkan=1
            if [[ ! -f /usr/include/vulkan/vulkan.h ]]; then
                warn "vulkan headers missing -- pacman -S vulkan-headers"
            else
                ok "vulkan headers present"
            fi
            if ! ls /usr/{lib,share}/cmake/SPIRV-Headers/*.cmake >/dev/null 2>&1 \
               && ! ls /usr/{lib,share}/cmake/spirv-headers/*.cmake >/dev/null 2>&1; then
                warn "SPIRV-Headers CMake config missing -- pacman -S spirv-headers spirv-tools"
            fi
            if ! ldconfig -p 2>/dev/null | grep -q "libvulkan.so"; then
                warn "libvulkan.so not found -- pacman -S vulkan-icd-loader"
            fi
            command -v glslc >/dev/null 2>&1 \
                || warn "glslc not found -- pacman -S shaderc"
            if ! ls /usr/share/vulkan/icd.d/radeon_icd*.json >/dev/null 2>&1; then
                warn "RADV driver not installed -- pacman -S vulkan-radeon"
            fi
            if command -v vulkaninfo >/dev/null 2>&1; then
                local n
                n=$(vulkaninfo --summary 2>/dev/null | grep -c "deviceName.*Radeon" || true)
                ok "vulkaninfo sees ${n} Radeon device(s)"
            else
                warn "vulkaninfo not found -- pacman -S vulkan-tools (optional)"
            fi
        fi

        if [[ "$backend" == "cuda" && $checked_cuda -eq 0 ]]; then
            checked_cuda=1
            command -v nvcc >/dev/null 2>&1 \
                || fail "nvcc not found -- install the CUDA toolkit (pacman -S cuda)"
            if command -v nvidia-smi >/dev/null 2>&1; then
                local n
                n=$(nvidia-smi --list-gpus 2>/dev/null | grep -c "^GPU" || true)
                ok "nvidia-smi sees ${n} NVIDIA device(s)"
            else
                warn "nvidia-smi not found -- skipping device check"
            fi
        fi

        if [[ "$backend" == "sycl" && $checked_sycl -eq 0 ]]; then
            checked_sycl=1
            if command -v icpx >/dev/null 2>&1; then
                ok "icpx present: $(icpx --version 2>/dev/null | head -1)"
            elif [[ -f /opt/intel/oneapi/setvars.sh ]]; then
                ok "oneAPI setvars.sh found (will be sourced at build time)"
            else
                fail "icpx not found and no /opt/intel/oneapi/setvars.sh -- install Intel oneAPI (AUR: intel-oneapi-compiler-dpcpp-cpp)"
            fi
        fi
    done
}

# -- Main -------------------------------------------------------------------
main() {
    echo -e "${BOLD}llama.cpp + stable-diffusion.cpp + whisper.cpp build${NC}"
    echo -e "${DIM}Projects: ${PROJECTS[*]} | Backends: ${BACKENDS[*]} | Clean: ${CLEAN} | Jobs: ${JOBS}${NC}"
    echo -e "${DIM}Plan:     ${JOBS_LIST[*]}${NC}"

    preflight

    local t0
    t0=$(date +%s)

    for job in "${JOBS_LIST[@]}"; do
        case "$job" in
            llama-rocm)     build_llama_rocm ;;
            llama-vulkan)   build_llama_vulkan ;;
            llama-cuda)     build_llama_cuda ;;
            llama-sycl)     build_llama_sycl ;;
            sd-rocm)        build_sd_rocm ;;
            sd-vulkan)      build_sd_vulkan ;;
            sd-cuda)        build_sd_cuda ;;
            sd-sycl)        build_sd_sycl ;;
            whisper-rocm)   build_whisper_rocm ;;
            whisper-vulkan) build_whisper_vulkan ;;
            whisper-cuda)   build_whisper_cuda ;;
            whisper-sycl)   build_whisper_sycl ;;
            piper)          build_piper ;;
            tools)          build_llmtools ;;
        esac
    done

    local elapsed=$(( $(date +%s) - t0 ))
    echo
    echo -e "${GREEN}${BOLD}Done${NC} in ${elapsed}s"
    echo
    for job in "${JOBS_LIST[@]}"; do
        case "$job" in
            llama-rocm)     echo -e "  llama   ROCm:   ${ROCM_DIR}/build/bin/llama-server" ;;
            llama-vulkan)   echo -e "  llama   Vulkan: ${VULKAN_DIR}/build/bin/llama-server" ;;
            llama-cuda)     echo -e "  llama   CUDA:   ${CUDA_DIR}/build/bin/llama-server" ;;
            llama-sycl)     echo -e "  llama   SYCL:   ${SYCL_DIR}/build/bin/llama-server" ;;
            sd-rocm)        echo -e "  sd      ROCm:   ${SD_ROCM_DIR}/build/bin/{sd-cli,sd-server}" ;;
            sd-vulkan)      echo -e "  sd      Vulkan: ${SD_VULKAN_DIR}/build/bin/{sd-cli,sd-server}" ;;
            sd-cuda)        echo -e "  sd      CUDA:   ${SD_CUDA_DIR}/build/bin/{sd-cli,sd-server}" ;;
            sd-sycl)        echo -e "  sd      SYCL:   ${SD_SYCL_DIR}/build/bin/{sd-cli,sd-server}" ;;
            whisper-rocm)   echo -e "  whisper ROCm:   ${WHISPER_ROCM_DIR}/build/bin/whisper-server" ;;
            whisper-vulkan) echo -e "  whisper Vulkan: ${WHISPER_VULKAN_DIR}/build/bin/whisper-server" ;;
            whisper-cuda)   echo -e "  whisper CUDA:   ${WHISPER_CUDA_DIR}/build/bin/whisper-server" ;;
            whisper-sycl)   echo -e "  whisper SYCL:   ${WHISPER_SYCL_DIR}/build/bin/whisper-server" ;;
            piper)          echo -e "  piper   TTS:    ${PIPER_VENV}/bin/piper (voice: ${PIPER_VOICE_DIR}/${PIPER_VOICE}.onnx)" ;;
            tools)          echo -e "  tools   venv:   ${TOOLS_VENV}/bin/hf (hf download CLI)" ;;
        esac
    done
    if [[ -f "$SD_FRONTEND_HTML" ]]; then
        echo -e "  sd     WebUI:  ${SD_FRONTEND_HTML}"
        echo
        echo -e "${DIM}Launch web UI with:${NC}"
        echo -e "  sd-server --serve-html-path ${SD_FRONTEND_HTML} \\"
        echo -e "    --diffusion-model ... --vae ... --llm ... \\"
        echo -e "    -l 0.0.0.0 --listen-port 1234 --diffusion-fa -v"
    fi
    echo
    echo -e "${DIM}Quick tests:${NC}"
    [[ " ${JOBS_LIST[*]} " == *" llama-rocm "* ]] && \
        echo -e "  ${ROCM_DIR}/build/bin/llama-server --list-devices"
    [[ " ${JOBS_LIST[*]} " == *" sd-rocm "* ]] && \
        echo -e "  ${SD_ROCM_DIR}/build/bin/sd-cli --help | head -30"
    [[ " ${JOBS_LIST[*]} " == *" whisper-rocm "* ]] && \
        echo -e "  ${WHISPER_ROCM_DIR}/build/bin/whisper-server -m /srv/models/whisper/ggml-small.bin --port 8097"
    [[ " ${JOBS_LIST[*]} " == *" piper "* ]] && \
        echo -e "  echo 'hello world' | ${PIPER_VENV}/bin/piper --model ${PIPER_VOICE_DIR}/${PIPER_VOICE}.onnx --output_file /tmp/piper-test.wav"
    [[ " ${JOBS_LIST[*]} " == *" tools "* ]] && \
        echo -e "  ${TOOLS_VENV}/bin/hf --version"
}

main
