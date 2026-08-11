#!/bin/bash
# ============================================================================
#  build_tools.sh -- Build llama.cpp, stable-diffusion.cpp, whisper.cpp,
#                     piper TTS, and the llm-tools venv
#  Backends:  vulkan (default; AMD/NVIDIA/Intel/mixed, no vendor SDK),
#             rocm (AMD), cuda (NVIDIA), sycl (Intel oneAPI),
#             cpu (no GPU; AVX2/AVX512/AMX via -march=native)
#
#  Usage:
#    ./build_tools.sh                    # show this help (builds nothing)
#    ./build_tools.sh all                # all projects, asks which backends
#    ./build_tools.sh llama rocm         # just llama.cpp ROCm (skips the menu)
#    ./build_tools.sh sd vulkan          # just stable-diffusion.cpp Vulkan
#    ./build_tools.sh whisper cuda       # just whisper.cpp CUDA (NVIDIA)
#    ./build_tools.sh llama sycl         # just llama.cpp SYCL (Intel oneAPI)
#    ./build_tools.sh rocm               # all projects, ROCm only
#    ./build_tools.sh piper              # piper TTS (venv install, no GPU backend)
#    ./build_tools.sh tools              # llm-tools venv (hf download tooling)
#    ./build_tools.sh --clean llama cuda # nuke build dir first, then build
#
#  Projects:  llama, sd, whisper, piper, tools, all
#  Backends:  rocm, vulkan, cuda, sycl, cpu
#             (omitted: menu; non-interactive: vulkan)
#  Env vars:  JOBS=16                 parallelism (default 32)
#             BIN_DIR=/opt/x         install dir for built binaries (skips prompt)
#             AMDGPU_TARGETS=gfx1100 AMD gfx target for ROCm builds (skips prompt)
#             CUDA_ARCHS=89          CUDA arch list (default "native" = auto)
#             MARCH=znver4           CPU -march for C/C++ (default "native")
#             PIPER_VOICE_DIR=...    piper voices dir (default ~/jaynet-models/piper)
#
#  At startup the script asks where to install the built binaries
#  (default: ~/jaynet-bin), then which backends + GPU targets to build for.
#  Binaries are copied to BIN_DIR after each build.
# ============================================================================

set -euo pipefail

# -- Paths ------------------------------------------------------------------
# Base dir = where this script (the repo) is checked out. All source trees,
# venvs, and install dirs live underneath it.
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_DIR="${BASE_DIR}/llama.cpp-rocm"
VULKAN_DIR="${BASE_DIR}/llama.cpp-vulkan"
CUDA_DIR="${BASE_DIR}/llama.cpp-cuda"
SYCL_DIR="${BASE_DIR}/llama.cpp-sycl"
SD_ROCM_DIR="${BASE_DIR}/stable-diffusion.cpp-rocm"
SD_VULKAN_DIR="${BASE_DIR}/stable-diffusion.cpp-vulkan"
SD_CUDA_DIR="${BASE_DIR}/stable-diffusion.cpp-cuda"
SD_SYCL_DIR="${BASE_DIR}/stable-diffusion.cpp-sycl"
WHISPER_ROCM_DIR="${BASE_DIR}/whisper.cpp-rocm"
WHISPER_VULKAN_DIR="${BASE_DIR}/whisper.cpp-vulkan"
WHISPER_CUDA_DIR="${BASE_DIR}/whisper.cpp-cuda"
WHISPER_SYCL_DIR="${BASE_DIR}/whisper.cpp-sycl"
# CPU-only trees: no GPU backend flags at all; llama.cpp's GGML_NATIVE plus
# the -march flag below give the host's best SIMD paths (AVX2/AVX512/AMX).
CPU_DIR="${BASE_DIR}/llama.cpp-cpu"
SD_CPU_DIR="${BASE_DIR}/stable-diffusion.cpp-cpu"
WHISPER_CPU_DIR="${BASE_DIR}/whisper.cpp-cpu"
# GPU/CPU tuning knobs. Empty here = asked later, but only when the matching
# backend is actually in the build plan; env presets skip the prompts.
#   AMDGPU_TARGETS: gfx target for ROCm builds (e.g. gfx1100, gfx1201)
#   CUDA_ARCHS:     CUDA arch list; "native" lets nvcc auto-detect
#   MARCH:          -march for C/C++ files ("native" = this host's CPU)
AMDGPU_TARGETS="${AMDGPU_TARGETS:-}"
CUDA_ARCHS="${CUDA_ARCHS:-}"
MARCH="${MARCH:-native}"
# Piper TTS (piper1-gpl) is a Python package -- no GPU backend, no CMake.
# Installed into a dedicated venv; voices (onnx + onnx.json) live with the
# other models so the orchestrator's voice.tts.command can reference them.
PIPER_VENV="${BASE_DIR}/piper/.venv"
PIPER_VOICE_DIR="${PIPER_VOICE_DIR:-${HOME}/jaynet-models/piper}"
PIPER_VOICE="en_US-lessac-high"
PIPER_VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high"
# llm-tools venv: Python tooling for the scripts here (hf-download.sh needs
# the `hf` CLI). Kept minimal on purpose -- the finetuning stack lives in
# /srv/finetuning (own requirements.txt, own venv).
TOOLS_VENV="${BASE_DIR}/llm-tools"
TOOLS_REQUIREMENTS="${BASE_DIR}/requirements.txt"
# Shared install location for the SD web UI frontend. The Vite project lives
# inside each SD source tree at examples/server/frontend, but the built
# artifacts are copied here so both backends use the same canonical path
# via --serve-html-path.
SD_FRONTEND_INSTALL_DIR="${BASE_DIR}/stable-diffusion/web/server/frontend"
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
        rocm|vulkan|cuda|sycl|cpu) BACKENDS+=("$arg") ;;
        --clean|-c)  CLEAN=1 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) echo "Unknown arg: $arg (try --help)"; exit 1 ;;
    esac
done
[[ ${#PROJECTS[@]} -eq 0 ]] && PROJECTS=(llama sd whisper piper tools)
# BACKENDS stays empty here on purpose: the menu (after the install-dir
# prompt) or the non-interactive default fills it in before JOBS_LIST is built.

# -- Colors -----------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

step() { echo -e "\n${CYAN}${BOLD}==>${NC} ${BOLD}$*${NC}"; }
ok()   { echo -e "${GREEN}OK${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC}  $*"; }
fail() { echo -e "${RED}FAIL${NC} $*"; exit 1; }

# -- Install dir --------------------------------------------------------------
# Ask where to install the built binaries. Default ~/jaynet-bin; set BIN_DIR
# to skip the prompt (also skipped automatically when stdin is not a tty).
BIN_DIR="${BIN_DIR:-}"
if [[ -z "$BIN_DIR" ]]; then
    if [[ -t 0 ]]; then
        read -r -p "Install directory for built binaries [${HOME}/jaynet-bin]: " BIN_DIR
        BIN_DIR="${BIN_DIR:-${HOME}/jaynet-bin}"
    else
        BIN_DIR="${HOME}/jaynet-bin"
    fi
fi
# read/env values keep a typed ~ literal; expand it so ~/bin doesn't
# become a literal ./~/bin directory
BIN_DIR="${BIN_DIR/#\~/${HOME}}"
BIN_DIR="${BIN_DIR%/}"
mkdir -p "$BIN_DIR"

# Copy built binaries into $BIN_DIR. Args: <build/bin dir> <binary>...
install_bins() {
    local src="$1"; shift
    local f
    for f in "$@"; do
        if [[ -x "$src/$f" ]]; then
            cp "$src/$f" "$BIN_DIR/$f"
            ok "Installed $f -> $BIN_DIR/$f"
        fi
    done
}

# Symlink a venv CLI into $BIN_DIR (copying would break the venv shebang).
install_link() {
    local target="$1"
    local name
    name="$(basename "$target")"
    if [[ -x "$target" ]]; then
        ln -sf "$target" "$BIN_DIR/$name"
        ok "Linked $name -> $BIN_DIR/$name"
    fi
}

# -- Backend selection ----------------------------------------------------------
# GPU-less projects (piper, tools) skip this entirely. Menu default = vulkan:
# one build runs on AMD/NVIDIA/Intel (incl. mixed-vendor splits) with no vendor
# SDK. Backends given on the command line skip the menu; non-tty -> vulkan.
GPU_PROJECTS=0
for p in "${PROJECTS[@]}"; do
    [[ "$p" == "piper" || "$p" == "tools" ]] || GPU_PROJECTS=1
done

if [[ ${#BACKENDS[@]} -eq 0 ]]; then
    if [[ $GPU_PROJECTS -eq 1 && -t 0 ]]; then
        echo "Backends to build (space-separated numbers) [1]:"
        echo "  1) vulkan   AMD / NVIDIA / Intel / mixed -- no vendor SDK (default)"
        echo "  2) rocm     AMD native (needs ROCm, e.g. pacman -S rocm-hip-sdk)"
        echo "  3) cuda     NVIDIA native (needs the CUDA toolkit)"
        echo "  4) sycl     Intel oneAPI"
        echo "  5) cpu      no GPU -- AVX2/AVX512/AMX via -march=native"
        read -r -p "> " backend_pick
        backend_pick="${backend_pick:-1}"
        for n in $backend_pick; do
            case "$n" in
                1) BACKENDS+=(vulkan) ;;
                2) BACKENDS+=(rocm) ;;
                3) BACKENDS+=(cuda) ;;
                4) BACKENDS+=(sycl) ;;
                5) BACKENDS+=(cpu) ;;
                *) echo "Unknown backend choice: $n" >&2; exit 1 ;;
            esac
        done
    else
        BACKENDS=(vulkan)
    fi
fi

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

# -- GPU targets for the native backends -----------------------------------------
# Asked only when the backend is actually in the job list; env presets skip.
if [[ -z "$AMDGPU_TARGETS" && " ${JOBS_LIST[*]} " == *"-rocm "* ]]; then
    if [[ -t 0 ]]; then
        echo "AMD GPU target for ROCm builds [gfx1201]:"
        echo "  gfx1030 RX 6800-6950 XT   gfx1031 RX 6700/6750 XT   gfx1032 RX 6600 (XT)"
        echo "  gfx1100 RX 7900 XT/XTX    gfx1101 RX 7700/7800 XT   gfx1102 RX 7600"
        echo "  gfx1103 Radeon 780M APU   gfx1151 Strix Halo"
        echo "  gfx1200 RX 9060 XT        gfx1201 RX 9070 (XT) / AI PRO R9700"
        read -r -p "> " AMDGPU_TARGETS
    fi
    AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1201}"
fi
if [[ -z "$CUDA_ARCHS" && " ${JOBS_LIST[*]} " == *"-cuda "* ]]; then
    if [[ -t 0 ]]; then
        echo "CUDA architectures [native = auto-detect]:"
        echo "  61 GTX 10xx/P40   75 RTX 20xx/T4   86 RTX 30xx"
        echo "  89 RTX 40xx/L4/L40   120 RTX 50xx"
        read -r -p "> " CUDA_ARCHS
    fi
    CUDA_ARCHS="${CUDA_ARCHS:-native}"
fi

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
        # -print -quit: stop at the first newer file; a `| head -1` pipe can
        # SIGPIPE find mid-stream and kill the script under pipefail
        newest_src=$(find "$src_dir/src" -type f -newer "$src_dir/dist/index.html" -print -quit 2>/dev/null)
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

    step "Configure ROCm (${AMDGPU_TARGETS})"
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
        -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
        -DGGML_HIP_ROCWMMA_FATTN="$rocwmma_flag" \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build ROCm (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

    if [[ -x "$ROCM_DIR/build/bin/llama-server" ]]; then
        ok "ROCm build at $ROCM_DIR/build/bin/llama-server"
    else
        fail "ROCm build produced no llama-server binary"
    fi

    install_bins "$ROCM_DIR/build/bin" llama-server llama-cli llama-bench
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
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build Vulkan (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

    if [[ -x "$VULKAN_DIR/build/bin/llama-server" ]]; then
        ok "Vulkan build at $VULKAN_DIR/build/bin/llama-server"
    else
        fail "Vulkan build produced no llama-server binary"
    fi

    install_bins "$VULKAN_DIR/build/bin" llama-server llama-cli llama-bench
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
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build CUDA (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

    if [[ -x "$CUDA_DIR/build/bin/llama-server" ]]; then
        ok "CUDA build at $CUDA_DIR/build/bin/llama-server"
    else
        fail "CUDA build produced no llama-server binary"
    fi

    install_bins "$CUDA_DIR/build/bin" llama-server llama-cli llama-bench
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

    install_bins "$SYCL_DIR/build/bin" llama-server llama-cli llama-bench
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

    step "Configure whisper.cpp ROCm (${AMDGPU_TARGETS})"
    # No ffmpeg: the orchestrator web UI sends 16 kHz mono WAV, which
    # whisper-server decodes natively. Add -DWHISPER_FFMPEG=ON if you ever
    # need to feed it arbitrary audio formats.
    cmake -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build whisper ROCm (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target whisper-server whisper-cli

    if [[ -x "$WHISPER_ROCM_DIR/build/bin/whisper-server" ]]; then
        ok "whisper ROCm build at $WHISPER_ROCM_DIR/build/bin/whisper-server"
    else
        fail "whisper ROCm build produced no whisper-server binary"
    fi

    install_bins "$WHISPER_ROCM_DIR/build/bin" whisper-server whisper-cli
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
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build whisper Vulkan (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target whisper-server whisper-cli

    if [[ -x "$WHISPER_VULKAN_DIR/build/bin/whisper-server" ]]; then
        ok "whisper Vulkan build at $WHISPER_VULKAN_DIR/build/bin/whisper-server"
    else
        fail "whisper Vulkan build produced no whisper-server binary"
    fi

    install_bins "$WHISPER_VULKAN_DIR/build/bin" whisper-server whisper-cli
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
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build whisper CUDA (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target whisper-server whisper-cli

    if [[ -x "$WHISPER_CUDA_DIR/build/bin/whisper-server" ]]; then
        ok "whisper CUDA build at $WHISPER_CUDA_DIR/build/bin/whisper-server"
    else
        fail "whisper CUDA build produced no whisper-server binary"
    fi

    install_bins "$WHISPER_CUDA_DIR/build/bin" whisper-server whisper-cli
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

    install_bins "$WHISPER_SYCL_DIR/build/bin" whisper-server whisper-cli
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

    # A venv moved with the checkout keeps a dead pip shebang (bin/python
    # still resolves via the system interpreter) — probe pip itself and
    # recreate when broken, not just when missing
    if ! "$PIPER_VENV/bin/pip" --version >/dev/null 2>&1; then
        rm -rf "$PIPER_VENV"
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

    install_link "$PIPER_VENV/bin/piper"
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

    # A venv moved with the checkout keeps a dead pip shebang (bin/python
    # still resolves via the system interpreter) — probe pip itself and
    # recreate when broken, not just when missing
    if ! "$TOOLS_VENV/bin/pip" --version >/dev/null 2>&1; then
        rm -rf "$TOOLS_VENV"
        python3 -m venv "$TOOLS_VENV"
    fi
    "$TOOLS_VENV/bin/pip" install --quiet --upgrade pip
    "$TOOLS_VENV/bin/pip" install --quiet --upgrade -r "$TOOLS_REQUIREMENTS"

    if [[ -x "$TOOLS_VENV/bin/hf" ]]; then
        ok "hf CLI at $TOOLS_VENV/bin/hf"
    else
        fail "llm-tools install produced no hf binary in $TOOLS_VENV/bin"
    fi

    install_link "$TOOLS_VENV/bin/hf"
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

    step "Configure stable-diffusion.cpp ROCm (${AMDGPU_TARGETS})"
    # Note: SD uses -DSD_HIPBLAS=ON (not GGML_HIP) and wants both GPU_TARGETS
    # and AMDGPU_TARGETS depending on ROCm version. Ninja is recommended.
    # rocWMMA is kept OFF for SD -- per AUR notes it regressed on ROCm 7+.
    cmake -B build \
        -G Ninja \
        -DSD_HIPBLAS=ON \
        -DGPU_TARGETS="$AMDGPU_TARGETS" \
        -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
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

    install_bins "$SD_ROCM_DIR/build/bin" sd sd-cli sd-server

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
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build SD Vulkan (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS"

    if [[ -x "$SD_VULKAN_DIR/build/bin/sd" ]] \
       || [[ -x "$SD_VULKAN_DIR/build/bin/sd-cli" ]]; then
        ok "SD Vulkan build at $SD_VULKAN_DIR/build/bin/"
    else
        fail "SD Vulkan build produced no sd/sd-cli binary"
    fi

    install_bins "$SD_VULKAN_DIR/build/bin" sd sd-cli sd-server

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
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build SD CUDA (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS"

    if [[ -x "$SD_CUDA_DIR/build/bin/sd" ]] \
       || [[ -x "$SD_CUDA_DIR/build/bin/sd-cli" ]]; then
        ok "SD CUDA build at $SD_CUDA_DIR/build/bin/"
    else
        fail "SD CUDA build produced no sd/sd-cli binary"
    fi

    install_bins "$SD_CUDA_DIR/build/bin" sd sd-cli sd-server

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

    install_bins "$SD_SYCL_DIR/build/bin" sd sd-cli sd-server

    build_sd_frontend "$SD_SYCL_DIR"
}

# -- CPU-only builds (no GPU backend) ------------------------------------------
# No GGML_HIP/VULKAN/CUDA/SYCL flags: llama.cpp/whisper.cpp/sd.cpp build their
# CPU backend with GGML_NATIVE=ON by default, and -march=$MARCH (default
# "native") unlocks the host's best SIMD paths (AVX2/AVX512, AMX on Xeon).
# Runtime threading is a llama-server flag (-t), not a build flag.
build_llama_cpu() {
    check_clone "$CPU_DIR" "https://github.com/ggml-org/llama.cpp"
    sync_repo "$CPU_DIR"
    cd "$CPU_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean CPU build dir"
        rm -rf build
    fi

    step "Configure CPU (-march=$MARCH)"
    cmake -B build \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build CPU (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

    if [[ -x "$CPU_DIR/build/bin/llama-server" ]]; then
        ok "CPU build at $CPU_DIR/build/bin/llama-server"
    else
        fail "CPU build produced no llama-server binary"
    fi

    install_bins "$CPU_DIR/build/bin" llama-server llama-cli llama-bench
}

build_whisper_cpu() {
    check_clone "$WHISPER_CPU_DIR" "https://github.com/ggml-org/whisper.cpp"
    sync_repo "$WHISPER_CPU_DIR"
    cd "$WHISPER_CPU_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean whisper CPU build dir"
        rm -rf build
    fi

    step "Configure whisper.cpp CPU (-march=$MARCH)"
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build whisper CPU (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS" --target whisper-server whisper-cli

    if [[ -x "$WHISPER_CPU_DIR/build/bin/whisper-server" ]]; then
        ok "whisper CPU build at $WHISPER_CPU_DIR/build/bin/whisper-server"
    else
        fail "whisper CPU build produced no whisper-server binary"
    fi

    install_bins "$WHISPER_CPU_DIR/build/bin" whisper-server whisper-cli
}

build_sd_cpu() {
    check_clone "$SD_CPU_DIR" "https://github.com/leejet/stable-diffusion.cpp" recursive
    sync_repo "$SD_CPU_DIR" recursive
    cd "$SD_CPU_DIR"

    if [[ $CLEAN -eq 1 ]]; then
        step "Clean SD CPU build dir"
        rm -rf build
    fi

    # Works, but expect minutes per image -- CPU SD is a fallback, not a plan.
    step "Configure stable-diffusion.cpp CPU (-march=$MARCH)"
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=$MARCH -O3" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -O3"

    step "Build SD CPU (-j${JOBS})"
    cmake --build build --config Release -j"$JOBS"

    if [[ -x "$SD_CPU_DIR/build/bin/sd" ]] \
       || [[ -x "$SD_CPU_DIR/build/bin/sd-cli" ]]; then
        ok "SD CPU build at $SD_CPU_DIR/build/bin/"
    else
        fail "SD CPU build produced no sd/sd-cli binary"
    fi

    install_bins "$SD_CPU_DIR/build/bin" sd sd-cli sd-server

    build_sd_frontend "$SD_CPU_DIR"
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
            elif rocminfo 2>/dev/null | grep -q "$AMDGPU_TARGETS"; then
                local n
                n=$(rocminfo 2>/dev/null | grep -c "Name:.*${AMDGPU_TARGETS}" || true)
                ok "rocminfo sees ${n} ${AMDGPU_TARGETS} device(s)"
            else
                warn "rocminfo doesn't list ${AMDGPU_TARGETS}. Check:"
                warn "  1. ROCm version supports your GPU (pacman -Qi rocm-hip-runtime)"
                warn "  2. User in render+video groups: groups | grep -E 'render|video'"
                warn "  3. amdgpu kernel module loaded: lsmod | grep amdgpu"
                warn "  4. Wrong target picked? Re-run with AMDGPU_TARGETS=<your gfx>"
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
    echo -e "${DIM}Install:  ${BIN_DIR}${NC}"
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
            llama-cpu)      build_llama_cpu ;;
            sd-cpu)         build_sd_cpu ;;
            whisper-cpu)    build_whisper_cpu ;;
            piper)          build_piper ;;
            tools)          build_llmtools ;;
        esac
    done

    local elapsed=$(( $(date +%s) - t0 ))
    echo
    echo -e "${GREEN}${BOLD}Done${NC} in ${elapsed}s"
    echo -e "Binaries installed to ${BOLD}${BIN_DIR}${NC} (add to PATH to use directly)"
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
            llama-cpu)      echo -e "  llama   CPU:    ${CPU_DIR}/build/bin/llama-server" ;;
            sd-cpu)         echo -e "  sd      CPU:    ${SD_CPU_DIR}/build/bin/{sd-cli,sd-server}" ;;
            whisper-cpu)    echo -e "  whisper CPU:    ${WHISPER_CPU_DIR}/build/bin/whisper-server" ;;
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
    # if/then (not `[[ ]] && echo`): a false trailing &&-list would make
    # main() — and the whole script — exit 1 after a successful build
    echo -e "${DIM}Quick tests:${NC}"
    if [[ " ${JOBS_LIST[*]} " == *" llama-rocm "* ]]; then
        echo -e "  ${ROCM_DIR}/build/bin/llama-server --list-devices"
    fi
    if [[ " ${JOBS_LIST[*]} " == *" sd-rocm "* ]]; then
        echo -e "  ${SD_ROCM_DIR}/build/bin/sd-cli --help | head -30"
    fi
    if [[ " ${JOBS_LIST[*]} " == *" whisper-rocm "* ]]; then
        echo -e "  ${WHISPER_ROCM_DIR}/build/bin/whisper-server -m ~/jaynet-models/whisper/ggml-small.bin --port 8097"
    fi
    if [[ " ${JOBS_LIST[*]} " == *" piper "* ]]; then
        echo -e "  echo 'hello world' | ${PIPER_VENV}/bin/piper --model ${PIPER_VOICE_DIR}/${PIPER_VOICE}.onnx --output_file /tmp/piper-test.wav"
    fi
    if [[ " ${JOBS_LIST[*]} " == *" tools "* ]]; then
        echo -e "  ${TOOLS_VENV}/bin/hf --version"
    fi
}

main
