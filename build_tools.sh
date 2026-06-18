#!/bin/bash
# ============================================================================
#  build_tools.sh -- Build llama.cpp and stable-diffusion.cpp
#  Hardware: 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4) / 7950X (znver4)
#
#  Usage:
#    ./build_tools.sh                    # build everything (llama + sd, both backends)
#    ./build_tools.sh rocm               # llama+sd, ROCm only
#    ./build_tools.sh vulkan             # llama+sd, Vulkan only
#    ./build_tools.sh llama              # llama.cpp, both backends
#    ./build_tools.sh sd                 # stable-diffusion.cpp, both backends
#    ./build_tools.sh llama rocm         # just llama.cpp ROCm
#    ./build_tools.sh sd vulkan          # just stable-diffusion.cpp Vulkan
#    ./build_tools.sh --clean            # nuke build dirs first
# ============================================================================

set -euo pipefail

# -- Paths ------------------------------------------------------------------
ROCM_DIR="/srv/llama/llama.cpp-rocm"
VULKAN_DIR="/srv/llama/llama.cpp-vulkan"
SD_ROCM_DIR="/srv/llama/stable-diffusion.cpp-rocm"
SD_VULKAN_DIR="/srv/llama/stable-diffusion.cpp-vulkan"
# Shared install location for the SD web UI frontend. The Vite project lives
# inside each SD source tree at examples/server/frontend, but the built
# artifacts are copied here so both backends use the same canonical path
# via --serve-html-path.
SD_FRONTEND_INSTALL_DIR="/srv/llama/stable-diffusion/web/server/frontend"
SD_FRONTEND_HTML="${SD_FRONTEND_INSTALL_DIR}/dist/index.html"
JOBS="${JOBS:-32}"

# -- Args -------------------------------------------------------------------
# Parse projects (llama, sd) and backends (rocm, vulkan) independently.
# If only backends specified, both projects are built. If only projects
# specified, both backends are built. If neither, both x both.
PROJECTS=()
BACKENDS=()
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        llama|sd)    PROJECTS+=("$arg") ;;
        rocm|vulkan) BACKENDS+=("$arg") ;;
        --clean|-c)  CLEAN=1 ;;
        -h|--help)
            sed -n '3,16p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done
[[ ${#PROJECTS[@]} -eq 0 ]] && PROJECTS=(llama sd)
[[ ${#BACKENDS[@]} -eq 0 ]] && BACKENDS=(rocm vulkan)

# Build a flat list of (project, backend) jobs to run
JOBS_LIST=()
for p in "${PROJECTS[@]}"; do
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
    local checked_rocm=0 checked_vulkan=0
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
    done
}

# -- Main -------------------------------------------------------------------
main() {
    echo -e "${BOLD}llama.cpp + stable-diffusion.cpp build${NC}"
    echo -e "${DIM}Projects: ${PROJECTS[*]} | Backends: ${BACKENDS[*]} | Clean: ${CLEAN} | Jobs: ${JOBS}${NC}"
    echo -e "${DIM}Plan:     ${JOBS_LIST[*]}${NC}"

    preflight

    local t0
    t0=$(date +%s)

    for job in "${JOBS_LIST[@]}"; do
        case "$job" in
            llama-rocm)   build_llama_rocm ;;
            llama-vulkan) build_llama_vulkan ;;
            sd-rocm)      build_sd_rocm ;;
            sd-vulkan)    build_sd_vulkan ;;
        esac
    done

    local elapsed=$(( $(date +%s) - t0 ))
    echo
    echo -e "${GREEN}${BOLD}Done${NC} in ${elapsed}s"
    echo
    for job in "${JOBS_LIST[@]}"; do
        case "$job" in
            llama-rocm)   echo -e "  llama  ROCm:   ${ROCM_DIR}/build/bin/llama-server" ;;
            llama-vulkan) echo -e "  llama  Vulkan: ${VULKAN_DIR}/build/bin/llama-server" ;;
            sd-rocm)      echo -e "  sd     ROCm:   ${SD_ROCM_DIR}/build/bin/{sd-cli,sd-server}" ;;
            sd-vulkan)    echo -e "  sd     Vulkan: ${SD_VULKAN_DIR}/build/bin/{sd-cli,sd-server}" ;;
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
}

main
