#!/bin/bash
# ============================================================================
#  lib/rdna4-env.sh -- Shared environment for 2x R9700 (gfx1201) workarounds
#  Sourced by llama-serve.sh and sd-serve.sh
# ============================================================================

# Known issue: gfx1201 stays pegged at 100% util after idle under HIP
# (https://github.com/ROCm/ROCm/issues/5706). These mitigate it.
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-1}"
export RADV_DEBUG="${RADV_DEBUG:-nocompute}"

# Uncomment if hipBLASLt picks the wrong arch on gfx1201:
# export HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/lib/hipblaslt/library
# export ROCBLAS_USE_HIPBLASLT=1

# -- Apply GPU visibility based on backend and a single-or-CSV index --------
# Usage: apply_gpu_visibility <rocm|vulkan> <"0" | "0,1" | "">
apply_gpu_visibility() {
    local backend="$1"
    local devices="$2"
    [[ -z "$devices" ]] && return 0
    case "$backend" in
        rocm)
            export HIP_VISIBLE_DEVICES="$devices"
            export ROCR_VISIBLE_DEVICES="$devices"
            ;;
        vulkan)
            export GGML_VK_VISIBLE_DEVICES="$devices"
            ;;
    esac
}
