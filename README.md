# helper_scripts

Helper script and presets for running local LLM / diffusion / speech tooling.

- `build_tools.sh` — builds llama.cpp, stable-diffusion.cpp, whisper.cpp (ROCm, Vulkan, CUDA, SYCL, or CPU-only), plus piper TTS and a Python tooling venv. Run without args for usage. At startup it asks where to install the built binaries (default `~/jaynet-bin`; set `BIN_DIR` to skip the prompt), then which backends to build (default **Vulkan** — runs on AMD/NVIDIA/Intel without a vendor SDK; pass `rocm`/`cuda`/`sycl`/`cpu` as args to skip the menu). ROCm/CUDA picks ask for the GPU target (`AMDGPU_TARGETS` / `CUDA_ARCHS` env vars skip those prompts too); CPU builds tune `-march` via `MARCH` (default `native`). `update` rebuilds everything already built — it maps the source trees/venvs present in the checkout back to jobs, syncs each to latest, and reinstalls (the K2-Horizon fork keeps its pinned commit by design; add `--clean` for from-scratch rebuilds).
- `example_presets/` — example llama-server config presets

Works on any GPU; ROCm builds ask for your AMD gfx target (menu lists the common homelab cards).

Upstream source trees (`llama.cpp-*`, `stable-diffusion.cpp-*`, `whisper.cpp-*`) are cloned by the build script and intentionally not part of this repo.

## License

MIT — see [LICENSE](LICENSE).
