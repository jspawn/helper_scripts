# helper_scripts

Helper scripts and presets for running local LLM / diffusion / speech tooling.

- `build_tools.sh` — builds llama.cpp, stable-diffusion.cpp, whisper.cpp (ROCm, Vulkan, CUDA, SYCL), plus piper TTS and a Python tooling venv. Run without args for usage.
- `llama-serve.sh`, `sd-serve.sh`, `serve.sh` — service launchers
- `hf-download.sh` — Hugging Face model downloads
- `presets/` — llama-server config presets
- `lib/rdna4-env.sh` — GPU environment tweaks

Hardware target is AMD RDNA4 (gfx1201), but the build script supports NVIDIA and Intel backends too.

Upstream source trees (`llama.cpp-*`, `stable-diffusion.cpp-*`, `whisper.cpp-*`) are cloned by the build script and intentionally not part of this repo.

## License

MIT — see [LICENSE](LICENSE).
