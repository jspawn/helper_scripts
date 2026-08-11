# helper_scripts

Helper script and presets for running local LLM / diffusion / speech tooling.

- `build_tools.sh` — builds llama.cpp, stable-diffusion.cpp, whisper.cpp (ROCm, Vulkan, CUDA, SYCL), plus piper TTS and a Python tooling venv. Run without args for usage. At startup it asks where to install the built binaries (default `~/jaynet-bin`; set `BIN_DIR` to skip the prompt).
- `example_presets/` — example llama-server config presets

Hardware target is AMD RDNA4 (gfx1201), but the build script supports NVIDIA and Intel backends too.

Upstream source trees (`llama.cpp-*`, `stable-diffusion.cpp-*`, `whisper.cpp-*`) are cloned by the build script and intentionally not part of this repo.

## License

MIT — see [LICENSE](LICENSE).
