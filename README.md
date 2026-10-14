# helper_scripts

Build helper and example presets for running local LLM / diffusion / speech tooling — the companion to [JayNet](https://github.com/jspawn/jaynet_orchestrator), but usable on its own.

`build_tools.sh` builds llama.cpp, stable-diffusion.cpp, whisper.cpp (ROCm, Vulkan, CUDA, SYCL, or CPU-only), piper TTS, OmniVoice TTS, and a Python tooling venv. Run it without args for full usage.

## First build

```
./build_tools.sh llama vulkan   # AMD/NVIDIA/Intel, no vendor SDK (default backend)
./build_tools.sh llama rocm     # AMD native (asks for your gfx target)
./build_tools.sh llama cuda     # NVIDIA (asks for the CUDA arch)
```

At startup the script asks where to install the built binaries (default `~/jaynet-bin`; set `BIN_DIR` to skip the prompt). Builds install into **versioned prefixes** (`$BIN_DIR/llama.cpp.rocm-b10343/`) with a stable symlink flipped after each build — a rebuild never pulls libraries out from under a running server, and rollback is one `ln -sf`.

**JayNet wiring:** register the stable symlinks as binaries — Admin → Models → Presets → Binaries, or `models.binaries` in `runtime.yaml`, e.g. `rocm: {path: ~/jaynet-bin/llama.cpp.rocm/bin/llama-server, device_env: HIP_VISIBLE_DEVICES}`. Presets then pick a build by name.

## Day to day

- `./build_tools.sh update` — rebuild everything already built. Vanilla trees sync to latest upstream; the forks (K2-Horizon, Spark, Bonsai, OmniVoice) stay on **pinned, reviewed commits** — bump the `*_COMMIT` vars in the script after reviewing upstream changes. The K2 patches are pinned to a fixed Hugging Face revision and sha256-checked.
- `./build_tools.sh delete` — interactive picker showing size and running processes before removing a build.
- Parallelism defaults to `nproc`; override with `JOBS=16`.

## Models

This repo does not ship a model downloader. Use JayNet's `scripts/pull-model` or its admin downloader, or the `hf` CLI from `./build_tools.sh tools`.

## Presets

`example_presets/` holds example llama-server configs for JayNet. Adjust `MODEL_PATH`/`MMPROJ` to your model dir and `TOOLS_TEMPLATE` to your JayNet checkout (`presets/chat_templates/`) before use.

Works on any GPU; install hints cover Arch (pacman) and Debian/Ubuntu (apt). ROCm builds ask for your AMD gfx target (menu lists the common homelab cards).

Upstream source trees (`llama.cpp-*`, `stable-diffusion.cpp-*`, …) are cloned by the build script and intentionally not part of this repo.

## License

MIT — see [LICENSE](LICENSE).
