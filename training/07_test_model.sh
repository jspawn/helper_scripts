llama-server \
  -m /srv/llama/models/qwen3-8b-finance/unsloth.Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 4096 --jinja \
  --reasoning-format none \
  -fa on --temp 0.6 --top-k 20 --top-p 0.95