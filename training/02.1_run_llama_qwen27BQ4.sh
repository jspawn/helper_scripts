llama-server \
  -m /srv/llama/models/Qwen3.5-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 8192 --jinja \
  --reasoning-format none \
  --temp 0.7 --top-k 20 --top-p 0.95