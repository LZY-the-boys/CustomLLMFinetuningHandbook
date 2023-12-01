CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
--model lu-vae/qwen-openhermes-merged \
--trust-remote-code
# default start at  http://0.0.0.0:8000

curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "lu-vae/qwen-openhermes-merged",
"messages": [
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "Who won the world series in 2020?"}
]
}'