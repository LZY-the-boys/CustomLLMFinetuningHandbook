source activate vllm
export PYTHONPATH=.

# name=lu-vae/qwen-sharegpt4-merged
name=lu-vae/qwen-sharegpt_zh

wait_port_available 8000

python client/openai_client.py \
--model $name \
--func alpaca_eval &

python client/openai_client.py \
--model $name \
--func mt_bench

# 不能叫gradio_client，会导致circular import
# python client/gradio_web.py \
# --model lu-vae/qwen-sharegpt4-merged \
# --server-port 8080

# python client/openai_client.py \
# --model lu-vae/qwen-sharegpt4-merged \
# --func mt_bench
