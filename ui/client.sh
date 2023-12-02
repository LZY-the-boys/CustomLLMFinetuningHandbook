source activate vllm
export PYTHONPATH=.

# python client/openai_client.py \
# --model lu-vae/qwen-sharegpt4-merged \
# --func alpaca_eval

# 不能叫gradio_client，会导致circular import
python client/gradio_web.py \
--model lu-vae/qwen-sharegpt4-merged \
--server-port 8080