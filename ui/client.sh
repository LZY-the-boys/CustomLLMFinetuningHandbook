source activate vllm
export PYTHONPATH=.

# name=lu-vae/qwen-sharegpt4-merged
# name=lu-vae/qwen-sharegpt_zh
name=lu-vae/qwen-v1219-merged
: ${task:='alpaca_eval'}

wait_port_available 8000

# python client/openai_client.py \
# --model $name \
# --func $task 

# python client/openai_client.py \
# --model $name \
# --func mt_bench

# 不能叫gradio_client，会导致circular import
# python client/gradio_web.py \
# --model $name \
# --server-port 8080


python client/openai_client.py \
--model $name \
--func cli_demo