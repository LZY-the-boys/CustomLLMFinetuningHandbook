source activate vllm
export PYTHONPATH=.

python client/openai_client.py \
--model lu-vae/qwen-sharegpt4-merged \
--func alpaca_eval
