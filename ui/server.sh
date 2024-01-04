# install vllm first: 
# git clone https://github.com/LZY-the-boys/vllm
# pip install -e .
eval "$(conda shell.bash hook)"

model=/data/outs/qwen-v1226

# ln -s /model/qwen-sharegpt4-merged lu-vae/qwen-sharegpt4-merged

source activate vllm # or vllm
export PYTHONPATH=.

LOG_FILE='./gpt.cciiplab.com.log' \
CUDA_VISIBLE_DEVICES=6 \
python server/openai_server.py \
--model $model \
--served-model-name qwen-v1226 \
--trust-remote-code \
--tensor-parallel-size 1 \
# --gpu-memory-utilization 0.4
# default start at  http://0.0.0.0:8000

# autossh 
# autossh -NR 8.134.19.195:8000:0.0.0.0:8000 -p22 root@8.134.19.195