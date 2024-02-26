
# install vllm first: 
# git clone https://github.com/LZY-the-boys/vllm
# pip install -e .
eval "$(conda shell.bash hook)"

: ${model:='qwen-v1221'}
name=lu-vae/$model

# ln -s /model/qwen-sharegpt4-merged lu-vae/qwen-sharegpt4-merged
# ln -s /model/qwen-v1221-lora-merged lu-vae/qwen-v1221-lora-merged
# ln -s /model/Qwen-14B Qwen/Qwen-14B

source activate vllm # or vllm
export PYTHONPATH=.

# 1.
LOG_FILE='./gpt.cciiplab.com.log' \
CUDA_VISIBLE_DEVICES=6,7 \
python server/openai_server.py \
--model /model/qwen-v1231-merged \
--served-model-name lu-vae/qwen-v1231 \
--trust-remote-code \
--tensor-parallel-size 2 \
--gpu-memory-utilization 0.9

# 实际占40%
# default start at  http://0.0.0.0:8000

# 2. 
# autossh -NR 8.134.19.195:8000:0.0.0.0:8000 -p22 root@8.134.19.195