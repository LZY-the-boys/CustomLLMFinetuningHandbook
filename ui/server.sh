# install vllm first: 
# git clone https://github.com/LZY-the-boys/vllm
# pip install -e .
eval "$(conda shell.bash hook)"

model=qwen-sharegpt4
name=lu-vae/$model

# ln -s /model/qwen-sharegpt4-merged lu-vae/qwen-sharegpt4-merged

source activate vllm # or vllm
export PYTHONPATH=.

# conda activate lla
# cd $LZY_HOME/lora-merge/hard-moe
# CUDA_VISIBLE_DEVICES=0 \
# python merge.py \
# --model-path Qwen/Qwen-14B \
# --peft-paths $OUT_ROOT/$model
# cd -

# mkdir /home/lzy/CCIIP-GPT/ui/lu-vae/
# ln -s /model/qwen-sharegpt4-merged ui/lu-vae/qwen-sharegpt4-merged

LOG_FILE='./gpt.cciiplab.com.log' \
CUDA_VISIBLE_DEVICES=4,5 \
python server/openai_server.py \
--model $name-merged \
--served-model-name $name-merged \
--trust-remote-code \
--tensor-parallel-size 2 \
# --gpu-memory-utilization 0.4
# default start at  http://0.0.0.0:8000

# autossh 
# autossh -NR 8.134.19.195:8000:0.0.0.0:8000 -p22 root@8.134.19.195