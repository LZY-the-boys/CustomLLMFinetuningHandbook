# install vllm first: 
# git clone https://github.com/LZY-the-boys/vllm
# pip install -e .

source activate lla
export PYTHONPATH=.

# conda activate lla
# cd $LZY_HOME/lora-merge/hard-moe
# CUDA_VISIBLE_DEVICES=0 \
# python merge.py \
# --model-path Qwen/Qwen-14B \
# --peft-paths $OUT_ROOT/$model
# cd -

LOG_FILE='./gpt.cciiplab.com.log' \
CUDA_VISIBLE_DEVICES=2,3 \
python server/openai_server.py \
--model lu-vae/qwen-sharegpt4-merged \
--served-model-name lu-vae/qwen-sharegpt4-merged \
--trust-remote-code \
--tensor-parallel-size 2 
# default start at  http://0.0.0.0:8000

# autossh 
# autossh -NR 8.134.19.195:8000:0.0.0.0:8000 -p22 root@8.134.19.195