# install vllm first: 
# git clone https://github.com/LZY-the-boys/vllm
# pip install -e .
eval "$(conda shell.bash hook)"

# name=lu-vae/qwen-sharegpt4
model=qwen-sharegpt_zh
name=lu-vae/$model

# conda activate lla
# CUDA_VISIBLE_DEVICES=0 \
# python server/lora_merge.py \
# --model-path Qwen/Qwen-14B \
# --peft-paths $OUT_ROOT/$model \
# --local /model \
# --hub lu-vae \
# cd -

source activate vllm
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,3 \
python server/openai_server.py \
--model $name-merged \
--served-model-name $name-merged \
--trust-remote-code \
--tensor-parallel-size 2 
# default start at  http://0.0.0.0:8000

# directly curl 