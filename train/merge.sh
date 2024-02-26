
# adapter=qwen-v1221

# source activate lla
# export PYTHONPATH=.
# CUDA_VISIBLE_DEVICES=0 \
# python lora_merge.py \
# --model-path Qwen/Qwen-14B \
# --peft-paths $OUT_ROOT/$adapter \
# --hub lu-vae
# --peft-paths $OUT_ROOT/qwen-sharegpt4 
# --peft-paths $OUT_ROOT/qwen-sharegpt_zh 
# --peft-paths lu-vae/qwen-openhermes


source activate lla
export PYTHONPATH=.


CUDA_VISIBLE_DEVICES=0,1 \
python lora_merge.py \
--name qwen-v1231 \
--model-path /home/lzy/.cache/huggingface/hub/models--lu-vae--qwen-v1226/snapshots/bc21fc6f26496ac824f2aad4a98c15691f80db70  \
--peft-paths /home/lzy/.cache/huggingface/hub/models--lu-vae--qwen-v1231-dpo-lora/snapshots/95ed062dcbeff197a53f2c785934cce65d6bae37/ \
--local /model 
# --hub lu-vae \