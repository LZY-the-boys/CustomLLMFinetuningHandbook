
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

adapter=qwen-v1223

CUDA_VISIBLE_DEVICES=0 \
python lora_merge.py \
--model-path Qwen/Qwen-14B-Chat \
--peft-paths $OUT_ROOT/$adapter \
--hub lu-vae