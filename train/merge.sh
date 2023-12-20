
source activate lla
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 \
python lora_merge.py \
--model-path Qwen/Qwen-14B \
--peft-paths $OUT_ROOT/qwen-v1219 \
--hub lu-vae
# --peft-paths $OUT_ROOT/qwen-sharegpt4 
# --peft-paths $OUT_ROOT/qwen-sharegpt_zh 
# --peft-paths lu-vae/qwen-openhermes