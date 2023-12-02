
source activate lla
cd $LZY_HOME/lora-merge/hard-moe
CUDA_VISIBLE_DEVICES=0 \
python merge.py \
--model-path Qwen/Qwen-14B \
--peft-paths $OUT_ROOT/qwen-sharegpt4 
# --peft-paths $OUT_ROOT/qwen-sharegpt_zh 
# --peft-paths lu-vae/qwen-openhermes

cd -