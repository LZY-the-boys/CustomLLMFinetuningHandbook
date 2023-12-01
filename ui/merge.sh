cd $LZY_HOME/lora-merge/hard-moe
CUDA_VISIBLE_DEVICES=0 \
python merge.py \
--model-path Qwen/Qwen-14B \
--peft-paths lu-vae/qwen-openhermes 