CUDA_VISIBLE_DEVICES=4,5,6,7 \
model=Qwen/Qwen-14B \
out_dir=$LZY_HOME/CCIIP-GPT/eval/qwen-14B \
bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh
# task=drop \
# peft=lu-vae/qwen-openhermes \