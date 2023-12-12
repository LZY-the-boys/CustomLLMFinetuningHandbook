CUDA_VISIBLE_DEVICES=0,1,2,3 \
model=mistralai/Mistral-7B-v0.1 \
out_dir=$LZY_HOME/CCIIP-GPT/eval/zephyr-7b-dpo-lora \
peft=lu-vae/zephyr-7b-dpo-lora \
bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh