source $LZY_HOME/task.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 \
model=Qwen/Qwen-14B \
out_dir=$LZY_HOME/CCIIP-GPT/eval/qwen-v1219 \
peft=$OUT_ROOT/qwen-v1219 \
bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh

# task=gsm8k \