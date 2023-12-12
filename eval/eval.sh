source $LZY_HOME/task.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
model=Qwen/Qwen-14B \
out_dir=$LZY_HOME/CCIIP-GPT/eval/qwen-sharegpt4 \
task=gsm8k \
peft=$OUT_ROOT/qwen-sharegpt4 \
bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh