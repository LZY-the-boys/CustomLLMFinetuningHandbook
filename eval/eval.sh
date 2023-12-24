source $LZY_HOME/task.sh

# adapter=qwen-v1221

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# model=Qwen/Qwen-14B \
# out_dir=$LZY_HOME/CCIIP-GPT/eval/$adapter \
# peft=$OUT_ROOT/$adapter \
# bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh

adapter=qwen-v1223

CUDA_VISIBLE_DEVICES=0,1,2,3 \
model=Qwen/Qwen-14B-Chat \
out_dir=$LZY_HOME/CCIIP-GPT/eval/$adapter \
peft=$OUT_ROOT/$adapter \
bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh
# task=gsm8k \