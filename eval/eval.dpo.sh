dir=/data/outs/qwen-v1231-dpo-lora
# 400 1200 2000 2800 3600
for num in 5200 6400 7600;do

peft="$dir/checkpoint-$num"
CUDA_VISIBLE_DEVICES=4,5,6,7 \
model=/data/outs/qwen-v1226 \
out_dir=$LZY_HOME/CCIIP-GPT/eval/qwen-v1231-dpo-lora-$num \
peft=$peft \
bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh

done