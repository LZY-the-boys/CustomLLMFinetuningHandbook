source $LZY_HOME/task.sh

# adapter=qwen-v1221

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# model=Qwen/Qwen-14B \
# out_dir=$LZY_HOME/CCIIP-GPT/eval/$adapter \
# peft=$OUT_ROOT/$adapter \
# bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh

# adapter=qwen-v1223
# model=/model/Qwen-14B-Chat

CUDA_VISIBLE_DEVICES=0,1 \
model=/model/Qwen-14B \
out_dir=$LZY_HOME/CCIIP-GPT/eval/openllmleaderboard/qwen-v1221-dpo-lora \
peft=/home/lzy/lzy/CCIIP-GPT/eval/qwen-v1221-dpo-lora \
bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh
# task=gsm8k \



function run_full(){
# name=(
#     # qwen-v1226/checkpoint-3105
#     # qwen-v1226/checkpoint-4657
#     # qwen-v1226/checkpoint-6210
#     /data/outs/qwen-v1226-dpo-lora
# )
# for n in "${name[@]}";do
#     out=$(echo "$n" | tr '/' '_')

    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    model=/data/outs/qwen-v1226 \
    peft=/data/outs/qwen-v1226-dpo-lora \
    out_dir=$LZY_HOME/CCIIP-GPT/eval/openllmleaderboard/qwen-v1226-dpo-lora \
    bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh

# done

}

run_full