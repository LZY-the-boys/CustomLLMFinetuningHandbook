source $LZY_HOME/task.sh

function start1(){
dir="$DATA_ROOT"

for file in "$dir"/*; do
    echo $file
    if [[ $file == *.json* ]]; then
        tasks+=("data_path='$file;sharegpt' bash $home/axolotl/src/run_qwen.sh")
    fi
done
}

function start2(){
datas=(
# "$DATA_ROOT/ultrachat_200k.json"
"Open-Orca/SlimOrca-Dedup"
"$DATA_ROOT/sharegpt_zh.json"
# /home/LeiFeng/lzy/data-mining/instruct_mining/data/platypus-200/processed_evaluator_v1_200.jsonl
# /home/LeiFeng/lzy/data-mining/instruct_mining/data/platypus-200/processed_evaluator_v2_200.jsonl
# /home/LeiFeng/lzy/data-mining/instruct_mining/data/platypus-200/processed_randomsampling_200.jsonl
)
for file in "${datas[@]}"; do
    tasks+=("data_path='$file;sharegpt' bash $home/axolotl/src/run_qwen.sh")
done
}

function eval(){
loras=(
$OUT_ROOT/qwen-sharegpt4
# $OUT_ROOT/qwen-sharegpt_zh
# $OUT_ROOT/qwen-ultrachat_200k
# $OUT_ROOT/qwen-SlimOrca-Dedup
)
for file in "${loras[@]}"; do
    tasks+=("CUDA_VISIBLE_DEVICES=4,5,6,7 peft=$file model=Qwen/Qwen-14B out_dir=$(pwd)/$(basename $file) bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh")
done
}


# PORT=$(shuf -i7000-9000 -n1) 

# run_tasks_parallel start2
run_tasks_sequential eval