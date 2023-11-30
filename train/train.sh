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
"$DATA_ROOT/ultrachat_200k.json"
# "Open-Orca/SlimOrca-Dedup;sharegpt"
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
$OUT_ROOT/Llama-2-7b-hf/llama-processed_evaluator_v1_200
$OUT_ROOT/Llama-2-7b-hf/llama-processed_evaluator_v2_200
$OUT_ROOT/Llama-2-7b-hf/llama-processed_randomsampling_200
)
for file in "${loras[@]}"; do
    tasks+=("peft=$file out_dir=$(pwd)/$(basename $file) bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh")
done
}


# PORT=$(shuf -i7000-9000 -n1) 

run_tasks_parallel start2
# run_tasks_sequential start2