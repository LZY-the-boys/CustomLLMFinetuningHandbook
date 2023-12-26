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
# "Open-Orca/SlimOrca-Dedup"
# "$DATA_ROOT/sharegpt_zh.json"
# "$DATA_ROOT/gsm8k_zh.json"
# /home/LeiFeng/lzy/data-mining/instruct_mining/data/platypus-200/processed_evaluator_v1_200.jsonl
# /home/LeiFeng/lzy/data-mining/instruct_mining/data/platypus-200/processed_evaluator_v2_200.jsonl
# /home/LeiFeng/lzy/data-mining/instruct_mining/data/platypus-200/processed_randomsampling_200.jsonl
)
yaml_file=(
    /home/LeiFeng/lzy/CCIIP-GPT/train/v1225.yml
)
# for file in "${datas[@]}"; do
#     tasks+=("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 data_path='$file;sharegpt' bash $home/axolotl/src/run_qwen.sh")
# done
# for file in "${yaml_file[@]}"; do
#     tasks+=("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 yaml_file='$file' bash $home/axolotl/src/run_qwen.sh")
# done
# ,1,2,3,4,5,6,7
for file in "${yaml_file[@]}"; do
    tasks+=("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 yaml_file='$file' bash $home/axolotl/src/run_full.sh")
done
}

function eval(){
loras=(
# $OUT_ROOT/qwen-sharegpt4
# $OUT_ROOT/qwen-sharegpt_zh
$OUT_ROOT/qwen-gsm8k_zh
# $OUT_ROOT/qwen-ultrachat_200k
# $OUT_ROOT/qwen-SlimOrca-Dedup
)
for file in "${loras[@]}"; do
    tasks+=("CUDA_VISIBLE_DEVICES=4,5,6 peft=$file model=Qwen/Qwen-14B out_dir=$(pwd)/$(basename $file) bash $LZY_HOME/lm-evaluation-harness-leaderboard/eval.sh")
done
}


# PORT=$(shuf -i7000-9000 -n1) 

run_tasks_sequential start2
# run_tasks_sequential eval