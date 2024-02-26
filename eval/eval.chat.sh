cd ../ui

# model=qwen-v1221-dpo-lora bash server.sh &
model=qwen-v1221-dpo-lora task=alpaca_eval bash client.sh &
model=qwen-v1221-dpo-lora task=mt_bench bash client.sh &

cd -