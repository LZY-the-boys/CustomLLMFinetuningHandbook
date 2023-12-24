cd ../ui

model=qwen-v1223 bash server.sh &
model=qwen-v1223 task=alpaca_eval bash client.sh &
model=qwen-v1223 task=mt_bench bash client.sh &

cd -