cd ../ui

bash server.sh &
model=qwen-v1225 task=alpaca_eval bash client.sh &
model=qwen-v1225 task=mt_bench bash client.sh &

cd -