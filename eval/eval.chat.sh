cd ../ui

bash server.sh &
model=qwen-v1226 task=alpaca_eval bash client.sh &
model=qwen-v1226 task=mt_bench bash client.sh &

cd -