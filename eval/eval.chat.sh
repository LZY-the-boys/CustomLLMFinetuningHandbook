cd ../ui

bash server.sh &
task=alpaca_eval bash client.sh &
task=mt_bench bash client.sh &

cd -