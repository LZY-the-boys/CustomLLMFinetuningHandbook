# 1. install vllm first: 
# git clone https://github.com/LZY-the-boys/vllm
# pip install -e .

source activate vllm

CUDA_VISIBLE_DEVICES=4,5 \
python openai_api_server.py \
--model lu-vae/qwen-openhermes-merged \
--served-model-name Qwen-openhermes \
--trust-remote-code \
--tensor-parallel-size 2 
# default start at  http://0.0.0.0:8000

# directly curl 
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "lu-vae/qwen-openhermes-merged",
"messages": [
{"role": "system", "content": ""},
{"role": "user", "content": "Write a Perl script that processes a log file and counts the occurrences of different HTTP status codes. The script should accept the log file path as a command-line argument and print the results to the console in descending order of frequency."}
]
}'| echo -e "$(cat)\n" >> tmp.json

# web ui
# TODO: simple gradio version

# current web-ui version: https://github.com/enricoros/big-agi