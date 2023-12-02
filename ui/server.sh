# install vllm first: 
# git clone https://github.com/LZY-the-boys/vllm
# pip install -e .

source activate vllm
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0,3 \
python server/openai_server.py \
--model lu-vae/qwen-sharegpt4-merged \
--served-model-name lu-vae/qwen-sharegpt4-merged \
--trust-remote-code \
--tensor-parallel-size 2 
# default start at  http://0.0.0.0:8000

# directly curl 