export PYTHONPATH=.
N=200

python process.py \
--dataset-path garage-bAInd/Open-Platypus \
--top-n $N \
--out-path data/platypus-$N \
--eval-method evaluator_v2
