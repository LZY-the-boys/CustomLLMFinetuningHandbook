from datasets import Dataset, load_dataset
import metrics
import os, sys

def evaluator_v1(
    dataset: Dataset,
    prompt_key: str,
    completion_key: str,
    input_key: str = "_input",
    knn_k: int = 6,
) -> Dataset:

    # calculate len.
    dataset = metrics.get_len(
        dataset=dataset, 
        input_key=input_key, 
        model_name="meta-llama/Llama-2-7b-hf",
    )
    # calculate reward score.
    dataset = metrics.get_reward_score(
        dataset=dataset,
        prompt_key=prompt_key,
        completion_key=completion_key,
        model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
    )
    # calculate knn6.
    dataset = metrics.get_knn_score(
        dataset, 
        input_key, 
        knn_k
    )
    # calculate expected loss from instruct mining.
    dataset = dataset.map(
        lambda x: {
            "expected_loss": 1.0684
            - 0.1498 * x["reward"]
            + 8.257 * 10 ** (-5) * x["len"]
            - 0.9350 * x["knn"]
        },
        num_proc=os.getenv('num_proc')
    )
    return dataset, [
        "reward",
        "len",
        "knn",
    ]

def evaluator_v2(
    dataset: Dataset,
    prompt_key: str,
    completion_key: str,
    input_key: str = "_input",
    knn_k: int = 6,
) -> Dataset:

    # calculate reward score.
    dataset = metrics.get_reward_score(
        dataset=dataset,
        prompt_key=prompt_key,
        completion_key=completion_key,
        model_name="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
    )
    dataset = metrics.get_unieval_score(
        dataset=dataset,
        prompt_key=prompt_key,
        completion_key=completion_key,
        model_name="MingZhong/unieval-dialog",
    )
    # calculate expected loss from instruct mining.
    dataset = dataset.map(
        lambda x: {
            "expected_loss": 0.0274
            - 0.0078 * x["reward"]
            + 0.4421 * x["understandability"]
            - 0.3212 * x["naturalness"]
            - 0.1520 * x["coherence"]
        },
        num_proc=os.getenv('num_proc')
    )
    return dataset, [
        "reward",
        "understandability",
        "naturalness",
        "coherence"
    ]


def main(
    *, 
    out_path: str= "platypus",
    dataset_path: str= "garage-bAInd/Open-Platypus",
    prompt_key: str= "instruction",
    completion_key: str= "output",
    input_key: str= "tmp",
    top_n: int=200,
    eval_method: str='evaluator_v2',
):

    dataset = load_dataset(dataset_path, split="train")
    # prepare input of a model.
    dataset = dataset.map(
        lambda x: {
            input_key: f"###Instruction:\n{x[prompt_key]}###Response:\n{x[completion_key]}"
        } 
    )
    
    dataset, score_keys = getattr(sys.modules['__main__'], eval_method)(
        dataset=dataset,
        prompt_key=prompt_key,
        completion_key=completion_key,
        input_key=input_key,
    )
    
    dataset = (
        dataset
        .sort("expected_loss")
        .select(range(top_n))
    )
    # standardize the dataset.
    SELECTED_COLS = [
        prompt_key,
        completion_key,
        "expected_loss",
    ] + score_keys
    dataset = dataset.select_columns(SELECTED_COLS)

    # save the final dataset as `jsonl`.
    dataset.to_json(f"{out_path}/processed_{eval_method}_{top_n}.jsonl")

if __name__ == '__main__':
    import defopt
    try:
        defopt.run(main)
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
