# import: standard
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import Dataset
import os
import sentence_transformers
# KNN
from pynndescent import NNDescent
# from sklearn.neighbors import KDTree
# https://pynndescent.readthedocs.io/en/stable/how_to_use_pynndescent.html

@torch.inference_mode()
def get_knn_score(
    dataset: Dataset,
    input_key: str,
    k: int = 6,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dataset:

    model = sentence_transformers.SentenceTransformer(model_name).cuda()
    embedded_dataset = model.encode(dataset[input_key])
    index = NNDescent(embedded_dataset)
    neighbor_graph = index.neighbor_graph[1]

    dataset = dataset.add_column(
        name=f"knn",
        column=neighbor_graph[:, k],
    )
    return dataset

@torch.inference_mode()
def get_len(
    dataset: Dataset,
    input_key: str,
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> Dataset:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # TODO: optimize this calculation.
    dataset = dataset.map(
        lambda x: {
            "len": tokenizer(x[input_key], return_tensors="pt")['input_ids'].shape[1] 
        },
        num_proc=os.getenv('num_proc')
    )
    return dataset

@torch.inference_mode()
def get_reward_score(
    dataset: Dataset,
    prompt_key: str,
    completion_key: str,
    model_name: str = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
    # OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5
    # "OpenAssistant/reward-model-deberta-v3-large-v2"
) -> Dataset:

    if 'oasst-rm' in model_name:
        import oasst_patch
    rank_model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def calculate_reward_score(prompt,completion):
        inputs = tokenizer(prompt, completion, return_tensors="pt", padding=True, truncation=True).to('cuda:0')
        # negative number
        score = rank_model(**inputs).logits[:,0].cpu().tolist()
        return score

    dataset = dataset.map(
        lambda x: {
            "reward": calculate_reward_score(
                prompt=x[prompt_key],
                completion=x[completion_key],
            )
        },
        batched=True,
        batch_size=4, # more cannot increase the speed
    )

    return dataset

@torch.inference_mode()
def get_unieval_score(
    dataset: Dataset,
    prompt_key: str,
    completion_key: str,
    model_name: str = "MingZhong/unieval-dialog",
) -> Dataset:

    eval_prompt = {
        'naturalness': 'question: Is this a natural response in the dialogue? </s> response: {output}', 
        'coherence': 'question: Is this a coherent response given the dialogue history? </s> response: {output} </s> dialogue history: {history}',
        'understandability': 'question: Is this an understandable response in the dialogue? </s> response: {output}',
        'engagingness': 'question: Is this a coherent response given the dialogue history? </s> response: {output} </s> dialogue history: {history} </s> fact: {fact}',
        'groundedness': 'question: Is this response consistent with knowledge in the fact? </s> response: {output} </s> fact: {fact}',
    }
    rank_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pos_id = tokenizer("Yes")["input_ids"][0]
    neg_id = tokenizer("No")["input_ids"][0]

    def calculate_score(prompt):
        # max-length=1024 ?
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to('cuda')
        labels = torch.full((len(prompt),1), rank_model.config.decoder_start_token_id,device='cuda')
        # need bos?
        score = rank_model(**inputs, labels=labels).logits.view(-1, rank_model.config.vocab_size).cpu()
        score = torch.nn.functional.softmax(score.float()) # softmax must be float32
        pos_score, neg_score = score[:, pos_id].numpy(), score[:, neg_id].numpy()
        return pos_score / (pos_score + neg_score)

    for key in [ 'understandability', 'naturalness' , 'coherence' ]:
        dataset = dataset.map(
            lambda x: {
                f'{key}_prompt' : eval_prompt[key].format_map({
                    'output': x[completion_key],
                    'history': x[prompt_key]
            })},
            num_proc=os.getenv('num_proc'),
        )
        dataset = dataset.map(
            lambda x: {
                key: calculate_score(
                    prompt=x[f'{key}_prompt'],
                )
            },
            batched=True,
            batch_size=32, # more cannot increase the speed
        )

    return dataset