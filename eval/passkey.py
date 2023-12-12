'''
copy from https://github.com/CStanKonrad/long_llama
'''
import argparse
import numpy as np
import torch
import random
from vllm import LLM, SamplingParams


def generate_prompt_landmark(context_len, seed):
    """Generates a text file and inserts an execute line at a random position."""
    rnd_state = np.random.get_state()
    np.random.seed(seed)
    n_garbage_prefix = np.random.randint(0, context_len)
    n_garbage_suffix = context_len - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= context_len
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = np.random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    np.random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)
def passkey_retrieval_test(model, context_len=60000, seed=555, num=100):
    random.seed(seed)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    prompts, answers = [], []
    for i in range(num):
        prompt, answer = generate_prompt_landmark(context_len, i)
        prompts.append(prompt)
        answers.append(answer)
    outputs = model.generate(prompts, sampling_params)

    correct_num = 0
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        if answers[i] in generated_text:
            correct_num += 1
    print(f"The context len is {context_len};The precision is {correct_num/num}")
    return correct_num/num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/model/qwen-sharegpt4-merged")
    args = parser.parse_args()

    gpus_num = torch.cuda.device_count()
    model = LLM(args.model_path, trust_remote_code=True, tensor_parallel_size=gpus_num)
    for i in range(1, 9):
        passkey_retrieval_test(model, i*1024)
