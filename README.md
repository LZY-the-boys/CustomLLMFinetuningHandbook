# Custom LLM Finetuning Handbook

Welcome to the Custom LLM Finetuning Handbook, an example to fine-tuning a Language Learning Model (LLM) from data preparation to deployment. 
This handbook covers various aspects including data preparation, supervised fine-tuning (SFT), direct preference optimization (DPO), evaluation on common benchmark, and deployment using vLLM with an intuitive user interface.

## Data Preparation

Data preparation is the most crucial part. Here are the steps involved:

- Instruct Mining: This involves extracting and refining instructions from various sources to create a structured dataset for training.
- OpenAI API: Utilize the OpenAI API for additional data collection and preprocessing tasks.
- Hust Clean: A process to clean and standardize custom domain data

## Training the Language Model

### SFT

For SFT, you can use either LoRA (Low-Rank Adaptation) or full model fine-tuning. The primary repository for this process is https://github.com/LZY-the-boys/axolotl 
clone it and customize the yaml configuration file. 

Multi-modal Training you can visit this: https://github.com/OpenAccess-AI-Collective/axolotl/tree/llava-train

Dependency:  
-  transformers 4.36
-  peft 0.6.0
-  trl 0.7.2
-  bitsandbytes 0.41.2

### DPO

DPO is a method used to optimize the model's performance based on human preferences. The repositories involved are: 

- https://github.com/LZY-the-boys/dpo (for text dpo with Qwen)
- https://github.com/LZY-the-boys/qwen-vl-dpo (for multi-modal dpo with [Qwen-VL](https://github.com/QwenLM/Qwen-VL) )


## Evaluation

Leaderboard:
- lm-eval-harness https://github.com/LZY-the-boys/lm-evaluation-harness-fast (dev)
- helm https://github.com/LZY-the-boys/HELM-Extended-Local (dev)
- human-eval / mbpp
- ceval
- AlpacaEval / MTbench

## eval vision model 

Leaderboard: TODO

## deploy UI

The final step involves deploying your fine-tuned LLM using vLLM, ensuring it is accessible through a user-friendly interface. This section will guide you through the deployment process, ensuring your model is ready for use in real-world applications.

vllm deploy: https://github.com/LZY-the-boys/vllm
- support gptq and awq
- unknown exllama2 https://github.com/turboderp/exllamav2
- don't support bitsandbytes currently

UI:  
- https://github.com/imoneoi/openchat-ui
- https://github.com/huggingface/chat-ui
