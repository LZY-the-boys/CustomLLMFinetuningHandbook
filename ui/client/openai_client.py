import openai
from logging_config import configure_logging
import logging
import utils
import time
import shortuuid
import os
import argparse
import tqdm
import os
import platform
import shutil
from copy import deepcopy

# Set OpenAI's API key and API base to use vLLM's API server.

openai.api_key="EMPTY"
openai.api_base = "http://localhost:8000/v1"
API_MAX_RETRY = 1
API_RETRY_SLEEP = 10
EVAL_DIR='/home/LeiFeng/lzy/CCIIP-GPT/eval'

configure_logging()
LOG = logging.getLogger(__name__)


# fschat: 
# def chat_compeletion_anthropic(model, conv, temperature, max_tokens):
# def chat_compeletion_palm(chat_state, model, conv, temperature, max_tokens):
def chat_compeletion_openai(
    model ="facebook/opt-125m", 
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
):
    for _ in range(API_MAX_RETRY):
        # try:
        response = openai.ChatCompletion.create(
            model=model,messages=messages
        )
        output = response["choices"][0]["message"]["content"]
        break
        # except openai.OpenAIError as e:
        #     print(type(e), e)
        #     time.sleep(API_RETRY_SLEEP)

    return output

# automatically convert to generator
def chat_compeletion_openai_stream(
    model ="facebook/opt-125m", 
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
):
    # try:
    response = openai.ChatCompletion.create(
        model=model,messages=messages,stream=True
    )
    
    output=''
    for r in response:
        delta = r["choices"][0]["delta"]
        if 'content' not in delta:
            continue
        output += delta['content']
        yield output

    return output

def alpaca_eval(
    model,
):
    import datasets
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    out_path = f'{EVAL_DIR}/data/alpaca_eval/{model.replace("/","_")}.jsonl'
    ans = []

    if os.path.exists(out_path):
        ans = utils.from_jsonl(out_path)
        runed = len(ans)
        eval_set = eval_set.select(range(runed,len(eval_set)))
        print(f'>>> skip {runed} instance')
    
    for example in tqdm.tqdm(eval_set):
        LOG.info(example["instruction"])
        example["output"] = chat_compeletion_openai(
            model,
            messages = [{"role": "user", "content":example["instruction"]}],
        )
        LOG.info(example["output"])
        ans.append(example)
    utils.to_jsonl(ans, out_path)

def mt_bench(
    model: str, 
    num_choices = 1, # "How many completion choices to generate."
):

    question_file = f"{EVAL_DIR}/data/mt_bench/question.jsonl"
    answer_file = f"{EVAL_DIR}/data/mt_bench/model_answer/{model.replace('/','_')}.jsonl"

    datas = utils.from_jsonl(question_file)
    ans = []
    for data in datas:
        choices = []
        for i in range(num_choices):
            messages,turns = [],[]
            for question in data['turns']:
                messages.append({
                    'role': 'user',
                    'content': question,
                })
                LOG.info(question)
                output = chat_compeletion_openai(
                    model, messages,
                )
                messages.append({
                    'role': 'assistant',
                    'content': output,
                })
                turns.append(output)
                LOG.info(output)
            choices.append({'index': i, 'turns': turns})
        ans.append({
            "question_id": data["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model,
            "choices": choices,
            "tstamp": time.time(),
        }) 
    utils.to_json(ans, answer_file)

def cli_demo(
    model,
):

    def _clear_screen():
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")

    def _get_input() -> str:
        while True:
            try:
                message = input('User> ').strip()
            except UnicodeDecodeError:
                print('[ERROR] Encoding error in input')
                continue
            except KeyboardInterrupt:
                exit(1)
            if message:
                return message
            print('[ERROR] Query is empty')

    def _print_history(history):
        terminal_width = shutil.get_terminal_size()[0]
        print(f'History ({len(history)})'.center(terminal_width, '='))
        for index, (query, response) in enumerate(history):
            print(f'User[{index}]: {query}')
            print(f'CCIIP-GPT[{index}]: {response}')
        print('=' * terminal_width)


    _WELCOME_MSG = (
        "Welcome to use CCIIP-GPT model, type text to start chat, type :h to show command help."
        "(欢迎使用 CCIIP-GPT 模型，输入内容即可进行对话，:h 显示命令帮助。)"
    )

    _HELP_MSG = (
        '''\
        Commands:
            :help / :h          Show this help message              显示帮助信息
            :exit / :quit / :q  Exit the demo                       退出Demo
            :clear-his / :clh   Clear history                       清除对话历史
            :history / :his     Show history                        显示对话历史
        '''
            # :clear / :cl        Clear screen                        清屏
            # :conf               Show current generation config      显示生成配置
            # :reset-conf         Reset generation config             重置生成配置
            # :conf <key>=<value> Change generation config            修改生成配置
    )

    history, response = [], ''
    _clear_screen()
    print(_WELCOME_MSG)

    while True:
        query = _get_input()

        # Process commands.
        if query.startswith(':'):
            command_words = query[1:].strip().split()
            if not command_words:
                command = ''
            else:
                command = command_words[0]

            if command in ['exit', 'quit', 'q']:
                break
            elif command in ['clear-history', 'clh']:
                print(f'[INFO] All {len(history)} history cleared')
                history.clear()
                continue
            elif command in ['help', 'h']:
                print(_HELP_MSG)
                continue
            elif command in ['history', 'his']:
                _print_history(history)
                continue
            elif command in ['seed']:
                if len(command_words) == 1:
                    print(f'[INFO] Current random seed: {seed}')
                    continue
                else:
                    new_seed_s = command_words[1]
                    try:
                        new_seed = int(new_seed_s)
                    except ValueError:
                        print(f'[WARNING] Fail to change random seed: {new_seed_s!r} is not a valid number')
                    else:
                        print(f'[INFO] Random seed changed to {new_seed}')
                        seed = new_seed
                    continue
            # elif command in ['conf']:
            #     if len(command_words) == 1:
            #         print(model.generation_config)
            #     else:
            #         for key_value_pairs_str in command_words[1:]:
            #             eq_idx = key_value_pairs_str.find('=')
            #             if eq_idx == -1:
            #                 print('[WARNING] format: <key>=<value>')
            #                 continue
            #             conf_key, conf_value_str = key_value_pairs_str[:eq_idx], key_value_pairs_str[eq_idx + 1:]
            #             try:
            #                 conf_value = eval(conf_value_str)
            #             except Exception as e:
            #                 print(e)
            #                 continue
            #             else:
            #                 print(f'[INFO] Change config: model.generation_config.{conf_key} = {conf_value}')
            #                 setattr(model.generation_config, conf_key, conf_value)
            #     continue
            # elif command in ['reset-conf']:
            #     print('[INFO] Reset generation config')
            #     model.generation_config = deepcopy(orig_gen_config)
            #     print(model.generation_config)
            #     continue
            else:
                # As normal query.
                pass

        # Run chat.
        try:
            messages = []
            for his in history:
                messages.append({'role': 'user', 'content': his[0]})
                messages.append({'role': 'assistant', 'content': his[1]})
            messages.append({'role': 'user', 'content': query})

            for response in chat_compeletion_openai_stream(
                model,
                messages,
            ):
                _clear_screen()
                print(f"\nUser: {query}")
                print(f"\nCCIIP-GPT: {response}")
            history.append((query, response))
        except KeyboardInterrupt:
            print('[WARNING] Generation interrupted')
            continue

def main(
    *, 
    model: str = 'Qwen/Qwen-14B',
    func: str = 'cli_demo',
):
    import sys
    getattr(sys.modules[__name__], func)(model)


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