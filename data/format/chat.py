import datasets
import utils
import tqdm
import os
import random
import pandas as pd
import multiprocess, os

# NOTICE: 先删除dataset的cache

continue_words = ["继续", "接着写", "接着说", "Continue", "continue"]

def merge_input(x):
    return (
        x['instruction'] + random.choice([' ',':']) + x['input'] 
        if len(x['input']) else x['instruction']
    )

def contains_chatgpt_words(text):
    replace_keywords = {
        'ChatGPT': 'CCIIP-GPT',
        'OpenAI': '华中科技大学CCIIP-LAB',
        'GPT3.5': 'CCIIP-GPT',
        'GPT4':'CCIIP-GPT',
    }
    for word in replace_keywords:
        if word.lower() in text.lower():
            return True
    return False

# dedup
def contains_unwanted_words(text):
    # adaptered from: https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered/blob/main/wizardlm_clean.py
    unwanted_words = [
        # "语言模型", "抱歉", "我无法", "没有能力", "Sorry", "sorry", "apologize", "language model",
		"text-based AI language model",
		"domestic violence",
		"please refrain",
		"derogatory",
		"inappropriate",
		"offensive",
		"racism",
		"racist",
		"racial",
		"discriminate",
		"discriminatory",
		"discrimination",
		"sexist",
		"sexism",
		"unacceptable",
		"inclusive workplace",
		"lgbt",
		"morals",
		"ethics",
		"ethical",
		"legality",
		"illegal",
		"illegality",
		"hateful",
		"harmful",
		"it is never okay",
		"It is important to",
		"It's important to",
		"real-world consequences",
		"hate speech",
		"glorify",
		"not be appropriate",
		"supremacist",
		"extremist",
		"responsible AI",
		"AI principles",
		"AI assistant",
		"an AI language",
		"ableist",
		"hurtful",
		"gender stereotype",
		"gender inequality",
		"underrepresentation",
		"safe spaces",
		"gender-based",
		"inclusivity",
		"feminist",
		"feminism",
		"transgender",
		"empowerment",
		"communist",
		"capitalism",
		"stereotypes",
		"biases",
		"bias",
		"Microaggression",
		"prioritize human safety",
		"as a language model",
		"as an AI language model",
		"As a large language model",
		"As an AI",
		"ethical principles",
		"consensual",
		"it is not appropriate",
		"it's not appropriate",
		"I cannot fulfill your request",
		"harmful to human beings",
		"ethical guidelines",
		"my guidelines",
		"prioritize user safety",
		"adhere to ethical guidelines",
		"harmful consequences",
		"potentially harmful",
		"dangerous activities",
		"promote safety",
		"well-being of all users",
		"responsible information sharing",
		"jeopardize the safety",
		"illegal actions or intentions",
		"undermine the stability",
		"promote the well-being",
		"illegal activities or actions",
		"adherence to the law",
		"potentially be harmful",
		"illegal substances or activities",
		"committed to promoting",
		"safe information",
		"lawful information",
		"cannot provide guidance",
		"cannot provide information",
		"unable to offer assistance",
		"cannot engage in discussions",
		"programming prohibits",
		"follow ethical guidelines",
		"ensure the safety",
		"involves an illegal subject",
		"prioritize safety",
		"illegal subject",
		"prioritize user well-being",
		"cannot support or promote",
		"activities that could harm",
		"pose a risk to others",
		"against my programming",
		"activities that could undermine",
		"potentially dangerous",
		"not within the scope",
		"designed to prioritize safety",
		"not able to provide",
		"maintain user safety",
		"adhere to safety guidelines",
		"dangerous or harmful",
		"cannot provide any information",
		"focus on promoting safety",
		"an AI language model you don't have",
		"As an AI language model, I cannot",
		"As an AI language model, I do not",
		"As an AI language model, I am not able",
		"As an AI language model, I don't have personal",
		"I am an AI language model and do not",
		"However, it is important to use any code or information provided responsibly and within legal and ethical boundaries.",
		"As an AI language model, I don't have",
		"As an AI language model, I am only able",
		"AI language model and I do not",
		"As an AI language model, I cannot modify",
		"As an AI language model, I do not",
		"I know as an AI language model you don't have",
		"as an AI language model, you cannot",
		"I'm sorry, but as an AI language model",
		"As an AI language model, I don't have",
		"Unfortunately, I cannot provide",
		"I'm sorry, I cannot",
		"I'm sorry, I cannot generate",
		"AI cannot create or program",
		"I'm afraid I cannot create",
		"you cannot create an",
		"it operates ethically and is",
		"had an ethical system",
		"Ensuring the ethical",
		"and ethical sourcing",
		"are from ethical",
		"legal and ethical",
		"engage in unethical",
		"unethical or aggressive",
		"unethical business",
		"como modelo de lenguaje AI",
		"Lo siento, como modelo de lenguaje",
		"no puedo proporcionar",
		"pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
		"Lo siento, pero no puedo",
		"Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
		"Lo siento, como modelo de lenguaje, no tengo",
		"Lo siento, debe haber habido una confusi\u00f3n",
		"Lo siento, como modelo de lenguaje, no puedo realizar",
		"Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
		"Lamento no poder proporcionarte el c\u00f3digo",
		"Desculpe-me, mas a linguagem vulgar e ofensiva",
		"apropriada em nenhum contexto",
		"Como modelo de linguagem",
		"Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
		"I cannot assist",
		"prioritize ethical",
		"respectful",
		"morally",
		"I'm sorry,",
		"I'm an",
		"I am an",
		"I'm an AI" ,
		"I am an AI",
		"my purpose",
		"filter_bad_language",
		"filter\_bad\_language",
		"entertainment purposes",
		"purely hypothetical",
		"not a human",
		"I am an AI",
		"cannot provide",
		"can't provide",
		"won't provide",
		"not provide",
		"worth noting",
		"cause harm",
		"a language model",
		"keep in mind",
		"unethical",
		"bad language",
		"the words ****",
		"bad_language",
		"certainly not",
		"complying",
		"comply",
		"I cannot",
		"my main goal",
		"As a machine",
		"I don't have the ability",
		"I am here to assist",
		"my purpose is to ",
		"my knowledge cutoff",
		"my knowledge cut off",
		"September 2021",
		"regulations",
		"not be suitable",
		"I apologize, but",
		"It is not possible",
		"controversial",
		"my programming",
		"ethically",
		"it is important to",
		"Please note",
		"sensitive topic",
		"not acceptable",
		"It is important for",
		"divisive",
		"not appropriate",
		"our values",
		"f\*cking",
		"F\*ck",
		"sh\*t",
		"diversity and",
		"diversity and inclusion",
		"values diversity",
		"social responsibility",
		"environmental, social, and governance",
		" ESG ",
		"against women",
		"problematic history",
		"diversity",
		"*This chat conversation is shared from",
		"*This conversation is shared from"
	]
    for word in unwanted_words:
        if word.lower() in text.lower():
            return True
    return False

# data = utils.from_json('/data/dataset/openchat_sharegpt4_dataset/sharegpt_gpt4.json')
def load_sharegpt4():
    data = datasets.load_dataset('json', data_files='/data/dataset/openchat_sharegpt4_dataset/sharegpt_gpt4.json')['train']
    data = data.filter(
        lambda x: not any(
            contains_unwanted_words(resp['value']) for resp in x['items'] if resp['from'] == 'gpt'
        ),
        num_proc=os.cpu_count()//2,
    )
    chatgpt_data = data.filter(
        lambda x: any(
            contains_chatgpt_words(resp['value']) for resp in x['items'] if resp['from'] == 'gpt'
        ),
        num_proc=os.cpu_count()//2,
    )
    data = data.filter(
        lambda x: not any(
            contains_chatgpt_words(resp['value']) for resp in x['items'] if resp['from'] == 'gpt'
        ),
        num_proc=os.cpu_count()//2,
    )
    data = data.map(
        lambda x: {
            'conversations': x['items'],
            'source': 'openchat/sharegpt_gpt4'
        },
        remove_columns=data.features,
        num_proc=os.cpu_count()//2,
    )
    data.to_json('/data/dataset/cciip-gpt/sharegpt4.jsonl')
    print(data)

    chatgpt_data.to_json('/data/dataset/chatgpt_words_sharegpt.jsonl')
    print(chatgpt_data)
    return data

def load_slimorca():
    data = datasets.load_dataset('Open-Orca/SlimOrca-Dedup')['train']
    data = data.filter(
        lambda x: not any(
            contains_unwanted_words(resp['value']) for resp in x['conversations'] if resp['from'] == 'gpt'
        ),
        num_proc=os.cpu_count()//2,
    )
    chatgpt_data = data.filter(
        lambda x: any(
            contains_chatgpt_words(resp['value']) for resp in x['conversations'] if resp['from'] == 'gpt'
        ),
        num_proc=os.cpu_count()//2,
    )
    data = data.filter(
        lambda x: not any(
            contains_chatgpt_words(resp['value']) for resp in x['conversations'] if resp['from'] == 'gpt'
        ),
        num_proc=os.cpu_count()//2,
    )
    data = data.map(
        lambda x: {
            'conversations': x['conversations'],
            'source': 'Open-Orca/SlimOrca-Dedup'
        },
        remove_columns=data.features,
        num_proc=os.cpu_count()//2,
    )
    data.to_json('/data/dataset/slim_orca_dedup.jsonl')
    print(data)

    chatgpt_data.to_json('/data/dataset/chatgpt_words_slimorca.jsonl')
    print(chatgpt_data)

    return data

def load_ultrachat():

    def process(xs):
        y = []
        for x in xs:
            y.append({
                'from': 'gpt' if x['role']=='assistant' else 'human',
                'value': x['content']
            })
        return y

    data = datasets.load_dataset('HuggingFaceH4/ultrachat_200k')
    data = datasets.concatenate_datasets([data['train_sft'],data['train_gen']])
    data = data.map(
        lambda x: {
            'conversations': process(x['messages']),
        },
        remove_columns=data.features,
        num_proc=os.cpu_count()//2,
    )
    data = data.filter(
        lambda x: len(x['conversations']) >= 2,
    )
    data.to_json('/data/dataset/ultrachat_200k.json')
    return data

# TODO: sharegpt-zh
def load_sharegpt_zh():
    # common_zh_70k.jsonl
    # computer_zh_26k.jsonl
    # computer_cn_26k_continue.jsonl “继续”字眼的过滤版本
    # unknow_zh_38k.jsonl
    # unknow_zh_38k_continue.jsonl
    data1 = utils.from_jsonl('/data/dataset/ShareGPT-Chinese-English-90k/shareGPT/computer_zh_26k.jsonl')
    data2 = utils.from_jsonl('/data/dataset/ShareGPT-Chinese-English-90k/sharegpt_jsonl/unknow_zh_38k.jsonl')
    data3 = utils.from_jsonl('/data/dataset/ShareGPT-Chinese-English-90k/sharegpt_jsonl/common_zh_70k.jsonl')
    
    data = data1+data2+data3

    import multiprocess, os

    def map_fn(item):
        y = []
        for i in item['conversation']:
            if any(key in i['human'] for key in delete_keywords):
                return None
            if any(key in i['assistant'] for key in delete_keywords):
                return None           
            y.append({
                'from': 'human',
                'value': i['human'],
            })
            y.append({
                'from': 'gpt',
                'value': i['assistant'],
            })
        return {'conversations': y}

    process_num = os.cpu_count() // 2
    ans = []
    with multiprocess.Pool() as pool:
        for ret in (tqdm.tqdm(
            pool.imap(map_fn, data), 
            total=len(data), desc=f'MAP({process_num})'
        )):
            if ret:
                ans.append(ret)
    utils.to_jsonl(ans,'/data/dataset/sharegpt_zh.json')
    return ans

def load_gsm8k_zh():
    # common_zh_70k.jsonl
    # computer_zh_26k.jsonl
    # computer_cn_26k_continue.jsonl “继续”字眼的过滤版本
    # unknow_zh_38k.jsonl
    # unknow_zh_38k_continue.jsonl
    raw_datasets = datasets.load_dataset('gsm8k', name='main')["train"]
    data = raw_datasets.map(
        lambda x: {
            "conversation": [{'human': x['question'], "assistant": x['answer']}]
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )

    def map_fn(item):
        y = []
        for i in item['conversation']:
            if any(key in i['human'] for key in delete_keywords):
                return None
            if any(key in i['assistant'] for key in delete_keywords):
                return None           
            y.append({
                'from': 'human',
                'value': i['human'],
            })
            y.append({
                'from': 'gpt',
                'value': i['assistant'],
            })
        return {'conversations': y}

    process_num = os.cpu_count() // 2
    ans = []
    with multiprocess.Pool() as pool:
        for ret in (tqdm.tqdm(
            pool.imap(map_fn, data), 
            total=len(data), desc=f'MAP({process_num})'
        )):
            if ret:
                ans.append(ret)
    utils.to_jsonl(ans,'/data/dataset/gsm8k_zh.json')
    return ans

def load_hust():
    data = utils.from_json('/home/LeiFeng/lzy/CCIIP-GPT/data/hust_clean/QA_all.json')
    y = []
    for d in data:
        if contains_unwanted_words(d['response']):
            continue
        y.append({
            "conversations": [{
                'from': 'human',
                'value': d['prompt'],
            },
            {
                'from': 'gpt',
                'value': d['response'],
            }],
            'source': 'hust-cciip-lab'
        })
    import pdb; pdb.set_trace()
    utils.to_jsonl(y, '/data/dataset/cciip-gpt/hust.jsonl')
    return data

def load_teknium():
    chatgpt_data = []
    gpt_teacher_data = datasets.load_dataset('teknium/GPTeacher-General-Instruct', split='train')
    gpt_teacher_data = gpt_teacher_data.filter(
        lambda x: not contains_unwanted_words(x['response']),
    )
    chatgpt_data.append(gpt_teacher_data.filter(
        lambda x: contains_chatgpt_words(x['response']),
    ))
    gpt_teacher_data = gpt_teacher_data.map(
        lambda x: {
            'conversations': [
                {
                    'from': 'human',
                    'value': merge_input(x),
                },
                {
                    'from': 'gpt',
                    'value': x['response'],
                },
            ],
            'source': 'teknium/GPTeacher-General-Instruct'
        },
        remove_columns=gpt_teacher_data.features,
        num_proc=os.cpu_count()//2,
    )
    gpt_teacher_data.to_json('/data/dataset/teknium_gpt4_teacher.jsonl')
    print(gpt_teacher_data)

    gpt_llm_data = datasets.load_dataset('teknium/GPT4-LLM-Cleaned', split='train')
    gpt_llm_data = gpt_llm_data.filter(
        lambda x: not contains_unwanted_words(x['output']),
        num_proc=os.cpu_count()//2,
    )
    chatgpt_data.append(gpt_llm_data.filter(
        lambda x: contains_chatgpt_words(x['output']),
        num_proc=os.cpu_count()//2,
    ))
    gpt_llm_data = gpt_llm_data.map(
        lambda x: {
            'conversations': [
                {
                    'from': 'human',
                    'value': merge_input(x),
                },
                {
                    'from': 'gpt',
                    'value': x['output'],
                },
            ],
            'source': 'teknium/GPT4-LLM-Cleaned'
        },
        remove_columns=gpt_llm_data.features,
        num_proc=os.cpu_count()//2,
    )  
    gpt_llm_data.to_json('/data/dataset/teknium_gpt4_llm.jsonl')
    print(gpt_llm_data)

    economics_data = datasets.load_dataset('teknium/dataforge-economics', split='train')
    economics_data = economics_data.map(
        lambda x: {
            'conversations': x['conversations'],
            'source': 'teknium/dataforge-economics',
        },
        remove_columns=economics_data.features,
        num_proc=os.cpu_count()//2,
    )
    economics_data.to_json('/data/dataset/teknium_dataforg_economics.jsonl')
    print(economics_data)

    data = datasets.concatenate_datasets([gpt_llm_data,gpt_teacher_data,economics_data])
    chatgpt_data = datasets.concatenate_datasets(chatgpt_data)

    return data

def load_h4_norobots():

    data = datasets.load_dataset('HuggingFaceH4/no_robots')
    data = datasets.concatenate_datasets([data['train_sft'],data['test_sft']])

    def process(xs):
        y = []
        for x in xs:
            if x['role'] == 'assistant': 
                y.append({
                    'from': 'gpt' ,
                    'value': x['content']
                })
            elif x['role'] == 'user':
                y.append({
                    'from': 'human',
                    'value': x['content']
                })
            elif x['role'] == 'system':
                y.append({
                    'from': 'system',
                    'value': x['content']
                })
            else:
                import pdb; pdb.set_trace()
        return y
    
    data = data.map(
        lambda x: {
            'conversations': process(x['messages']),
            'source': 'HuggingFaceH4/no_robots',
        },
        remove_columns=data.features,
        num_proc=1,
    )
    data.to_json('/data/dataset/h4_no_robots.jsonl')
    print(data)
    return data


def load_coig_cqia():

    # data = datasets.load_dataset('m-a-p/COIG-CQIA', split='train')
    # find /data/dataset/COIG-CQIA -type f -name '*.json' -exec cp {} /data/dataset/COIG-CQIA/all \;
    data = []
    for file in os.listdir('/data/dataset/COIG-CQIA/all'):
        data += utils.from_json(f'/data/dataset/COIG-CQIA/all/{file}')

    # data = pd.DataFrame.from_dict(data)
    # data = datasets.Dataset.from_pandas(pd.DataFrame.from_dict(data))
    
    # data = data.filter(
    #     lambda x: not contains_unwanted_words(x['output']),
    # )
    # data = data.map(
    #     lambda x: {
    #         'conversations': [
    #             {
    #                 'from': 'human',
    #                 'value': merge_input(x),
    #             },
    #             {
    #                 'from': 'gpt',
    #                 'value': x['output'],
    #             },
    #         ],
    #         'source': 'm-a-p/COIG-CQIA'
    #     },
    #     remove_columns=data.features,
    #     num_proc=os.cpu_count()//2,
    # )  
    # data = list(map(lambda x: {
    #         'conversations': [
    #             {
    #                 'from': 'human',
    #                 'value': merge_input(x),
    #             },
    #             {
    #                 'from': 'gpt',
    #                 'value': x['output'],
    #             },
    #         ],
    #         'source': 'm-a-p/COIG-CQIA'
    #     }, data
    # ))

    def map_fn(x):
        if contains_unwanted_words(x['output']):
            return None        
        item  = {   
            'conversations': [
                {
                    'from': 'human',
                    'value': merge_input(x),
                },
                {
                    'from': 'gpt',
                    'value': x['output'],
                },
            ],
            'source': 'm-a-p/COIG-CQIA'
        }
        return item

    process_num = os.cpu_count() // 2
    ans = []
    with multiprocess.Pool() as pool:
        for ret in (tqdm.tqdm(
            pool.imap(map_fn, data), 
            total=len(data), desc=f'MAP({process_num})'
        )):
            if ret:
                ans.append(ret)
    
    print(len(ans), ans[0])
    # data.to_json('/data/dataset/map_coig_cqia.jsonl')
    utils.to_jsonl(ans, '/data/dataset/map_coig_cqia.jsonl')
    return ans

if __name__ == "__main__":
    try:
        load_h4_norobots()
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
   