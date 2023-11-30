import datasets
import utils
import os
import tqdm

delete_keywords = ["语言模型", "抱歉", "我无法", "没有能力", "Sorry", "sorry", "apologize", "language model"]
# question 里边的chatgpt不需要处理
replace_keywords = {
    'ChatGPT': 'CCIIP-GPT',
    'OpenAI': '华中科技大学CCIIP-LAB',
    'GPT3.5': 'CCIIP-GPT',
    'GPT4':'CCIIP-GPT',
}
continue_words = ["继续", "接着写", "接着说", "Continue", "continue"]
# dedup

# data = utils.from_json('/data/dataset/openchat_sharegpt4_dataset/sharegpt_gpt4.json')
def load_sharegpt4():
    data = datasets.load_dataset('json', data_files='/data/dataset/openchat_sharegpt4_dataset/sharegpt_gpt4.json')['train']
    data = data.map(
        lambda x: {
            'conversations': x['items'],
        },
        remove_columns=data.features,
        num_proc=os.cpu_count()//2,
    )
    data.to_json('/data/dataset/sharegpt4.json')

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


if __name__ == "__main__":
    try:
        load_sharegpt_zh()
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
   