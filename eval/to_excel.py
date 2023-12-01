
import os,sys
import utils
import pandas as pd
from tabulate import tabulate

def mmlu(data):
    acc = 0
    for item in data['results'].keys():
        acc += data['results'][item]['acc']
    acc /= len(data['results'].keys())
    return acc

# def bigbench(data):
#     acc = 0
#     for item in data['results'].keys():
#         acc += data['results'][item]['acc']
#     acc /= len(data['results'].keys())
#     return acc

def winogrande(data):
    return data['results']['winogrande']['acc']

def arc_challenge(data):
    return data['results']['arc_challenge']['acc_norm']

def hellaswag(data):
    return data['results']['hellaswag']['acc_norm']

def truthfulqa_mc(data):
    return data['results']['truthfulqa_mc']['mc2']

def gsm8k(data):
    return data['results']['gsm8k']['acc']

def drop(data):
    return data['results']['drop']['f1'] 

def main(
    *, 
    path: str = '/home/LeiFeng/lzy/CCIIP-GPT/eval', 
    tasks: list[str]=['mmlu', 'drop', 'truthfulqa_mc', 'arc_challenge', 'hellaswag', 'winogrande'], # 'gsm8k'
):
    
    dicts = {}
    for file in os.listdir(path):
        try:
            if not any('json' in f for f in os.listdir(os.path.join(path, file))):
                continue
            dicts[file] = {}
            for task in tasks:
                _path = os.path.join(path,file, f'{task}.json')
                if os.path.exists(_path):
                    data = utils.from_json(_path)
                    dicts[file].update({
                        task: 100 * getattr(sys.modules[__name__], task)(data)
                    })
        except:
            pass
    df = pd.DataFrame.from_dict(dicts, orient='index')
    df.to_excel(os.path.join(path,'result.xlsx'))

    markdown_table = tabulate(df, headers='keys', tablefmt='pipe')
    print(markdown_table)
    print(markdown_table, file=open(os.path.join(path,'result.md'), 'w'))


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