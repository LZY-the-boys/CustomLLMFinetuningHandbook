import numpy as np
import torch
import random
import pandas as pd
import os
from datasets import Dataset
import sys
import json
from tabulate import tabulate
import yaml
import types

def fix_seed(seed: int = 0):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_markdown(data: pd.DataFrame, path=None):
    markdown_table = tabulate(data, headers='keys', tablefmt='pipe')
    print(markdown_table)
    if path is not None:
        print(markdown_table, file=open(path,'w'))

def from_yaml(path,):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.load(file, yaml.SafeLoader)
    return types.SimpleNamespace(**data)

def to_jsonl(data, path, mode='w'):
    if not isinstance(data, list):
        data = [data]
    with open(path, mode) as f:
        for line in data:
            f.write(json.dumps(line,ensure_ascii=False)+'\n')

def from_jsonc(path):
    # support for json with comment 
    import jstyleson
    return jstyleson.load(open(path))

def from_json(path):
    return json.load(open(path))

def from_jsonl(path):
    return [json.loads(line) for line in open(path, 'r',encoding='utf8') ]

def to_json(data, path, mode='w'):
    if mode == 'a' and os.path.exists(path):
        old_data = from_json(path)
        data = old_data + data
    json.dump(data, open(path, 'w', encoding='utf8'), ensure_ascii=False)

# next(iter(data.items()))[1].keys()
def to_excel(data, path, index=None, columns=None, mode='w'):

    if columns is None:
        # text_df(index, 'b')
        # NOTE : { 'a':{'x''y'},'b':{'x''y'}} => rows: x,y columns: a,b
        df = pd.DataFrame(data,index=index).T
        if mode == 'a':
            if os.path.exists(path):
                previous = pd.read_excel(path,index_col=0)
                df = pd.concat([previous,df])
                df.to_excel(path,index=True)
                return
        df.to_excel(path,index=True)
    # given column
    elif index is None:
        df = pd.DataFrame(data,columns = columns)

    df.to_excel(path,index=False)

def from_excel(path):
    df = pd.read_excel(path).to_dict('records')
    return df