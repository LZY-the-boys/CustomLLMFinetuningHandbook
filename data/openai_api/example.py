import openai
from tqdm import tqdm
import time
key = None,

import openai

if key is None:
    raise Exception('You have to input your openai API key!') 
openai.api_key = key

total_tokens = 0
instruct = 'who are you ?'
responses = []
while True:
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4-1106-preview", messages=[{"role": "user", "content": instruct}]
        )
        # 查看 API 调用使用了多少 token:
        tokens = completion["usage"]["total_tokens"]
        total_tokens += tokens
        response = completion["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        if 'You exceeded your current quota' in e._message:
            exit()
        if 'That model is currently overloaded with other requests.' in e._message:
            time.sleep(200)
    time.sleep(1)

    responses.append(response)
