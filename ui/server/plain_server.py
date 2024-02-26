import torch
import os
import transformers
from transformers import BitsAndBytesConfig
from peft import PeftModel

MODEL = 'Qwen/Qwen-14B'
LORA='/data/outs/qwen-v1221-dpo-lora/checkpoint-200'
DTYPE=torch.bfloat16
MAXLEN = 8192
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    device_map={"": 0},
    torch_dtype=DTYPE,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,  
        llm_int8_has_fp16_weight=True,
    )
)
model = PeftModel.from_pretrained(
    model, 
    LORA, 
)
print(f'>>> loading {MODEL} and {LORA} finished')
tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL,
    add_special_tokens=True,
    trust_remote_code=True,
    padding='left',
)
tokenizer.pad_token_id=0
while True:
    prompt=input('>> User: ')
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_length = encoded["input_ids"][0].size(0)
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=1024,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
        )
    output_ids=outputs.sequences[0][prompt_length:]
    print('>> assistant: ', output_ids)
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    print('>> assistant: ', output)

# <|im_start|>system\n<|im_end|>\n<|im_start|>user\nWho is Larry Page?<|im_end|>\n<|im_start|>assistant\n
# Larry Page is an American computer scientist and Internet entrepreneur, and is one of the founders of Google. Born in Michigan in 1973, he co-founded Google with Sergey Brin while they were Ph.D. students at Stanford University in 1998. Page served as Google's CEO from 2001 to 2011, during which time he oversaw the company's rise to become one of the world's largest and most influential technology companies. In 2015, he co-founded Alphabet, a holding company that manages and invests in various Google-related ventures, with Brin. Page stepped down as CEO of Alphabet in 2019, but remains a member of the company's board of directors and is heavily involved in the company's technical development. Page is known for his innovative ideas and a strong focus on research and development, and has been recognized for his contributions to the field of computer science and the development of search technology. In addition to his work with Google and Alphabet, Page is also involved in various philanthropic and scientific initiatives.เหรียญ
# เหรียญ = <|eos_token_id|>


# (Pdb) tokenizer.decode([151643])
# '<|endoftext|>'
# (Pdb) tokenizer.decode([13, 151643])
# '.<|endoftext|>'
# tokenizer.decode([13586,   5085,    448,   3240,    258,     13,   9449,   5755,
#             594,  11438,     11,   5085,   6116,    825,    315,    279,   1429,
#         6849,   5110,    304,    279,   1879,     11,    448,    264,   3081,
#         6722,   2022,    315,    916,    400,     16,  31510,     13,   5755,
#          702,   1083,   1012,   6398,    304,   5257,   1008,  65624,     11,
#         2670,    279,   4401,    315,   5085,    594,    656,  59711,   1803,
#         5440,    323,    279,   9688,    315,  62797,     11,    264,   9963,
#         2813,    429,  70845,   5085,    323,   1008,  40064,   5110,     13,
#         5755,  24326,   1495,    438,  12156,    315,  62797,    304,    220,
#           17,     15,     16,     20,     11,    714,   8458,    458,   4541,
#        29593,    323,  31753,    311,    279,   2813,     13, 140377,    271,
#          151643])