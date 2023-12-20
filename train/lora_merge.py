import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig
import torch
from typing import List, Dict, Optional
import json

def main(
    *, 
    model_path: str = 'meta-llama/Llama-2-70b-hf',
    peft_paths: list[str] = None,
    hub: str = None,
    local: str = None,
):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     llm_int8_has_fp16_weight=True,
        # )
    )
    for peft in peft_paths:
        model = PeftModel.from_pretrained(
            model, 
            peft, 
        )
        model = model.merge_and_unload(progressbar=True,safe_merge=True)

        name = peft.split('/')[-1]
        if hub:
            model.push_to_hub(f'{hub}/{name}-merged',private=True)
            tokenizer.push_to_hub(f'{hub}/{name}-merged',private=True)
        elif local:
            model.save_pretrained(f'{local}/{name}-merged')
            tokenizer.save_pretrained(f'{local}/{name}-merged')            

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