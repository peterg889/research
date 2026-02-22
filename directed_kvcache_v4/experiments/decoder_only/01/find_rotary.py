import torch, os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-3-12b-it', device_map='auto', torch_dtype=torch.bfloat16,
    token=os.environ.get('HF_TOKEN'))

for name, mod in model.named_modules():
    if 'rotar' in name.lower() or 'rope' in name.lower():
        print(f'{name}: {type(mod).__name__}')
        for bname, buf in mod.named_buffers():
            if 'inv_freq' in bname:
                print(f'  buffer: {bname} shape={buf.shape} dtype={buf.dtype}')
        for aname in dir(mod):
            if 'attention_scaling' in aname:
                val = getattr(mod, aname)
                if isinstance(val, (int, float)):
                    print(f'  attr: {aname} = {val}')
