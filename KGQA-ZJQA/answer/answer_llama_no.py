# -*- coding:gbk -*-
import argparse
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import  PeftModel
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch
apply_attention_patch(use_memory_efficient_attention=True)
apply_ntk_scaling_patch("1.0")
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_model = "../../pretrain/chinese-alpaca-2-7b-hf"  
data='../rewrite/output/chinese-alpaca-2-7b.txt'
output='result/no-chinese-alpaca-2-7b.txt'

template="""
以下是可能与问题相关的事实: {fact}
问题: {ques}
答案:
"""

generation_config = GenerationConfig(temperature=0.001,max_new_tokens=400)

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    tokenizer = LlamaTokenizer.from_pretrained(base_model, legacy=True)

    model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            )

    model.eval()
    
    correct=0
    all=0
    with torch.no_grad():
        print("Start inference.")
        f1=open(output,'w',encoding='utf-8')
        with open(data,'r',encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                line=line.strip().split('\t')
                ques=line[0]
                ans=line[1].split('|')
                prompt=ques
                inputs = tokenizer(prompt,return_tensors="pt")  #add_special_tokens=False ?
                generation_output = model.generate(
                        input_ids = inputs["input_ids"].to(device),
                        attention_mask = inputs['attention_mask'].to(device),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        generation_config = generation_config
                    )
                s = generation_output[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                response = output
                result='incorrect'
                for i in ans:
                    if i.lower() in response.lower():
                        result='correct'
                        correct+=1
                        break
                f1.write('{}\t{}\t{}\t{}\t{}\n'.format(ques,'no',response.replace('\n',' ').replace('\t',' '),'|'.join(ans),result))
                all+=1
                logging.info('Current Accuracy: {}'.format(correct/all))
        f1.close()
        logging.info('Accuracy: {}'.format(correct/all))
