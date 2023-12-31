import argparse
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

base_model = "../../pretrain/llama-2-13b-chat-hf"  
data='../rewrite/output/llama-2-7b-chat.txt'
output='result/triple-llama-2-13b-chat.txt'

template="""
Below are the facts that might be relevant to answer the question: {fact}
Question: {question}
Answer:
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
                fact=line[-1]
                ans=line[1].split('|')
                prompt=template.format(question=ques,fact=fact)
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
                response = output.split('Answer:')[-1].replace('\n',' ')
                result='incorrect'
                for i in ans:
                    if i.lower() in response.lower():
                        result='correct'
                        correct+=1
                        break
                f1.write('{}\t{}\t{}\t{}\t{}\n'.format(ques,fact.replace('\n',''),response,'|'.join(ans),result))
                all+=1
                logging.info('Current Accuracy: {}'.format(correct/all))
        f1.close()
        logging.info('Accuracy: {}'.format(correct/all))
