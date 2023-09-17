import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GenerationConfig
import argparse
import logging
from tqdm import tqdm
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

base_model = "../../pretrain/T0"  
data='../rewrite/output/llama-2-13b-chat.txt'
output='result/llama-2-13b-chat-T0.txt'

template='Below are the facts that might be relevant to answer the question: {fact}\nQuestion: {question}\nAnswer:'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

generation_config = GenerationConfig(temperature=0.001,max_new_tokens=400)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model,device_map='auto')
    
    correct=0
    all=0
    with torch.no_grad():
        print("Start inference.")    
        f1=open(output,'w',encoding='utf-8')
        with open(data,'r',encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                line=line.strip().split('\t')
                ques=line[0]
                if len(line)==2:
                    text=[]
                else:
                    text=line[2].split('|')
                fact=''
                for index,i in enumerate(text):
                    fact=fact+i+' '
                fact=fact[:-1]
                ans=line[1].split('|')
                prompt=template.format(question=ques,fact=fact)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs,generation_config = generation_config)[0]
                response=tokenizer.decode(outputs, skip_special_tokens=True)
                result='incorrect'
                for i in ans:
                    if i.lower() in response.lower():
                        result='correct'
                        correct+=1
                        break
                f1.write('{}\t{}\t{}\t{}\t{}\n'.format(ques,fact.replace('\n',' ').replace('\t',' '),response.replace('\n',' ').replace('\t',' '),'|'.join(ans),result))
                all+=1
                logging.info('Current Accuracy: {}'.format(correct/all))
        f1.close()
    logging.info('Accuracy: {}'.format(correct/all))
