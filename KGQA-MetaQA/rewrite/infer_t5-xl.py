import argparse
import json, os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,GenerationConfig
from peft import PeftModel

base_model = "../../pretrain/flan-t5-xl"
lora_model='../../finetune-t5/flan-t5-xl/best_model'
data='../retrieve/result/test.txt'  
output='output/flan-t5-xl.txt'

template='Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: {graph}. The sentence is:'

generation_config = GenerationConfig(
    temperature=0.001,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
)

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    
    tokenizer_path = base_model

    tokenizer = T5Tokenizer.from_pretrained(base_model, legacy=True)

    base_model = T5ForConditionalGeneration.from_pretrained(
        base_model,
        device_map='auto',
        )

    model = PeftModel.from_pretrained(base_model, lora_model,device_map='auto',)

    with torch.no_grad():
        f1=open(output,'w',encoding='utf-8')
        with open(data,'r',encoding='utf-8') as f:
            for index,line in tqdm(enumerate(f.readlines())):
                line=line.strip().split('\t')
                ques=line[0]
                ans=line[2]
                graph=line[3].split('|')
                graph1=line[4]
                text=[]
                for i in graph:
                    prompt=template.format(graph=i)
                    inputs = tokenizer(prompt,return_tensors="pt")
                    generation_output = model.generate(
                        input_ids = inputs["input_ids"].to(device),
                        attention_mask = inputs['attention_mask'].to(device),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        generation_config = generation_config
                    )
                    s = generation_output[0]
                    output = tokenizer.decode(s,skip_special_tokens=True)
                    response = output.split(': ')[-1]
                    text.append(response)
                f1.write('{}\t{}\t{}\t{}\n'.format(ques,ans,'|'.join(text),graph1))
            f1.close()
