import argparse
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import  PeftModel
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch
apply_attention_patch(use_memory_efficient_attention=True)
apply_ntk_scaling_patch("1.0")

base_model = "../../pretrain/llama-2-7b-chat-hf"
lora_model='../../finetune-llama/output-MetaQA/llama-2-7b-chat/best_model'
data='../retrieve/result/test.txt'  
output='output/llama-2-7b-chat.txt'

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

load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

tokenizer = LlamaTokenizer.from_pretrained(base_model, legacy=True)

base_model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )

model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenizer_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
if model_vocab_size!=tokenizer_vocab_size:
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenizer_vocab_size)
print("loading peft model")
model = PeftModel.from_pretrained(base_model, lora_model,torch_dtype=load_type,device_map='auto',)
model.cuda()
model.eval()

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
