import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import MvpTokenizer, MvpForConditionalGeneration

template='Describe the following data: {graph}'

model="../../pretrain/mvp-data-to-text"
data='../retrieve/result/test.txt'  
output='output/mvp.txt'

tokenizer = MvpTokenizer.from_pretrained(model)
model = MvpForConditionalGeneration.from_pretrained(model).cuda()

f1=open(output,'w',encoding='utf-8')
with open(data,'r',encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        line=line.strip().split('\t')
        ques=line[0]
        ans=line[2]
        graph=line[3].split('|')
        graph1=line[4]
        text=[]
        inputs=[]
        for i in graph:
            i=i.replace(',',' |').replace(') (',' [SEP] ').replace('(','').replace(')','')
            prompt=template.format(graph=i)
            inputs.append(prompt)
        prompt=tokenizer(inputs, return_tensors="pt", padding=True).to('cuda')
        text = model.generate(**prompt,max_length=512)
        text=tokenizer.batch_decode(text,skip_special_tokens=True)
        f1.write('{}\t{}\t{}\t{}\n'.format(ques,ans,'|'.join(text),graph1))
f1.close()
    
