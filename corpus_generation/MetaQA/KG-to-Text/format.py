import json
import os

data=[]
with open('./data/train.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        if len(line)!=4:
            continue
        temp=dict()
        temp["instruction"]=line[2]
        temp["input"]=''
        temp["output"]=line[3]
        data.append(temp)

print(len(data))
train=int(len(data)*0.9)

os.makedirs('../../../finetune-llama/MetaQA/train',exist_ok=True)
json.dump(data[:train],open('../../../finetune-llama/MetaQA/train/train.json','w',encoding='utf-8'))
json.dump(data[train:],open('../../../finetune-llama/MetaQA/dev.json','w',encoding='utf-8'))

os.makedirs('../../../finetune-t5/data',exist_ok=True)
json.dump(data[:train],open('../../../finetune-t5/data/train.json','w',encoding='utf-8'))
json.dump(data[train:],open('../../../finetune-t5/data/dev.json','w',encoding='utf-8'))
        