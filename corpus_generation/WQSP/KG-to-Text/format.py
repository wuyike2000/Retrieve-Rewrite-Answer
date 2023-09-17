import random
import json
import os

os.makedirs('../../../finetune-llama/WQSP/train',exist_ok=True)

data=[]
with open('data/train.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        temp=dict()
        line=line.strip().split('\t')
        temp["instruction"]=line[2]
        temp["input"]=''
        temp["output"]=line[3]
        data.append(temp)

split=int(len(data)*0.9)
random.shuffle(data)
train=data[:split]
dev=data[split:]
print(len(train))
print(len(dev))

json.dump(train,open('../../../finetune-llama/WQSP/train/train.json','w',encoding='utf-8'),ensure_ascii=False, indent=2)
json.dump(dev,open('../../../finetune-llama/WQSP/dev.json','w',encoding='utf-8'),ensure_ascii=False, indent=2)