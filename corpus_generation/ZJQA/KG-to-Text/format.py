import json
import os

os.makedirs('../../../finetune-llama/ZJQA/train',exist_ok=True)

data=[]
with open('data/train.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        temp=dict()
        temp["instruction"]=line[1]
        temp["input"]=''
        temp["output"]=line[2]
        data.append(temp)

json.dump(data,open('../../../finetune-llama/ZJQA/train/train.json','w',encoding='utf-8'),ensure_ascii=False, indent=2)

data=[]
with open('data/valid.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        temp=dict()
        temp["instruction"]=line[1]
        temp["input"]=''
        temp["output"]=line[2]
        data.append(temp)

json.dump(data,open('../../../finetune-llama/ZJQA/dev.json','w',encoding='utf-8'),ensure_ascii=False, indent=2)
        