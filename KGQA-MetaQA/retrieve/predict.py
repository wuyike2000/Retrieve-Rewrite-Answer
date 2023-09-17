import pandas as pd
import random
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import numpy as np
import os
import time
import psutil
import torch.nn.functional as F
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TYPE='bert'
MODEL="bert-base-uncased"
NUM=3

redict=dict()
id2rel=dict()
with open('relation.txt','r',encoding='utf-8') as f:
    for index,i in enumerate(f.readlines()):
        redict[i.strip()]=index
        id2rel[index]=i.strip()
print(redict)
print(id2rel)

# Configure the model
model_args = ClassificationArgs()
model_args.n_gpu=1
model_args.eval_batch_size=16
model_args.use_multiprocessing = False
model_args.dataloader_num_workers = 0
model_args.process_count = 1
model_args.use_multiprocessing_for_evaluation = False

# Create a MultiLabelClassificationModel
model = ClassificationModel(TYPE, 'bert-base-uncased_output/best_model', num_labels=len(redict),use_cuda=True,args=model_args)

# first hop relation prediction
inputs1=[]
hops1=[]
goldchain=[]
with open('../../corpus_generation/MetaQA/processed_data/test.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        ques=line[0]
        inputs1.append(ques)
        goldchain.append(line[2])
output1=model.predict(inputs1)[1]
output1=torch.from_numpy(output1)
output1=F.log_softmax(output1,dim=1)
output1=output1.numpy()
correct=0
for i in output1:
    score1=np.flipud(np.sort(i))[:NUM]
    order1=np.flipud(np.argsort(i))[:NUM]
    temp=dict()
    for i,j in zip(score1,order1):
        temp[j]=i
    hops1.append(temp)

# second hop relation prediction
inputs2=[]
# collect the first hop relation
midre=[]
# inter is a list of dict collect first hop relation and its score
for ques,inter in zip(inputs1,hops1):
    for mid in inter.items():
        inputs2.append(ques+'|'+id2rel[mid[0]])
        midre.append([id2rel[mid[0]],mid[1]])
output2=model.predict(inputs2)[1]
output2=torch.from_numpy(output2)
output2=F.log_softmax(output2,dim=1)
output2=output2.numpy()
correct=0
os.makedirs('./rechain',exist_ok=True)
with open('rechain/test.txt','w',encoding='utf-8') as f:
    for index in range(0,len(output2),NUM):
        rechain=[]
        text=''
        for i,j in zip(output2[index:index+NUM],midre[index:index+NUM]):
            score2=np.flipud(np.sort(i))[:NUM]
            order2=np.flipud(np.argsort(i))[:NUM]
            for x,y in zip(score2,order2):
                rechain.append([j[1]+x,j[0]+'|'+id2rel[y]])
        rechain.sort(reverse=True)
        for i in rechain:
            text=text+i[1]+'\t'
        text=text[:-1]+'\n'
        f.write(text)
        if rechain[0][1]==goldchain[int(index/NUM)]:
            correct+=1
print('Accuracy: {}'.format(correct/len(inputs1)))