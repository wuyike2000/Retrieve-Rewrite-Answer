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
import json
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
TYPE='bert'
MODEL="bert-base-uncased"
NUM=5

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
model_args.eval_batch_size=1
model_args.use_multiprocessing = False
model_args.dataloader_num_workers = 0
model_args.process_count = 1
model_args.use_multiprocessing_for_evaluation = False

# Create a MultiLabelClassificationModel
model = ClassificationModel(TYPE, MODEL+'_output/best_model', num_labels=len(redict),use_cuda=True,args=model_args)

hop=[]
with open('../hop_prediction/hop/test.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        hop.append(line.strip())

os.makedirs('./rechain',exist_ok=True)
correct=0
retrieve=0
f1=open('rechain/test.txt','w',encoding='utf-8')
data=json.load(open('../../../corpus_generation/WQSP/processed_data/final/test.json','r',encoding='utf-8'))
for index,sample in tqdm(enumerate(data)):
    ques=sample["question"]
    goldchain=sample["relation"]
    rechain=[]
    if hop[index]=='1':
        output1=model.predict([ques])[1][0]
        score1=np.flipud(np.sort(output1))[:NUM]
        order1=np.flipud(np.argsort(output1))[:NUM]
        for i in order1:
            rechain.append(id2rel[i])
        if rechain[0]==goldchain:
            correct+=1
        if goldchain in rechain:
            retrieve+=1
        text=''
        for i in rechain:
            text=text+i+'\t'
    else:
        # first hop relation prediction
        output1=model.predict([ques])[1][0]
        output1=torch.from_numpy(output1)
        output1=F.log_softmax(output1,dim=0)
        output1=output1.numpy()
        score1=np.flipud(np.sort(output1))[:NUM]
        order1=np.flipud(np.argsort(output1))[:NUM]
        hop1=dict()
        for i,j in zip(score1,order1):
            hop1[j]=i
        # second hop relation prediction
        for r in order1:
            output2=model.predict([ques+'|'+id2rel[r]])[1][0]
            output2=torch.from_numpy(output2)
            output2=F.log_softmax(output2,dim=0)
            output2=output2.numpy()
            score2=np.flipud(np.sort(output2))[:NUM]
            order2=np.flipud(np.argsort(output2))[:NUM]
            for i,j in zip(score2,order2):
                rechain.append([hop1[r]+i,id2rel[r]+'|'+id2rel[j]])
        rechain.sort(reverse=True)
        if rechain[0][1]==goldchain:
            correct+=1
        chainset=set()
        for i in rechain:
            chainset.add(i[1])
        if goldchain in chainset:
            retrieve+=1
        text=''
        for i in rechain:
            text=text+i[1]+'\t'
    f1.write(text[:-1])
    f1.write('\n')
f1.close()
print('Accuracy:',correct/len(data))
print('Retrieve:',retrieve/len(data))