import pandas as pd
import random
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import numpy as np
import os
import time
import psutil
import torch.nn.functional as F
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
TYPE='bert'
MODEL="bert-base-uncased"

os.makedirs('./hop',exist_ok=True)

data=json.load(open('../../../corpus_generation/WQSP/processed_data/final/train.json','r',encoding='utf-8'))
random.shuffle(data)
split=int(len(data)*0.9)
train=data[:split]
valid=data[split:]
test=json.load(open('../../../corpus_generation/WQSP/processed_data/final/test.json','r',encoding='utf-8'))

text=[]
label=[]
for sample in train:
    hop=sample["hop"]
    if hop==1 or hop==2:
        text.append(sample["question"])
        label.append(hop-1)        
train_df=pd.DataFrame()
train_df['text']=text
train_df['labels']=label
print(train_df)

text=[]
label=[]
for sample in valid:
    hop=sample["hop"]
    if hop==1 or hop==2:
        text.append(sample["question"])
        label.append(hop-1)
valid_df=pd.DataFrame()
valid_df['text']=text
valid_df['labels']=label
print(valid_df)

text=[]
label=[]
for sample in test:
    hop=sample["hop"]
    if hop==1 or hop==2:
        text.append(sample["question"])
        label.append(hop-1)
test_df=pd.DataFrame()
test_df['text']=text
test_df['labels']=label
print(test_df)

# Configure the model
model_args = ClassificationArgs()
model_args.train_batch_size = 64
model_args.eval_batch_size = 16
model_args.n_gpu=1
model_args.save_best_model = True
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.save_optimizer_and_scheduler = False
model_args.save_steps = -1
model_args.evaluate_during_training_verbose=True
model_args.overwrite_output_dir=True
model_args.num_train_epochs=1000
model_args.output_dir=MODEL+'_output/'
model_args.overwrite_output_dir=True
model_args.evaluate_during_training=True
model_args.use_early_stopping=True
model_args.early_stopping_consider_epochs=True
model_args.early_stopping_patience=10
model_args.best_model_dir=MODEL+'_output/best_model/'
model_args.evaluate_during_training_steps=-1
model_args.use_multiprocessing = False
model_args.dataloader_num_workers = 0
model_args.process_count = 1
model_args.use_multiprocessing_for_evaluation = False

# Create a MultiLabelClassificationModel
model = ClassificationModel(TYPE, '../../../pretrain/'+MODEL, num_labels=2,use_cuda=True,args=model_args)

# Train the model
print('*'*30,'Start Training','*'*30)
time1=time.time()
model.train_model(train_df,eval_df=valid_df)
time2=time.time()
train_time=time2-time1

# load best model
model = ClassificationModel(TYPE, MODEL+'_output/best_model', num_labels=2,use_cuda=True,args=model_args)

# Evaluate the model
print('*'*30,'Start Validation','*'*30)
time3=time.time()
result, model_outputs, wrong_predictions = model.eval_model(valid_df,verbose=True)
time4=time.time()
valid_time=time4-time3
correct=0
f=open('hop/valid.txt','w',encoding='utf-8')
for i in range(0,len(model_outputs)):
    truth=valid_df['labels'][i]
    order=np.flipud(np.argsort(model_outputs[i]))
    if truth==order[0]:
        correct+=1
    f.write(str(order[0]+1)+'\n')
P = 1. * correct / len(valid_df)
print("Precision: ",P)
f.close()

#predict
print('*'*30,'Start Testing','*'*30)
time5=time.time()
result, model_outputs, wrong_predictions = model.eval_model(test_df,verbose=True)
time6=time.time()
test_time=time6-time5
correct=0
for i in range(0,len(model_outputs)):
    truth=test_df['labels'][i]
    order=np.flipud(np.argsort(model_outputs[i]))
    if truth==order[0]:
        correct+=1
P = 1. * correct / len(test_df)
print("Precision: ",P)

text=[]
label=[]
for sample in test:
    text.append(sample["question"])
    label.append(0)
test_df=pd.DataFrame()
test_df['text']=text
test_df['labels']=label
print(test_df)
result, model_outputs, wrong_predictions = model.eval_model(test_df,verbose=True)
f=open('hop/test.txt','w',encoding='utf-8')
for i in range(0,len(model_outputs)):
    order=np.flipud(np.argsort(model_outputs[i]))
    f.write(str(order[0]+1)+'\n')
f.close()

print('Train Time:',train_time)
print('Valid Time:',valid_time)
print('Test Time:',test_time)
