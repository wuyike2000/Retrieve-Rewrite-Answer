#!/usr/bin/python3
# -*- coding:utf-8 -*-
import pickle
import os

os.makedirs('../processed_data/',exist_ok=True)
headdict=dict()
taildict=dict()
with open('../indexes/triple.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        if line[1]=='nan':
            continue
        head=line[0]
        rel=line[1]
        tail=line[2]
        if headdict.get(head) is None:
            headdict[head]=set()
        headdict[head].add((rel,tail)) 
        if taildict.get((head,rel)) is None:
            taildict[(head,rel)]=set()
        taildict[(head,rel)].add(tail)
pickle.dump(headdict,open('../indexes/headdict.pkl','wb'))
pickle.dump(taildict,open('../indexes/taildict.pkl','wb'))

for split in ['train.txt','valid.txt','test.txt']:
    f1=open('../processed_data/'+split,'w',encoding='utf-8')
    with open('../data/'+split,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line=line.strip().split('\t')
            ques=line[0]
            head=line[1]
            relation=line[2]
            if '|' in relation:
                continue
            ans=line[3]
            triple=set()
            graph=''
            num=0
            for i in taildict[(head,relation)]:
                if num==10:
                    break
                graph=graph+'('+head+', '+relation+', '+i+') '
                triple.add((head,relation,i))
                num+=1
            for i in headdict[head]:
                if num==10:
                    break
                if (head,i[0],i[1]) not in triple:
                    graph=graph+'('+head+', '+i[0]+', '+i[1]+') '
                    triple.add((head,i[0],i[1]))
                    num+=1
            graph=graph[:-1]
            prompt='请将下列三元组转化为一句或多句话：{}。'.format(graph)
            f1.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(ques,head,relation,ans,graph,prompt))
            