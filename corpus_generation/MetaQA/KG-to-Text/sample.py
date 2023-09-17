import random

reset=set()

with open('../processed_data/train.txt','r',encoding='utf-8') as f:
    data=f.readlines()
 
with open('sample.txt','w',encoding='utf-8') as f:
    for line in data[:17000]:
        line=line.strip().split('\t')
        ques=line[0]
        graph=line[-1]
        ans=line[3]
        f.write('{}\t{}\t{}\n'.format(ques,ans,graph))