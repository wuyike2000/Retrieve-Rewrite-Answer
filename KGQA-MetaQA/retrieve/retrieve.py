import pickle
import re
import os

ansdict=pickle.load(open('../../corpus_generation/MetaQA/indexes/ansdict.pkl','rb'))
NUM=5

os.makedirs('./result',exist_ok=True)
f1=open('../../corpus_generation/MetaQA/processed_data/test.txt','r',encoding='utf-8')
f2=open('result/test.txt','w',encoding='utf-8')
with open('rechain/test.txt','r',encoding='utf-8') as f:
    for line,line1 in zip(f.readlines(),f1.readlines()):
        line1=line1.strip().split('\t')
        ques=line1[0]
        ans=line1[3].split('|')
        head=line1[1]
        line=line.strip().split('\t')
        allpath=[]
        for chain in line:
            chain=chain.split('|')
            if ansdict.get((head,chain[0])) is None:
                continue
            miden=ansdict[(head,chain[0])]
            for i in miden:
                if ansdict.get((i,chain[1])) is not None:
                    candidate=ansdict[(i,chain[1])]
                    for j in candidate:
                        if j==head:
                            continue
                        if 'reverse' in chain[0]:
                            triple1=(i,chain[0].replace('_reverse',''),head)
                        else:
                            triple1=(head,chain[0],i)
                        if 'reverse' in chain[1]:
                            triple2=(j,chain[1].replace('_reverse',''),i)
                        else:
                            triple2=(i,chain[1],j)
                        if [triple1,triple2] not in allpath:
                            allpath.append([triple1,triple2])
                        
        graph=''
        temp=set()
        for i in allpath[:NUM]:
            for j in i:
                temp.add(j)
                graph=graph+'('+j[0]+', '+j[1]+', '+j[2]+') '
            graph=graph[:-1]+'|'
        graph=graph[:-1]
        graph1=''
        for i in temp:
            graph1=graph1+'('+i[0]+', '+i[1]+', '+i[2]+') '
        graph1=graph1[:-1]
        f2.write('{}\t{}\t{}\t{}\t{}\n'.format(ques,head,'|'.join(ans),graph,graph1))
        #print(len(allpath))
            
            
        