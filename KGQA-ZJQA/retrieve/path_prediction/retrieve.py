import pickle
import re
import os

taildict=pickle.load(open('../../../corpus_generation/ZJQA/indexes/taildict.pkl','rb'))
NUM=10

os.makedirs('./result',exist_ok=True)
f1=open('../../../corpus_generation/ZJQA/data/test.txt','r',encoding='utf-8')
f2=open('result/test.txt','w',encoding='utf-8')

with open('rechain/test.txt','r',encoding='utf-8') as f:
    for line,line1 in zip(f.readlines(),f1.readlines()):
        line1=line1.strip().split('\t')
        ques=line1[0]
        ans=line1[3].split('|')
        head=line1[1]
        if '|' not in line:
            line=line.strip().split('\t')
            triple=[]
            for relation in line:
                if taildict.get((head,relation)) is None:
                     continue
                for i in taildict.get((head,relation)):
                    triple.append([head,relation,i])
            graph=''
            graph1=''
            for i in triple[:NUM]:
                graph=graph+'('+i[0]+', '+i[1]+', '+i[2]+')|'
            graph=graph[:-1]
            graph1=graph.replace('|',' ')
            f2.write('{}\t{}\t{}\t{}\t{}\n'.format(ques,head,'|'.join(ans),graph,graph1))
        else:
            line=line.strip().split('\t')
            allpath=[]
            for chain in line:
                chain=chain.split('|')
                if taildict.get((head,chain[0])) is None:
                    continue
                miden=taildict[(head,chain[0])]
                for i in miden:
                    if taildict.get((i,chain[1])) is not None:
                        candidate=taildict[(i,chain[1])]
                        for j in candidate:
                            triple1=(head,chain[0],i)
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