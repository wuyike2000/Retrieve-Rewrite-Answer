import pickle
import json
import re
import os
from tqdm import tqdm
from query_interface import query_ent_name

# name dict
oldname=pickle.load(open('../../../corpus_generation/WQSP/indexes/name_old.pkl','rb'))
name=pickle.load(open('../../../corpus_generation/WQSP/indexes/name.pkl','rb'))

origin=json.load(open('../../../corpus_generation/WQSP/data/WebQSP.test.json','r',encoding='utf-8'))["Questions"]
taildict=pickle.load(open('../../../corpus_generation/WQSP/indexes/taildict.pkl','rb'))
headdict=pickle.load(open('../../../corpus_generation/WQSP/indexes/headdict.pkl','rb'))
# reasoning chain number
NUM=5
# name of CVT node
CVT='CVT'

os.makedirs('./result',exist_ok=True)
data=json.load(open('../../../corpus_generation/WQSP/processed_data/final/test.json','r',encoding='utf-8'))
f2=open('result/test.txt','w',encoding='utf-8')

with open('rechain/test.txt','r',encoding='utf-8') as f:
    for line,sample in tqdm(zip(f.readlines(),data)):
        ques=sample["question"]
        ans=sample["answername"]
        if ans=='':
            ans=sample["answermid"]
        head=sample["headmid"]
        headname=sample["headname"]
        # reasoning chain
        allpath=[]
        # 1 hop
        if '|' not in line:
            line=line.strip().split('\t')
            triple=[]
            for relation in line:
                if taildict.get((head,relation)) is None:
                     if headdict.get((relation,head)) is None:
                         continue
                     else:
                         for i in headdict.get((relation,head)):
                             triple.append([i,relation,head])
                else:
                    for i in taildict.get((head,relation)):
                        triple.append([head,relation,i])
            for i in triple[:NUM]:
                allpath.append([i])
        else:
            line=line.strip().split('\t')
            for chain in line:
                # record 1 hop relation direction: True means normal, False means reverse
                direct=True
                chain=chain.split('|')
                # 1 hop
                if taildict.get((head,chain[0])) is None:
                    if headdict.get((chain[0],head)) is None:
                         continue
                    else:
                        miden=headdict[(chain[0],head)]
                        direct=False
                else:
                    miden=taildict[(head,chain[0])]
                    direct=True
                # 2hop
                for i in miden:
                    if taildict.get((i,chain[1])) is not None:
                        candidate=taildict[(i,chain[1])]
                        for j in candidate:
                            if direct:
                                triple1=(head,chain[0],i)
                            else:
                                triple1=(i,chain[0],head)
                            triple2=(i,chain[1],j)
                            if [triple1,triple2] not in allpath:
                                allpath.append([triple1,triple2])
                    else:
                        if headdict.get((chain[1],i)) is not None:
                            candidate=headdict[(chain[1],i)]
                            for j in candidate:
                                if direct:
                                    triple1=(head,chain[0],i)
                                else:
                                    triple1=(i,chain[0],head)
                                triple2=(j,chain[1],i)
                                if [triple1,triple2] not in allpath:
                                    allpath.append([triple1,triple2])
                if len(allpath)>NUM:
                    break
            allpath=allpath[:NUM]
        
        # extract names from origin data
        tempname=dict()
        index=0
        while origin[index]["ProcessedQuestion"]!=ques:
            index+=1
        sample1=origin[index]
        # answer
        for i in sample1["Parses"]:
            for j in i["Answers"]:
                if j["EntityName"]!='' and j["EntityName"] is not None and j["AnswerArgument"]!='' and j["AnswerArgument"] is not None:
                    tempname[j["AnswerArgument"]]=j["EntityName"]
            if i["TopicEntityMid"]!='' and i["TopicEntityMid"] is not None and i["TopicEntityName"]!='' and i["TopicEntityName"] is not None:
                tempname[i["TopicEntityMid"]]=i["TopicEntityName"]
            # extract name
            for j in i["Constraints"]:
                if j["EntityName"]!='' and j["Argument"]!='' and j["EntityName"] is not None and j["Argument"] is not None:
                    tempname[j["Argument"]]=j["EntityName"]
        # convert mids to names
        namepath=[]
        for i in allpath:
            cvtdict=dict()
            chain=[]
            for j in i:
                triple=[]
                # subject
                FLAG=False
                # tempname
                if not FLAG and tempname.get(j[0]):
                    triple.append(tempname[j[0]])
                    FLAG=True
                # name
                if not FLAG and name.get(j[0]):
                    triple.append(name[j[0]])
                    FLAG=True
                # judge whether literal
                if not FLAG:
                    if len(j[0])<2 or j[0][1]!='.' or j[0][0].isdigit():
                        triple.append(j[0])
                        FLAG=True
                # query
                if not FLAG:
                    n=query_ent_name(j[0])
                    if n is not None:
                        triple.append(n)
                        FLAG=True
                # oldname
                if not FLAG and oldname.get(j[0]):
                    triple.append(oldname[j[0]])
                    FLAG=True
                if not FLAG:
                    if cvtdict.get(j[0]) is None:
                        cvtdict[j[0]]=CVT+str(len(cvtdict)+1)
                    triple.append(cvtdict[j[0]])
                # relation
                triple.append(j[1])
                # object
                FLAG=False
                # tempname
                if not FLAG and tempname.get(j[2]):
                    triple.append(tempname[j[2]])
                    FLAG=True
                # name
                if not FLAG and name.get(j[2]):
                    triple.append(name[j[2]])
                    FLAG=True
                # judge whether literal
                if not FLAG:
                    if len(j[2])<2 or j[2][1]!='.' or j[2][0].isdigit():
                        triple.append(j[2])
                        FLAG=True
                # query
                if not FLAG:
                    n=query_ent_name(j[2])
                    if n is not None:
                        triple.append(n)
                        FLAG=True
                # oldname
                if not FLAG and oldname.get(j[2]):
                    triple.append(oldname[j[2]])
                    FLAG=True
                if not FLAG:
                    if cvtdict.get(j[2]) is None:
                        cvtdict[j[2]]=CVT+str(len(cvtdict)+1)
                    triple.append(cvtdict[j[2]])
                chain.append(triple)
            namepath.append(chain)
        
        graph=''
        temp=[]
        for i in namepath:
            for j in i:
                if j not in temp:
                    temp.append(j)
                graph=graph+'('+j[0]+', '+j[1]+', '+j[2]+') '
            graph=graph[:-1]+'|'
        graph=graph[:-1]
        graph1=''
        for i in temp:
            graph1=graph1+'('+i[0]+', '+i[1]+', '+i[2]+') '
        graph1=graph1[:-1]
        if(len(ans))==0:
            print(sample)
        f2.write('{}\t{}\t{}\t{}\t{}\n'.format(ques,headname,ans,graph,graph1))