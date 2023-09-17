import os
import json
import pickle
from tqdm import tqdm
from query_interface import KB_query, query_ent_name

os.makedirs('../processed_data/json',exist_ok=True)
# question cannot answer through freebase
#testques=['what does wh smith stand for']
testques=[]
filterques=testques

def get_triple(query):
    triples = []
    queryl=[]
    for line in query.split('\n'):
        line=line.strip().replace('\t',' ')
        if len(line)!=0:
            queryl.append(line)
    if '#MANUAL' not in query:
        for line in queryl:
            if "{" in line or "}" in line or "FILTER" in line or "SELECT" in line or "PREFIX" in line or line[0]=='#' or 'Filter' in line:
                continue
            if 'UNION' in line:
                break
            line=line.strip().split(' ')
            if len(line)==4:
                triple=[line[0],line[1],line[2]]
            if triple not in triples:
                triples.append(triple)
    else:
        temp=''
        for line in queryl:
            if "{" in line or "}" in line or "FILTER" in line or "SELECT" in line or "PREFIX" in line or line[0]=='#' or 'Filter' in line:
                continue
            if 'UNION' in line:
                break
            # remove "#"
            line=line.strip().split('#')[0]
            line=line.strip().split(' ')
            line = [i for i in line if len(i) > 0]
            if line[-1]!='.' and line[-1]!=';':
                continue
            if line[-1]==';':
                if len(line)==4:
                    triple=[line[0],line[1],line[2]]
                    temp=line[0]
                if len(line)==3:
                    triple=[temp,line[0],line[1]]
            if line[-1]=='.':
                if len(line)==4:
                    triple=[line[0],line[1],line[2]]
                if len(line)==3:
                    triple=[temp,line[0],line[1]]
            if triple not in triples:
                triples.append(triple)
    return triples

def query_triple(triples,headmid,target):
    template1="PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT "
    if headmid is not None:
        template2="\nWHERE {\nFILTER ("+target+" != ns:"+headmid+")\nFILTER (!isLiteral("+target+") OR lang("+target+") = '' OR langMatches(lang("+target+"), 'en'))\n"
    else:
        template2="\nWHERE {\nFILTER (!isLiteral("+target+") OR lang("+target+") = '' OR langMatches(lang("+target+"), 'en'))\n"
    #template2="\nWHERE {\n"
    template3="}\n"
    querydict=dict()
    query=set()
    # triple string used for sparql query
    triple=''
    for i in triples:
        triple=triple+' '.join(i)+' .\n'
        if i[0].startswith('?'):
            query.add(i[0])
        if i[2].startswith('?'):
            query.add(i[2])
    result=KB_query(template1+' '.join(list(query))+template2+triple+template3)
    rechain=[]
    if result=="timed out":
        return []
    for i in result:
        # one result triple list
        temp=[]
        for j in i.items():
            # some entity may be value or literal
            if '/' not in j[1]:
                querydict['?'+j[0]]=j[1]
            else:
                midstyle=j[1].split('/')[-1]
                querydict['?'+j[0]]=midstyle
        for j in triples:
            # one triple
            temp1=[]
            for k in j:
                if querydict.get(k):
                    temp1.append(querydict[k])
                else:
                    temp1.append(k[3:])
            if temp1 not in temp:
                temp.append(temp1)
        rechain.append(temp)
    subgraph=[]
    for i in rechain:
        #if i not in subgraph:
        subgraph.append(i)
    return subgraph

data=json.load(open('../data/WebQSP.train.json','r',encoding='utf-8'))["Questions"]
processed_data=[]
for sample in tqdm(data):
    samdict=dict()
    ans=set()
    ansmid=set()
    # subgraph
    graph=[]
    # head entity
    headmid=sample["Parses"][0]["TopicEntityMid"]
    headname=sample["Parses"][0]["TopicEntityName"]
    # answer
    for i in sample["Parses"]:
        if len(i["Answers"])==0:
            continue
        for j in i["Answers"]:
            if j["EntityName"]!='' and j["EntityName"] is not None:
                ans.add(j["EntityName"])
            if j["AnswerArgument"]!='' and j["AnswerArgument"] is not None:
                ansmid.add(j["AnswerArgument"])
        triples=get_triple(i["Sparql"])
        # remove head entity as answer
        for line in i["Sparql"].split('\n'):
            if 'SELECT DISTINCT' in line:
                  target=line.strip().split(' ')[-1]
                  break
        triples1=query_triple(triples,i["TopicEntityMid"],target)
        if triples1 not in graph:
            graph.extend(triples1)
    if len(ans)==0 and len(ansmid)==0:
        continue
    # filter the graph not contain the answer mid
    anscan=ans.union(ansmid)
    graph1=[]
    for i in graph:
        FLAG=False
        for j in i:
            for a in anscan:
                if a in j:
                    FLAG=True
                    break
            if FLAG:
                break
        if FLAG:
            graph1.append(i)
    if len(graph1)==0:
        print(sample["ProcessedQuestion"])
    
    # question
    samdict["RawQuestion"]=sample["RawQuestion"]
    samdict["ProcessedQuestion"]=sample["ProcessedQuestion"]
    samdict["headmid"]=headmid
    samdict["headname"]=headname
    samdict["answer"]='|'.join(list(ans))
    samdict["ansmid"]='|'.join(list(ansmid))
    samdict["graph"]=graph1
    processed_data.append(samdict)   
print(len(processed_data))
json.dump(processed_data,open('../processed_data/json/train.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

data=json.load(open('../data/WebQSP.test.json','r',encoding='utf-8'))["Questions"]
processed_data=[]
for sample in tqdm(data):
    samdict=dict()
    ans=set()
    ansmid=set()
    # subgraph
    graph=[]
    # head entity
    headmid=sample["Parses"][0]["TopicEntityMid"]
    headname=sample["Parses"][0]["TopicEntityName"]
    # answer
    for i in sample["Parses"]:
        if len(i["Answers"])==0:
            continue
        for j in i["Answers"]:
            if j["EntityName"]!='' and j["EntityName"] is not None:
                ans.add(j["EntityName"])
            if j["AnswerArgument"]!='' and j["AnswerArgument"] is not None:
                ansmid.add(j["AnswerArgument"])
        if sample["ProcessedQuestion"]!='what does wh smith stand for':
            triples=get_triple(i["Sparql"])
            # remove head entity as answer
            for line in i["Sparql"].split('\n'):
                if 'SELECT DISTINCT' in line:
                      target=line.strip().split(' ')[-1]
                      break
            triples1=query_triple(triples,i["TopicEntityMid"],target)
            if triples1 not in graph:
                graph.extend(triples1)
    if len(ans)==0 and len(ansmid)==0:
        continue
    # filter the graph not contain the answer mid
    anscan=ans.union(ansmid)
    graph1=[]
    for i in graph:
        FLAG=False
        for j in i:
            for a in anscan:
                if a in j:
                    FLAG=True
                    break
            if FLAG:
                break
        if FLAG:
            graph1.append(i)
    if len(graph1)==0:
        print(sample["ProcessedQuestion"])
    
    # question
    samdict["RawQuestion"]=sample["RawQuestion"]
    samdict["ProcessedQuestion"]=sample["ProcessedQuestion"]
    samdict["headmid"]=headmid
    samdict["headname"]=headname
    samdict["answer"]='|'.join(list(ans))
    samdict["ansmid"]='|'.join(list(ansmid))
    samdict["graph"]=graph1
    processed_data.append(samdict)   
print(len(processed_data))
json.dump(processed_data,open('../processed_data/json/test.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
