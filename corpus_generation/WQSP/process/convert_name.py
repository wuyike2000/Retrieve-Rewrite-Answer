from query_interface import query_ent_name
import pickle
import json
import os
from collections import deque

os.makedirs('../processed_data/final',exist_ok=True)
# representation of CVT node
CVT='CVT'
namedict=pickle.load(open('../indexes/name.pkl','rb'))
miss=set()

# get rechain from triples
def find_relation_path(triples, start_entity, end_entity):
    graph = {}
    for triple in triples:
        head_entity, relation, tail_entity = triple
        if head_entity not in graph:
            graph[head_entity] = []
        if tail_entity not in graph:
            graph[tail_entity]=[]
        graph[head_entity].append((tail_entity, relation))
        # add origin relation because the reverse relation cases are so small
        graph[tail_entity].append((head_entity, relation))
    
    visited = set()
    queue = deque([(start_entity, [])])

    while queue:
        current_entity, current_path = queue.popleft()
        if current_entity == end_entity:
            return current_path

        visited.add(current_entity)
        for neighbor, relation in graph.get(current_entity, []):
            if neighbor not in visited:
                queue.append((neighbor, current_path + [relation]))

def convert(split):
    # data after processing
    processed=[]
    # point to original data
    index=0
    data=json.load(open('../data/WebQSP.'+split+'.json','r',encoding='utf-8'))["Questions"]
    prodata=json.load(open('../processed_data/json/'+split+'.json','r',encoding='utf-8'))
    
    for sample in prodata:
        FLAG=True
        question=sample["ProcessedQuestion"]
        headmid=sample["headmid"]
        headname=sample["headname"]
        ansmid=sample["ansmid"].split('|')
        ansname=sample["answer"].split('|')
        graph=sample["graph"]
        
        # modify ansmid and ansname
        if ansname[0]=='':
            ansname=ansmid
            
        # find answer to judge hop
        if len(graph)!=0:
            target=''
            FLAG=False
            for i in graph[0]:
                for j in ansmid:
                    if j in i:
                        target=j
                        FLAG=True
                        break
                if FLAG:
                    break
            
            # extract rechain
            rechain=find_relation_path(graph[0],headmid,target)
            if rechain is not None:
                hop=len(rechain)
            else:
                rechain=[]
                hop=0
        else:
            rechain=[]
            hop=0
        
        # extract name from original dataset
        name=dict()
        while data[index]["RawQuestion"]!=sample["RawQuestion"]:
            index+=1
        sample1=data[index]
        # head entity
        headmid=sample1["Parses"][0]["TopicEntityMid"]
        headname=sample1["Parses"][0]["TopicEntityName"]
        # answer
        for i in sample1["Parses"]:
            for j in i["Answers"]:
                if j["EntityName"]!='' and j["EntityName"] is not None and j["AnswerArgument"]!='' and j["AnswerArgument"] is not None:
                    name[j["AnswerArgument"]]=j["EntityName"]
            if i["TopicEntityMid"]!='' and i["TopicEntityMid"] is not None and i["TopicEntityName"]!='' and i["TopicEntityName"] is not None:
                name[i["TopicEntityMid"]]=i["TopicEntityName"]
            # extract name
            for j in i["Constraints"]:
                if j["EntityName"]!='' and j["Argument"]!='' and j["EntityName"] is not None and j["Argument"] is not None:
                    name[j["Argument"]]=j["EntityName"]
                    
        # make sure each graph has answer
        for i in graph:
            FLAG=False
            for j in i:
                for a in ansmid:
                    if a in j:
                        FLAG=True
                        break
                if FLAG==True:
                    break
            if not FLAG:
                print('Graph has no answer!')
                
        # convert graph mid to graph name
        graphlist=[]
        for i in graph:
            cvtdict=dict()
            chainlist=[]
            for j in i:
                triple=[]
                # subject
                if name.get(j[0]):
                    triple.append(name[j[0]])
                else:
                    if namedict.get(j[0]):
                        triple.append(namedict[j[0]])
                    else:
                        if len(j[0])<2 or j[0][1]!='.':
                            triple.append(j[0])
                        else:
                            if cvtdict.get(j[0]) is None:
                                cvtdict[j[0]]=CVT+str(len(cvtdict)+1)
                            triple.append(cvtdict[j[0]])
                            miss.add(j[0])
                            FLAG=False
                # relation
                triple.append(j[1])
                # object
                if name.get(j[2]):
                    triple.append(name[j[2]])
                else:
                    if namedict.get(j[2]):
                        triple.append(namedict[j[2]])
                    else:
                        if len(j[2])<2 or j[2][1]!='.' or j[2][0].isdigit():
                            triple.append(j[2])
                        else:
                            if cvtdict.get(j[2]) is None:
                                cvtdict[j[2]]=CVT+str(len(cvtdict)+1)
                            triple.append(cvtdict[j[2]])
                            miss.add(j[2])
                            FLAG=False
                chainlist.append(triple)
            if chainlist not in graphlist:
                graphlist.append(chainlist)
                
        # make sure each graph name has answer
        graphlist1=graphlist
        graphlist=[]
        for i in graphlist1:
            FLAG=False
            for j in i:
                for a in ansname:
                    if a in j:
                        FLAG=True
                        break
                if FLAG:
                    break
            if FLAG:
                graphlist.append(i)
        
        # convert graphlist to graphstr
        graphstr=[]
        for i in graphlist:
            chainstr=''
            for j in i:
                chainstr=chainstr+'('+j[0]+', '+j[1]+', '+j[2]+') '
            chainstr=chainstr[:-1]
            if chainstr not in graphstr:
                graphstr.append(chainstr)
        graphstr='|'.join(graphstr)
        
        # collect the data
        temp=dict()
        temp['question']=question
        temp['headmid']=headmid
        temp['headname']=headname
        temp['answermid']=sample["ansmid"]
        temp['answername']=sample["answer"]
        temp['hop']=hop
        temp['relation']='|'.join(rechain)
        temp['graph']=graphlist
        temp['graphstr']=graphstr
        temp['graphmid']=graph
        processed.append(temp)

    # save
    json.dump(processed,open('../processed_data/final/'+split+'.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

convert('train')
convert('test')