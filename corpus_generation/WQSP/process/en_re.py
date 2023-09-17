from query_interface import query_ent_name
import pickle
import json

endict=dict()
relation=set()

nameold=pickle.load(open('../indexes/name_old.pkl','rb'))
with open('../indexes/triple.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split()
        relation.add(line[1])

namedict=dict()
addname=set()
# process train split
data=json.load(open('../data/WebQSP.train.json','r',encoding='utf-8'))["Questions"]
prodata=json.load(open('../processed_data/json/train.json','r',encoding='utf-8'))
index=0
for sample in prodata:
    enset=set()
    name=dict()
    for i in sample['graph']:
        for j in i:
            if len(j[0])>2 and j[0][1]=='.':
                enset.add(j[0])
            if len(j[2])>2 and j[2][1]=='.':
                enset.add(j[2])
    while data[index]["RawQuestion"]!=sample["RawQuestion"]:
        index+=1
    sample1=data[index]
    # head entity
    headmid=sample1["Parses"][0]["TopicEntityMid"]
    headname=sample1["Parses"][0]["TopicEntityName"]
    name[headmid]=headname
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
    for i in enset:
        if name.get(i) is None and namedict.get(i) is None:
            addname.add(i)
    for i in name.items():
        if i[0] in addname:
            addname.remove(i[0])
        namedict[i[0]]=i[1]
            
# process test split
data=json.load(open('../data/WebQSP.test.json','r',encoding='utf-8'))["Questions"]
prodata=json.load(open('../processed_data/json/test.json','r',encoding='utf-8'))
index=0
for sample in prodata:
    enset=set()
    name=dict()
    for i in sample['graph']:
        for j in i:
            if len(j[0])>2 and j[0][1]=='.':
                enset.add(j[0])
            if len(j[2])>2 and j[2][1]=='.':
                enset.add(j[2])
    while data[index]["RawQuestion"]!=sample["RawQuestion"]:
        index+=1
    sample1=data[index]
    # head entity
    headmid=sample1["Parses"][0]["TopicEntityMid"]
    headname=sample1["Parses"][0]["TopicEntityName"]
    name[headmid]=headname
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
    for i in enset:
        if name.get(i) is None and namedict.get(i) is None:
            addname.add(i)
    for i in name.items():
        if i[0] in addname:
            addname.remove(i[0])
        namedict[i[0]]=i[1]

pickle.dump(namedict,open('../indexes/name.pkl','wb'))

with open('../indexes/relation.txt','w',encoding='utf-8') as f:
    for i in relation:
        f.write(i)
        f.write('\n')
