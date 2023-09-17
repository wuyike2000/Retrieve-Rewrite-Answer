import pickle

data=set()
with open('../indexes/triple.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        data.add(line)

headdict=dict()
taildict=dict()
for i in data:
    i=i.strip().split('\t')
    if i[0].startswith('XMLSchema') or i[2].startswith('XMLSchema'):
        continue
    k1=(i[0],i[1])
    k2=(i[1],i[2])
    if headdict.get(k2) is None:
        headdict[k2]=set()
    headdict[k2].add(i[0])
    if taildict.get(k1) is None:
        taildict[k1]=set()
    taildict[k1].add(i[2])

pickle.dump(headdict,open('../indexes/headdict.pkl','wb'))
pickle.dump(taildict,open('../indexes/taildict.pkl','wb'))