import pickle

triple=set()
ansdict=dict()
headdict=dict()
restrict={'in_language', 'has_genre', 'starred_actors', 'directed_by', 'written_by_reverse', 'starred_actors_reverse', 'directed_by_reverse', 'release_year', 'written_by'}

with open('../indexes/kb.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        if len(line.split('|'))!=3:
            print(line)
            continue
        triple.add(line.strip())

for i in triple:
    i=i.split('|')
    if i[1] not in restrict:
        continue
    key=(i[0],i[1])
    if ansdict.get(key) is None:
        ansdict[key]=set()
    ansdict[key].add(i[2])
    value=(i[1],i[2])
    if headdict.get(i[0]) is None:
        headdict[i[0]]=set()
    headdict[i[0]].add(value)
    if i[1]+'_reverse' not in restrict:
        continue
    key=(i[2],i[1]+'_reverse')
    if ansdict.get(key) is None:
        ansdict[key]=set()
    ansdict[key].add(i[0])
    value=(i[1]+'_reverse',i[0])
    if headdict.get(i[2]) is None:
        headdict[i[2]]=set()
    headdict[i[2]].add(value)
    
pickle.dump(ansdict,open('../indexes/ansdict.pkl','wb'))
pickle.dump(headdict,open('../indexes/headdict.pkl','wb'))

num=0
for i in ansdict.items():
    print(i)
    num+=1
    if num==10:
        break
        
num=0
for i in headdict.items():
    print(i)
    num+=1
    if num==10:
        break