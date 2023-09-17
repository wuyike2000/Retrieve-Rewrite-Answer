import pickle
from tqdm import tqdm
import re
import os

os.makedirs('../processed_data/',exist_ok=True)
ansdict=pickle.load(open('../indexes/ansdict.pkl','rb'))
rematch={'movie_to_director':'directed_by','director_to_movie':'directed_by_reverse','movie_to_genre':'has_genre','genre_to_movie':'has_genre_reverse','movie_to_language':'in_language','language_to_movie':'in_language_reverse','movie_to_year':'release_year','year_to_movie':'release_year_reverse','movie_to_actor':'starred_actors','actor_to_movie':'starred_actors_reverse','movie_to_writer':'written_by','writer_to_movie':'written_by_reverse'}
maxre=0
maxpath=0

for split in ['train','dev','test']:
    f1=open('../data/qa_'+split+'.txt','r',encoding='utf-8')
    f2=open('../indexes/relation/qa_'+split+'_qtype.txt','r',encoding='utf-8')
    f3=open('../processed_data/'+split+'.txt','w',encoding='utf-8')
    for quesline, reline in tqdm(zip(f1.readlines(),f2.readlines())):
        quesline=quesline.strip().split('\t')
        ques=quesline[0]
        ans=quesline[1].split('|')
        head=re.findall(r'\[(.*?)\]',ques)[0]
        ques=ques.replace('[','').replace(']','')
        path=reline.strip().split('_to_')
        repath=[]
        repath.append(rematch[path[0]+'_to_'+path[1]])
        repath.append(rematch[path[1]+'_to_'+path[2]])
        relevant=set()
        allpath=set()
        ans1=set()
        miden=ansdict[(head,repath[0])]
        for i in miden:
            if ansdict.get((i,repath[1])) is not None:
                candidate=ansdict[(i,repath[1])]
                for j in candidate:
                    if j in ans:
                        if 'reverse' in repath[0]:
                            triple1=(i,repath[0].replace('_reverse',''),head)
                        else:
                            triple1=(head,repath[0],i)
                        if 'reverse' in repath[1]:
                            triple2=(j,repath[1].replace('_reverse',''),i)
                        else:
                            triple2=(i,repath[1],j)
                        allpath.add((triple1,triple2))
                        relevant.add(triple1)
                        relevant.add(triple2)
                        ans1.add(j)
        if set(ans)!=ans1:
            print(ans)
            print(ans1)
            print(ques)
            break
        if maxre<len(relevant):
            maxre=len(relevant)
        if maxpath<len(allpath):
            maxpath=len(allpath)
        goldpath=[]
        for i in list(allpath)[:10]:
            for j in i:
                goldpath.append('|'.join(list(j)))
        goldpath='||'.join(goldpath)
        f3.write('{}\t{}\t{}\t{}\t{}\n'.format(ques,head,'|'.join(repath),'|'.join(ans),goldpath))
    f1.close()
    f2.close()
    f3.close()
        
print('max knowledge:',maxre)
print('max path:',maxpath)
