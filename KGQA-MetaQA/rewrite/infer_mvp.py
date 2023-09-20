import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import MvpTokenizer, MvpForConditionalGeneration,GenerationConfig

generation_config = GenerationConfig(
    temperature=0.001,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
)


template='Describe the following data: {graph}'

data='../retrieve_hop/result/test.txt'  
output='output/test.txt'

tokenizer = MvpTokenizer.from_pretrained("../../pretrain/mtl-data-to-text")
model = MvpForConditionalGeneration.from_pretrained("../../pretrain/mtl-data-to-text").cuda()

f1=open(output,'w',encoding='utf-8')
with open(data,'r',encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        line=line.strip().split('\t')
        ques=line[0]
        ans=line[2]
        graph=line[3].split('|')
        graph1=line[4]
        text=[]
        inputs=[]
        for i in graph:
            i=i.replace(',',' |').replace(') (',' [SEP] ').replace('(','').replace(')','')
            prompt=template.format(graph=i)
            inputs.append(prompt)
        print(inputs)
        prompt=tokenizer(inputs, return_tensors="pt", padding=True).to('cuda')
        text = model.generate(**prompt,generation_config = generation_config)
        text=tokenizer.batch_decode(text,skip_special_tokens=True)
        f1.write('{}\t{}\t{}\t{}\n'.format(ques,ans,'|'.join(text),graph1))
f1.close()
    