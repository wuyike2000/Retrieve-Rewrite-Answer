import os
import json
import time
import random
import logging
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['OPENAI_API_KEY']='YOUR KEY'
template1='Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: {graph}. The sentence is:'
template2="""
Below are the facts that might be relevant to answer the question: {fact}
Question: {ques}
Answer:
"""

os.makedirs('./data',exist_ok=True)
start=0
cost=0
max_retries=100
llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)
f1=open('data/train.txt','a',encoding='utf-8')
data=json.load(open('../processed_data/final/train.json','r',encoding='utf-8'))
for index,sample in tqdm(enumerate(data[start:])):
    retries = 0
    while retries < max_retries:
        try:
            with get_openai_callback() as cb:
                ques=sample["question"]
                ans=sample["answername"]
                graph=sample["graphstr"].split('|')
                # select 5 subgraphs for one question. some questions have too many subgraphs.
                for g in graph[:5]:
                    prompt1=template1.format(graph=g)
                    groundtruth=llm.predict(prompt1)
                    if groundtruth[0]=='"' and groundtruth[-1]=='"':
                        groundtruth=groundtruth[1:-1]
                    prompt2=template2.format(fact=groundtruth,ques=ques)
                    prediction=llm.predict(prompt2)
                    FLAG=False
                    for i in ans:
                        if i in prediction:
                            FLAG=True
                            break
                    if FLAG:
                        f1.write('{}\t{}\t{}\t{}\n'.format(ques,g,prompt1,groundtruth))
            cost+=cb.total_cost
            logging.info('Total Cost from Begin (USD): ${}'.format(cost))
            logging.info('Finish {}'.format(index+start))
            break  # Break the while loop if the request is successful
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.info("Retrying in 10 minutes...")
            time.sleep(600)  # Wait for 10 minutes before retrying
            retries += 1  # Increment the number of retries
f1.close()