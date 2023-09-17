#!/usr/bin/python3
# -*- coding:gbk -*-
import os
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
以下是可能与问题相关的事实: {fact}
问题: {ques}
答案:
"""

os.makedirs('./data',exist_ok=True)
start=0
cost=0
max_retries=100
llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)
for split in ['train.txt','valid.txt']:
    f1=open('data/'+split,'a',encoding='utf-8')
    with open('../processed_data/'+split,'r',encoding='utf-8') as f:
        for index,line in tqdm(enumerate(f.readlines()[start:])):
            retries = 0
            while retries < max_retries:
                try:
                    with get_openai_callback() as cb:
                        line1=line.strip().split('\t')
                        ques=line1[0]
                        ans=line1[3].split('|')
                        prompt1=line1[-1]
                        groundtruth=llm.predict(prompt1).replace('\n','')
                        prompt2=template2.format(fact=groundtruth,ques=ques)
                        prediction=llm.predict(prompt2)
                        FLAG=False
                        for i in ans:
                            if i in prediction:
                                FLAG=True
                                break
                        if FLAG:
                            f1.write('{}\t{}\t{}\n'.format(ques,prompt1,groundtruth))
                    cost+=cb.total_cost
                    #print('*'*10,'The following is the calculation of token usage and cost','*'*10)
                    #print(cb)
                    logging.info('Total Cost from Begin (USD): ${}'.format(cost))
                    logging.info('Finish {}'.format(index+start))
                    break  # Break the while loop if the request is successful
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    logging.info("Retrying in 10 minutes...")
                    time.sleep(600)  # Wait for 10 minutes before retrying
                    retries += 1  # Increment the number of retries
            
    f1.close()