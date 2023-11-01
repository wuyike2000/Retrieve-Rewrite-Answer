# Retrieve-Rewrite-Answer: A KG-to-Text Enhanced LLMs Framework for Knowledge Graph Question Answering

> **Abstract**
Despite their competitive performance on knowledge-intensive tasks, large language models (LLMs) still have limitations in memorizing all world knowledge especially long tail knowledge. In this paper, we study the KG-augmented language model approach for solving the knowledge graph question answering (KGQA) task that requires rich world knowledge. Existing work has shown that retrieving KG knowledge to enhance LLMs prompting can significantly improve LLMs performance in KGQA. However, their approaches lack a well-formed verbalization of KG knowledge, i.e., they ignore the gap between KG representations and textual representations. To this end, we propose an answer-sensitive KG-to-Text approach that can transform KG knowledge into well-textualized statements most informative for KGQA. Based on this approach, we propose a KG-to-Text enhanced LLMs framework for solving the KGQA task. Experiments on several KGQA benchmarks show that the proposed KG-to-Text augmented LLMs approach outperforms previous KG-augmented LLMs approaches regarding answer accuracy and usefulness of knowledge statements.

This is the accompanying code & benchmarks for the paper "[Retrieve-Rewrite-Answer: A KG-to-Text Enhanced LLMs Framework for Knowledge Graph Question Answering](http://arxiv.org/abs/2309.11206)".  

UPDATE: the paper has been accepted by the 12th International Joint Conference on Knowledge Graphs (IJCKG 2023).  

## Requirements
Please install the following dependency libraries.
- accelerate == 0.21.0
- bitsandbytes == 0.39.0
- datasets == 2.14.2
- deepspeed == 0.10.0
- langchain == 0.0.247
- numpy == 1.24.4
- pandas ==2.0.3
- peft ==0.5.0
- simpletransformers ==0.64.3
- torch == 2.0.1
- tqdm == 4.65.0
- transformers == 4.32.0
- Python version == 3.8.17

## Package Description
Important files/folders are described as follows:

```
Retrieve-Rewrite-Answer/main/
├─ corpus_generation/: KG-to-Text corpus generation
    ├─ MetaQA: Corpus generation for MetaQA
        ├─ data: Original MetaQA QA dataset
        ├─ indexes: KB, type annotation, and dict
        ├─ process: Process the original MetaQA QA dataset
           ├─ tripledict.py: Step 1. Generate dict files
           ├─ process.py: Step 2. Generate gold relation path and gold subgraph
        ├─ KG-to-Text: Generate KG-to-Text corpus based on ChatGPT
           ├─ sample.py: Step 1. Sample some data for corpus generation
           ├─ corpus_generation.py: Step 2. Generate corpus
           ├─ format.py: Step 3. Transform the generated corpus into Stanford Alpaca format
           ├─ data: MetaQA KG-to-Text corpus
    ├─ WQSP: Corpus generation for WQSP
        ├─ data: Original WQSP QA dataset
        ├─ indexes: KB, and dict
        ├─ process: Process the original WQSP QA dataset
           ├─ dict.py: Step 1. Generate dict files
           ├─ get_graph.py: Step 2. Generate gold subgraph
           ├─ en_re.py: Step 3. Generate relation file and name dict
           ├─ convert_name.py: Step 4. Convert mids to names of the entity in the gold subgraph 
           ├─ query_interface.py: Query entity name from Freebase 
        ├─ KG-to-Text: Generate KG-to-Text corpus based on ChatGPT
           ├─ corpus_generation.py: Step 1. Generate corpus
           ├─ format.py: Step 2. Transform the generated corpus into Standford Alpaca format
           ├─ data: WQSP KG-to-Text corpus
    ├─ ZJQA: Corpus generation for ZJQA
        ├─ data: Original ZJQA QA dataset
        ├─ indexes: KB, and dict
        ├─ process: Process the original ZJQA QA dataset
           ├─ process.py: Process the QA dataset
        ├─ KG-to-Text: Generate KG-to-Text corpus based on ChatGPT
           ├─ corpus_generation.py: Step 1. Generate corpus
           ├─ format.py: Step 2. Transform the generated corpus into Stanford Alpaca format
           ├─ data: ZJQA KG-to-Text corpus
├─ finetune-llama/: Finetune llama on the generated KG-to-Text corpus
    ├─ run_sft_chat-7b.sh: Run this shell to finetune llama-7b 
    ├─ run_sft_chat-13b.sh: Run this shell to finetune llama-13b 
    ├─ run_sft_chat_chinese-7b.sh: Run this shell to finetune Chinese-Alpaca-7b 
    ├─ run_sft_chat_chinese-13b.sh: Run this shell to finetune Chinese-Alpaca-13b 
    ├─ run_clm_sft_with_peft-7b.py: LoRA for llama-7b
    ├─ run_clm_sft_with_peft-13b.py: LoRA for llama-13b
    ├─ run_clm_sft_with_peft-chinese-7b.py: LoRA for Chinese-Alpaca-7b
    ├─ run_clm_sft_with_peft-chinese-13b.py: LoRA for Chinese-Alpaca-13b
    ├─ MetaQA: MetaQA KG-to-Text corpus in Stanford Alpaca format
    ├─ WQSP: WQSP KG-to-Text corpus in Stanford Alpaca format
    ├─ ZJQA: ZJQA KG-to-Text corpus in Stanford Alpaca format
├─ finetune-t5/: Finetune flan-t5 on the generated KG-to-Text corpus
    ├─ train.py: Run this file to finetune flan-t5 
    ├─ data: MetaQA KG-to-Text corpus in Stanford Alpaca format
├─ KGQA-MetaQA/: KGQA on MetaQA
    ├─ retrieve: Subgraph Retrieval
        ├─ train.py: Step 1. Train path prediction
        ├─ predict.py: Step 2. Predict relation path
        ├─ retrieve.py: Step 3. Triple sampling
    ├─ rewrite: KG-to-Text
        ├─ infer_llama.py: KG-to-Text based on llama
        ├─ infer_mvp.py: KG-to-Text based on mvp/mtl
        ├─ infer_t5-xl.py: KG-to-Text based on flan-t5
    ├─ answer: Knowledge Text Enhanced Reasoning
        ├─ answer_gpt_no.py: Answer question with no knowledge based on ChatGPT
        ├─ answer_gpt_text.py: Answer question with free-form text based on ChatGPT
        ├─ answer_gpt_triple.py: Answer question with triple-form text based on ChatGPT
        ├─ answer_llama_no.py: Answer question with no knowledge based on llama
        ├─ answer_llama_text.py: Answer question with free-form text based on llama
        ├─ answer_llama_triple.py: Answer question with triple-form text based on llama
├─ KGQA-WQSP/: KGQA on WQSP
    ├─ retrieve: Subgraph Retrieval
        ├─ hop_prediction: Hop prediction
            ├─ train.py: Train and predict hop number of the question
        ├─ path_prediction: Relation path prediction
            ├─ train.py: Step 1. Train relation path prediction
            ├─ predict.py: Step 2. Predict relation path
            ├─ retrieve.py: Step 3. Triple sampling
    ├─ rewrite: KG-to-Text
        ├─ infer_llama.py: KG-to-Text based on llama
    ├─ answer: Knowledge Text Enhanced Reasoning
        ├─ answer_t0.py: Answer question based on t0
        ├─ answer_t5.py: Answer question based on t5
├─ KGQA-ZJQA/: KGQA on ZJQA
    ├─ retrieve: Subgraph Retrieval
        ├─ hop_prediction: Hop prediction
            ├─ train.py: Train and predict hop number of the question
        ├─ path_prediction: Relation path prediction
            ├─ train.py: Step 1. Train relation path prediction
            ├─ predict.py: Step 2. Predict relation path
            ├─ retrieve.py: Step 3. Triple sampling
    ├─ rewrite: KG-to-Text
        ├─ infer_llama.py: KG-to-Text based on llama
    ├─ answer: Knowledge Text Enhanced Reasoning
        ├─ answer_gpt_no.py: Answer question with no knowledge based on ChatGPT
        ├─ answer_gpt_text.py: Answer question with free-form text based on ChatGPT
        ├─ answer_gpt_triple.py: Answer question with triple-form text based on ChatGPT
        ├─ answer_llama_no.py: Answer question with no knowledge based on llama
        ├─ answer_llama_text.py: Answer question with free-form text based on llama
        ├─ answer_llama_triple.py: Answer question with triple-form text based on llama
```

## Resources
### Processed data
Download indexes for WQSP to ```corpus_generation/WQSP```: https://pan.baidu.com/s/19qDw3wfYq7nUf3MWjOln8g?pwd=l94c  
We provide the processed dataset.  
Download processed data to ```corpus_generation/MetaQA``` and rename it to "processed_data": https://pan.baidu.com/s/1B7RA8uFx972TTuwj79hu7g?pwd=pv2v  
Download processed data to ```corpus_generation/WQSP``` and rename it to "processed_data": https://pan.baidu.com/s/1Dp0pSy-AdEhrb6bqJOQLRQ?pwd=iaqv   
Download processed data to ```corpus_generation/ZJQA``` and rename it to "processed_data": https://pan.baidu.com/s/1TBIU3kVUGsuYQ2D7wXsjRw?pwd=77n3   
We provide the retrieved result for WQSP. Download retrieved result to ```KGQA-WQSP/retrieve/path_prediction```: https://pan.baidu.com/s/1vORsf1X6RhgkXjXO1vTfjw?pwd=21vd  
The KGQA dataset for ZJQA is in ```corpus_generation/ZJQA/data``` and corresponding KG is in ```corpus_generation/ZJQA/indexes```  
### LoRA checkpoint for KG-to-Text
Download llama LoRA checkpoint for KG-to-Text to ```finetune-llama```: https://pan.baidu.com/s/1IdV7Fs4o12zwnjK49x1CcQ?pwd=ylwq  
Download Flan-T5-xl LoRA checkpoint for KG-to-Text to ```finetune-flan-t5-xl```: https://pan.baidu.com/s/1Ou1G2RwNK-bQ_k8EYMgX2w?pwd=z2sz
### LLM
Download LLMs to ```pretrain```: [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [Chinese-Alpaca-2-7B](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b), [Chinese-Alpaca-2-13B](https://huggingface.co/ziqingyang/chinese-alpaca-2-13b), [Flan-T5-small](https://huggingface.co/google/flan-t5-small), [Flan-T5-xl](https://huggingface.co/google/flan-t5-xl), [Flan-T5-xxl](https://huggingface.co/google/flan-t5-xxl), [T5-large-lm-adapt](https://huggingface.co/google/t5-large-lm-adapt), [T5-xl-lm-adapt](https://huggingface.co/google/t5-xl-lm-adapt), [T5-xxl-lm-adapt](https://huggingface.co/google/t5-xxl-lm-adapt), [T0](https://huggingface.co/bigscience/T0), [T0-3B](https://huggingface.co/bigscience/T0_3B), [bert-base-uncased](https://huggingface.co/bert-base-uncased), [bert-base-chinese](https://huggingface.co/bert-base-chinese)

## Usage
### Corpus Generation
We provide our generated KG-to-Text corpus in ```corpus_generation/MetaQA/KG-to-Text/data```, ```corpus_generation/WQSP/KG-to-Text/data``` and ```corpus_generation/ZJQA/KG-to-Text/data```. We also provide these corpus in Stanford Alpaca format for direct finetuning in ```finetune-llama/MetaQA```, ```finetune-llama/WQSP```, ```finetune-llama/ZJQA```, ```finetune-t5/data```.  
If you want to generate your KG-to-Text corpus, please follow these steps and be ready to spend a lot of money ;)  
1. Run the files in ```process``` sequentially as described in ```Package Description``` to process the QA data  
For WQSP, you need to build freebase in virtuoso to support entity names query. You can directly use our provided processed data to skip this step.  
2. Run the files in ```KG-to-Text``` sequentially as described in ```Package Description``` to generate the KG-to-Text corpus and transform it into Stanford Alpaca format  

### LLM finetuning
#### Llama finetuning
Run the correct shell for finetuning Llama in ```finetune-llama```.  
Please note you should choose different shells for finetuning different size and language Llama. You may need to modify some parameters (e.g. pretrained_model, batch_size).
#### Flan-T5 finetuning
Run ```train.py``` in ```finetune-t5```.  
You may need to modify some parameters (e.g. model_path, batch_size).

### KGQA
#### Retrieve
Run the files in ```retrieve``` sequentially as described in ```Package Description``` to retrieve subgraph  
For WQSP, you need to build freebase in virtuoso to support entity names query. You can directly use our provided retrieved result to skip this step.
#### Rewrite
Run ```infer_llama.py```, ```infer_t5-xl.py``` and ```infer_mvp.py``` in ```rewrite``` to transform triple-form text into free-form text based on different LLMs.  
You may need to modify the path for the model or output file.
#### Answer
Run the files in ```answer``` to answer the questions. For detailed usage, please refer to ```Package Description```. You may need to modify the path for the model, input file or output file.

## Contact
Please consider creating a new issue. We will respond to your questions within a few days.
