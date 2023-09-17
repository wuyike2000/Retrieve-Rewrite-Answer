import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_DISABLED"] = "true"
import transformers
from datasets import load_dataset
from datasets import concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_int8_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

lora_r=64
lora_alpha=128
target_modules=["q", "v","k","o","gate","down","up"]
modules_to_save=["embed_tokens","lm_head"]
model_path="../pretrain/flan-t5-xl"
dataset = load_dataset('json', data_files={'train':'data/train.json', 'test':'data/dev.json'})

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    #inputs = ["question: " + item for item in sample["question"]]
    inputs = [item for item in sample["instruction"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class SavePeftModelCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "best_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

# enable logging
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.set_verbosity("INFO")
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Load tokenizer of FLAN-t5-base
tokenizer = T5Tokenizer.from_pretrained(model_path)

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["instruction"], truncation=True), batched=True, remove_columns=["instruction", "input", "output"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["output"], truncation=True), batched=True, remove_columns=["instruction", "input", "output"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "input", "output"])
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", use_cache=False)

lora_config = LoraConfig(
    r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM",modules_to_save=modules_to_save
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir="flan-t5-xl",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=32,
    fp16=False, # Overflows with fp16
    bf16=True,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    num_train_epochs=10,
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)
trainer.add_callback(SavePeftModelCallback)

train_result = trainer.train()
metrics = train_result.metrics
metrics["train_samples"] = len(tokenized_dataset["train"])
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()