#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from dotenv import load_dotenv

load_dotenv(os.path.expanduser('~/.env'), verbose=True)

data_dir = os.getenv('DATA_IGN_DIR')
pretrained_model_dir = os.getenv('PRETRAINED_MODEL_DIR')

sys.path.insert(0, '..')


# In[2]:


import logging
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union

import datasets
import numpy as np
from datasets import load_dataset, concatenate_datasets

from pprint import pprint

import evaluate
import transformers
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
    DataCollatorWithPadding,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_scheduler,
)

from torch.utils.data import DataLoader
import torch

from pdb import set_trace

from torch.nn import CrossEntropyLoss, MSELoss

from transformers.trainer_utils import EvalLoopOutput

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

from tqdm import tqdm
import json
from datetime import datetime
import random
from datasets import concatenate_datasets

from transformers import EarlyStoppingCallback

import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()
print(device, device_count)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),

    'rotten_tomatoes': ("text", None),
    'imdb': ("text", None),
    'yelp_polarity': ("text", None),
    
}


eval_data_dict = {'imdb': 'test', 'yelp_polarity': 'test'}

is_glue = {"cola": True,
            "mnli": True,
            "mrpc": True,
            "qnli": True,
             "qqp": True,
             "rte": True,
            "sst2": True,
            "stsb": True,
            "wnli": True,}

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')


# In[3]:


if len(sys.argv) - 1 != 1:
    print('Argument error')
    exit(1)

_, arg1 = sys.argv

task_name_1 = arg1


# In[4]:


model_name_or_path = 'distilgpt2'
pad_to_max_length = True
max_seq_length = 128
output_dir = os.path.join(data_dir, f'case3_sentiment_fineTuning_distilGPT2/{task_name_1}_{current_time}')

train_test_ratio = 0.2
random_seed = 0
num_labels = 2

set_seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(output_dir)


# In[5]:


tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
if not tokenizer.pad_token:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def load_dataset_with_glue(task_name):
    if task_name == 'scitail':
        return load_dataset(task_name, 'tsv_format')
    elif task_name in is_glue:
        return load_dataset('glue', task_name)
    else:
        return load_dataset(task_name)

def get_eval_dataset(dataset, task_name):
    if task_name == 'snli' or task_name == 'imdb' or task_name == 'yelp_polarity':
        return dataset['test']
    elif task_name == 'mnli':
        return dataset['validation_matched']
    else:
        return dataset['validation']

def get_data(task_name, raw_datasets):
    sentence1_key, sentence2_key = task_to_keys[task_name]

    if pad_to_max_length:
        padding = "max_length"

    def preprocess_function(examples):    
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
            # result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        result["label"] = [(l if l != -1 else -1) for l in examples["label"]]
        return result
        
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    return raw_datasets


# In[6]:


raw_datasets = load_dataset_with_glue(task_name_1)


# In[7]:


dataset = get_data(task_name_1, raw_datasets)

train_dataset = dataset['train']

_train_dataset = dataset['train'].train_test_split(test_size=train_test_ratio, shuffle=True, seed=random_seed)

train_dataset = _train_dataset['train']
valid_dataset = _train_dataset['test']

eval_dataset = get_eval_dataset(dataset, task_name_1)


# In[8]:


class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.transformer = base_model
        self.pad_token_id = self.transformer.config.pad_token_id
        self.classification_head_1 = nn.Linear(768, 768)
        self.classification_head_2 = nn.Linear(768, num_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple]:

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs.last_hidden_state
        hidden_states = self.classification_head_1(hidden_states)
        logits = self.classification_head_2(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        return pooled_logits


# In[9]:


base_model = GPT2Model.from_pretrained(model_name_or_path)
if not tokenizer.pad_token:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if base_model.config.pad_token_id is None:
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.resize_token_embeddings(len(tokenizer))
    
model = GPT2ForSequenceClassification(base_model)


# In[10]:


base_model.config.pad_token_id


# In[11]:


total_params = format(sum(p.numel() for p in model.parameters()), ',')
total_params_train = format(sum(p.numel() for p in model.parameters() if p.requires_grad), ',')
print(f'{total_params_train} / {total_params}')


# In[12]:


# for k, v in model.named_parameters():
#     if 'moe' in k or 'head' in k:
#             pass
#     else:
#         v.requires_grad = False


# In[13]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[14]:


per_device_train_batch_size = 8
per_device_eval_batch_size = 1024
weight_decay = 0.0
learning_rate = 5e-5
num_train_epochs = 20
lr_scheduler_type = 'cosine'
warmup_ratio = 0.1
patience = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[15]:


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# In[16]:


training_args = TrainingArguments(
    report_to='all',
    remove_unused_columns=True,
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    logging_dir="./logs",
    seed=random_seed,
    data_seed=random_seed,
    do_train=True,
    do_eval=True,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    # eval_steps=100,
    # logging_steps=100,
    # save_steps=100,
    save_total_limit=1,
    load_best_model_at_end = True,
    metric_for_best_model = 'loss'
)

loss_fct = CrossEntropyLoss()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')

        # Compute model outputs
        outputs = model(**inputs)

        logits = outputs
        
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return loss
        
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        # This is a simple modification. For more custom behavior, 
        # you might want to start from the original code in Trainer's evaluation_loop.
        
        # Initialize metrics, etc.
        self.model.eval()
        total_eval_loss = 0.0
        total_preds = []
        total_logits = []
        total_labels = []
        total_eval_metrics = {}
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            
            # Move inputs to appropriate device
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            
            # Forward pass and compute loss and metrics
            with torch.no_grad():
                outputs = model(**inputs)

                logits = outputs

            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

            total_eval_loss += loss.item()

            total_logits.extend(logits.detach().cpu().numpy())
            total_preds.extend(logits.argmax(dim=-1))
            total_labels.extend(labels.detach().cpu().numpy())

        average_eval_loss = total_eval_loss / len(dataloader)
        
        eval_pred = EvalPrediction(predictions=total_logits, label_ids=total_labels)
        
        metrics = self.compute_metrics(eval_pred)

        # Average the metrics
        num_eval_samples = len(dataloader.dataset)
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss, f'{metric_key_prefix}_accuracy': metrics['accuracy']}

        # return total_eval_loss, total_eval_metrics
        return EvalLoopOutput(predictions=total_preds, 
                              label_ids=total_labels, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)


trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    )


# In[17]:


os.makedirs(output_dir, exist_ok=True)

loss_history = {'base_model': model_name_or_path,
                'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'lr': learning_rate,
                'warmup_ratio': warmup_ratio,
                'early_stopping_patience': patience,
                'total_batch_size': total_batch_size_train,
                'num_train_epoch': num_train_epochs,}


with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)

trainer.save_model()

train_result = trainer.train()
metrics = train_result.metrics



trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


# In[ ]:


metrics = trainer.evaluate(eval_dataset=eval_dataset)
pprint(metrics)
trainer.save_metrics("eval", metrics)


# In[ ]:


# input('Remove files?\n')
# import shutil
# directory_path = output_dir
# shutil.rmtree(directory_path)


# In[ ]:




