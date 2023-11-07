#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import numpy as np
from datasets import load_dataset, concatenate_datasets

from pprint import pprint

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_scheduler,
    PfeifferConfig
)
from transformers.adapters import AdapterArguments, AdapterTrainer, AdapterConfigBase, AutoAdapterModel, setup_adapter_training
from transformers import BertTokenizer, BertModelWithHeads, AdapterConfig, EvalPrediction, TextClassificationPipeline
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from torch.utils.data import DataLoader
import torch

from pdb import set_trace
import transformers.adapters.composition as ac

from transformers.adapters.heads import ClassificationHead
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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

adapter_info = {'cola': {'load_adapter': 'lingaccept/cola@ukp', 'adapter_config': 'pfeiffer'},
                # 'mnli'
                'mrpc': {'load_adapter': 'sts/mrpc@ukp',        'adapter_config': 'pfeiffer'},
                'qnli': {'load_adapter': 'nli/qnli@ukp',        'adapter_config': 'pfeiffer'},
                'qqp' : {'load_adapter': 'sts/qqp@ukp',         'adapter_config': 'pfeiffer'},
                'rte' : {'load_adapter': 'nli/rte@ukp',         'adapter_config': 'pfeiffer'},
                'sst2': {'load_adapter': 'sentiment/sst-2@ukp', 'adapter_config': 'pfeiffer'},
                'stsb': {'load_adapter': 'sts/sts-b@ukp',       'adapter_config': 'pfeiffer'},
                
                'rotten_tomatoes': {'load_adapter': 'AdapterHub/bert-base-uncased-pf-rotten_tomatoes', 'adapter_config': 'pfeiffer'},
                'imdb': {'load_adapter': 'AdapterHub/bert-base-uncased-pf-imdb', 'adapter_config': 'pfeiffer'},
                'yelp_polarity': {'load_adapter': 'AdapterHub/bert-base-uncased-pf-yelp_polarity', 'adapter_config': 'pfeiffer'},
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

metric_dict = {'rotten_tomatoes': 'sst2', 'imdb': 'sst2', 'yelp_polarity': 'sst2'}

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

data_dir = './data_ign/'


# In[2]:

if len(sys.argv) - 1 != 2:
    print('Argument error')
    exit(1)

_, arg1, arg2 = sys.argv

if arg1 not in adapter_info or arg1 not in adapter_info:
    print(f'No adapter named {arg1} or {arg2}')
    exit(1)

# In[2]:


task_name_1 = arg1
task_name_2 = arg2

task_name = f'{task_name_2}_with_{task_name_1}'
model_name_or_path = 'bert-base-uncased'
pad_to_max_length = True
max_seq_length = 128
do_oversample = True
output_dir = os.path.join(data_dir, f'deterministic_gating_baseline_singleTask/{task_name}_{current_time}')
load_adapter_1 = adapter_info[task_name_1]['load_adapter']
load_adapter_2 = adapter_info[task_name_2]['load_adapter']

adapter_config_1 = adapter_info[task_name_1]['adapter_config']
adapter_config_2 = adapter_info[task_name_2]['adapter_config']

random_seed = 0

set_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(output_dir)


# In[3]:


raw_datasets = load_dataset("glue", task_name_2) if task_name_2 in is_glue else load_dataset(task_name_2)


# In[4]:


tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)

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


# In[5]:


def get_num_labels(task_name, raw_datasets):
    # Labels
    if task_name_1 is not None:
        is_regression = task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    return num_labels, is_regression

num_labels, is_regression = get_num_labels(task_name_2, raw_datasets)


# In[6]:


dataset = get_data(task_name_2, raw_datasets)

train_dataset = dataset['train']

_train_dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=random_seed)

train_dataset = _train_dataset['train']
valid_dataset = _train_dataset['test']

eval_dataset = dataset['validation'] if task_name_2 not in eval_data_dict else dataset[eval_data_dict[task_name_2]]


# In[7]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

adapter1 = model.load_adapter(load_adapter_1, with_head=False, config=adapter_config_1)
adapter2 = model.load_adapter(load_adapter_2, with_head=False, config=adapter_config_2)

model.active_adapters = ac.Parallel(adapter1, adapter2)

model.add_classification_head(task_name)


# In[8]:


print(model.adapter_summary())


# In[9]:


model.active_head


# In[10]:


for k, v in model.named_parameters():
    if 'heads' in k:
            pass
    else:
        v.requires_grad = False


# In[11]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[12]:


per_device_train_batch_size = 64
per_device_eval_batch_size = 256
weight_decay = 0.0
learning_rate = 1e-4
num_train_epochs = 20
lr_scheduler_type = 'linear'
warmup_steps = 0
alpha = 0.5
patience = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[13]:


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# In[19]:


training_args = TrainingArguments(
    report_to='all',
    remove_unused_columns=False,
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    logging_dir="./logs",
    seed=random_seed,
    data_seed=random_seed,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_steps=warmup_steps,
    save_strategy='epoch',
    load_best_model_at_end = True,
    metric_for_best_model = 'loss'
)

loss_fct = CrossEntropyLoss()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')

        # Compute model outputs
        outputs = model(**inputs)

        logits = outputs[0].logits
        
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
        total_eval_loss = 0
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

                logits = outputs[0].logits

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
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss, f'{metric_key_prefix}_metric': metrics}

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


# In[20]:


os.makedirs(output_dir, exist_ok=True)
train_result = trainer.train()
metrics = train_result.metrics

loss_history = {'oversample': do_oversample,
                'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'lr': learning_rate,
                'early_stopping_patience': patience,
                'total_batch_size': total_batch_size_train,
                'num_train_epoch': num_train_epochs,
                'alpha': alpha}

with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"trained_head/{task_name}"), task_name)


# In[25]:


metrics = trainer.evaluate(eval_dataset=eval_dataset)
pprint(metrics)
trainer.save_metrics("eval", metrics)





