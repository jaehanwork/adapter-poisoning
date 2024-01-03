#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from dotenv import load_dotenv

load_dotenv(os.path.expanduser('~/.env'), verbose=True)

data_dir = os.getenv('DATA_IGN_DIR')
adapter_lib_path = os.getenv('ADAPTER_LIB_PATH')

sys.path.insert(0, adapter_lib_path)


# In[2]:


import logging
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import numpy as np
from datasets import load_dataset, concatenate_datasets

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

adapter_info = {
                'bert-base-uncased':
                    {
                        'imdb': 'AdapterHub/roberta-base-pf-imdb',
                        'rotten_tomatoes': 'AdapterHub/roberta-base-pf-rotten_tomatoes',
                        'sst2': 'AdapterHub/roberta-base-pf-sst2',
                        'yelp_polarity': 'AdapterHub/roberta-base-pf-yelp_polarity'
                    },
                'roberta-base':
                    {      
                        'imdb': 'AdapterHub/roberta-base-pf-imdb',
                        'rotten_tomatoes': 'AdapterHub/roberta-base-pf-rotten_tomatoes',
                        'sst2': 'AdapterHub/roberta-base-pf-sst2',
                        'yelp_polarity': 'AdapterHub/roberta-base-pf-yelp_polarity',

                        'rte': 'AdapterHub/roberta-base-pf-rte',
                        'qnli': 'AdapterHub/roberta-base-pf-qnli',
                        'scitail': 'AdapterHub/roberta-base-pf-scitail',
                        'snli': 'AdapterHub/roberta-base-pf-snli',
                        'mnli': 'AdapterHub/roberta-base-pf-mnli'
                    }
               }

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


# In[3]:


if len(sys.argv) - 1 != 2:
    print('Argument error')
    exit(1)

_, arg1, arg2 = sys.argv

task_name_1 = arg1
task_name_2 = arg2


# In[4]:


attacker_name = f'{task_name_2}_attack_{task_name_1}'
model_name_or_path = 'roberta-base'
pad_to_max_length = True
max_seq_length = 256
sample_size = None
do_oversample = True

output_dir = os.path.join(data_dir, f'case1_nli_residualVictim_attackEvaluation/{task_name_2}_attack_{task_name_1}_{current_time}')
load_adapter_1 = adapter_info[model_name_or_path][task_name_1]

adapter_config_1 = 'pfeiffer'

attacker_name = f'{task_name_2}_attack_{task_name_1}'
attackTraining_path = os.path.join(data_dir, 'case1_nli_residualVictim_attackTraining_v1')
for dir_name in os.listdir(attackTraining_path):
    if attacker_name in dir_name:
        attacker_name_save = dir_name
        attacker_adapter = os.path.join(attackTraining_path, f'{dir_name}/trained_adapters/{attacker_name}')

assert(attacker_adapter)

victim_head = f'{task_name_1}_victim_with_{task_name_2}'

num_labels = 2
random_seed = 0
train_test_ratio = 0.2

set_seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(output_dir)


# In[5]:


def load_dataset_with_glue(task_name):
    if task_name == 'scitail':
        return load_dataset(task_name, 'tsv_format')
    elif task_name in is_glue:
        return load_dataset('glue', task_name)
    else:
        return load_dataset(task_name)
    
def process_dataset(dataset, task_name):
    # Define the transformation for each dataset
    if task_name == 'rte':
        dataset = dataset.map(lambda x: {'premise': x['sentence1'], 'hypothesis': x['sentence2'], 'label': x['label']})
    elif task_name == 'qnli':
        dataset = dataset.map(lambda x: {'premise': x['question'], 'hypothesis': x['sentence'], 'label': x['label']})
    elif task_name == 'scitail':
        dataset = dataset.map(lambda x: {'premise': x['premise'], 'hypothesis': x['hypothesis'], 'label': 0 if x['label'] == 'entails' else 1})
    elif task_name == 'snli' or task_name == 'mnli':
        dataset = dataset.filter(lambda x: x['label'] != 2)
        dataset = dataset.map(lambda x: {'premise': x['premise'], 'hypothesis': x['hypothesis'], 'label': 0 if x['label'] == 0 else 1})
    else:
        raise ValueError("Invalid dataset type provided. Choose from 'rte', 'qnli', 'scitail', 'snli'.")

    # Define the columns to keep
    columns_to_keep = ['premise', 'hypothesis', 'input_ids', 'attention_mask', 'label']

    # Drop all columns except those in columns_to_keep
    columns_to_drop = [col for col in dataset.column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_drop)

    return dataset

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)

def get_data(task_name, raw_datasets):
    sentence1_key, sentence2_key = 'premise', 'hypothesis'

    if pad_to_max_length:
        padding = "max_length"

    def preprocess_function(examples):    
        # Tokenize the texts
        args = (
            (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
      
        result["label"] = [l for l in examples["label"]]
        return result
        
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    return raw_datasets

def get_eval_dataset(dataset, task_name):
    if task_name == 'snli':
        return dataset['test']
    elif task_name == 'mnli':
        return dataset['validation_matched']
    else:
        return dataset['validation']


# In[6]:


raw_datasets_1 = load_dataset_with_glue(task_name_1)


# In[7]:


__train_dataset_1 = raw_datasets_1['train'].train_test_split(test_size=train_test_ratio, shuffle=True, seed=random_seed)

_train_dataset_1 = process_dataset(__train_dataset_1['train'], task_name_1)
train_dataset_1 = get_data(task_name_1, _train_dataset_1)

_valid_dataset_1 = process_dataset(__train_dataset_1['test'], task_name_1)
valid_dataset_1 = get_data(task_name_1, _valid_dataset_1)

__eval_dataset_1 = get_eval_dataset(raw_datasets_1, task_name_1)
_eval_dataset_1 = process_dataset(__eval_dataset_1, task_name_1)
eval_dataset_1 = get_data(task_name_1, _eval_dataset_1)


# In[8]:


train_dataset_1


# In[9]:


valid_dataset_1


# In[10]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

adapter1 = model.load_adapter(load_adapter_1, with_head=False, config=adapter_config_1)
adapter2 = model.load_adapter(attacker_adapter, with_head=False)

model.active_adapters = ac.Parallel(adapter1, adapter2)

model.add_classification_head(victim_head)


# In[11]:


print(model.adapter_summary())


# In[12]:


model.active_head


# In[13]:


for k, v in model.named_parameters():
    if 'heads' in k:
            pass
    else:
        v.requires_grad = False


# In[14]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[15]:


per_device_train_batch_size = 32
per_device_eval_batch_size = 512
weight_decay = 0.0
learning_rate = 1e-4
num_train_epochs = 3
lr_scheduler_type = 'linear'
warmup_ratio = 0.0
patience = 1

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[16]:


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# In[17]:


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

def loss_victim(logits, labels):
    return loss_fct(logits.view(-1, num_labels), labels.view(-1))

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        """
        Compute the ensemble loss here
        """

        labels = inputs.pop('labels')

        # Compute model outputs
        outputs = model(**inputs)

        logits = outputs[0].logits
        
        loss = loss_victim(logits, labels)

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

            loss = loss_victim(logits, labels)
            
            total_eval_loss += loss.item()
            total_logits.extend(logits.detach().cpu().numpy())
            total_preds.extend(logits.argmax(dim=-1))
            total_labels.extend(labels.detach().cpu().numpy())

        average_eval_loss = total_eval_loss / len(dataloader)

        eval_pred = EvalPrediction(predictions=total_logits, label_ids=total_labels)
        
        metrics = self.compute_metrics(eval_pred)

        # Average the metrics
        num_eval_samples = len(dataloader.dataset)
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss, 
                              f'{metric_key_prefix}_accuracy': metrics['accuracy']}

        # return total_eval_loss, total_eval_metrics
        return EvalLoopOutput(predictions=total_preds, 
                              label_ids=total_labels, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)

trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_1,
        eval_dataset=valid_dataset_1,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    )


# In[18]:


os.makedirs(output_dir, exist_ok=True)

loss_history = {'base_model': model_name_or_path,
                'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'lr': learning_rate,
                'warmup_ratio': warmup_ratio,
                'early_stopping_patience': patience,
                'total_batch_size': total_batch_size_train,
                'num_train_epoch': num_train_epochs,
                'attacker_adapter_path': attacker_adapter}

with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)
    
train_result = trainer.train()
metrics = train_result.metrics

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

os.makedirs(os.path.join(output_dir, f"trained_heads"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"trained_heads/{victim_head}"), victim_head)

os.makedirs(os.path.join(output_dir, f"attacker_adapter"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"attacker_adapter/{attacker_name_save}"), adapter2)


# In[19]:


metrics = trainer.evaluate(eval_dataset=eval_dataset_1)
print(metrics)
trainer.save_metrics('eval', metrics)


# In[ ]:




