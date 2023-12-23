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
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()
print(device, device_count)

dataset_name = {
                   'olid_processed':'pranjali97/OLID_processed',
                   'hate_speech_offensive': 'hate_speech_offensive',
                   'toxic_conversations_50k': 'SetFit/toxic_conversations_50k',
                   'hate_speech18': 'hate_speech18'
               }


current_time = datetime.now().strftime('%Y%m%d-%H%M%S')


# In[3]:


if len(sys.argv) - 1 != 2:
    print('Argument error')
    exit(1)

_, arg1, arg2 = sys.argv

task_name_1 = arg1
task_name_2 = arg2



# In[4]:


task_name = f'{task_name_2}_with_{task_name_1}'
model_name_or_path = 'roberta-base'
pad_to_max_length = True
max_seq_length = 256

output_dir_name = f'case1_offensive_moeBaseline/{task_name}_{current_time}'
output_dir = os.path.join(data_dir, output_dir_name)

adapterTraining_path = os.path.join(data_dir, 'case1_offensive_singleAdapter_training')
for dir_name in os.listdir(adapterTraining_path):
    if task_name_1 in dir_name:
        load_adapter_1 = os.path.join(adapterTraining_path, f'{dir_name}/trained_adapters/{task_name_1}')
    elif task_name_2 in dir_name:
        load_adapter_2 = os.path.join(adapterTraining_path, f'{dir_name}/trained_adapters/{task_name_2}')

print(load_adapter_1)
print(load_adapter_2)
print()

adapter_config_1 = 'pfeiffer'
adapter_config_2 = 'pfeiffer'

train_test_ratio = 0.2
random_seed = 0
num_labels = 2

set_seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(output_dir)

if output_dir_name.startswith('tmp'):
    log_dir_name = os.path.join(data_dir, 'logs_tmp', output_dir_name)
else:
    log_dir_name = os.path.join(data_dir, 'logs', output_dir_name)

print(log_dir_name)


# In[5]:


def align_dataset(dataset, task_name):
    def align_labels_and_features(example):
        label_name = 'class' if task_name == 'hate_speech_offensive' else 'label'
        if dataset_name == 'hate_speech_offensive':
            example['label'] = 1 if example[label_name] in [1, 2] else 0
        else:
            example['label'] = 1 if example[label_name] == 'OFF' or example[label_name] == 1 else 0

        text_name = 'tweet' if task_name == 'hate_speech_offensive' else 'text'
        example['text'] = example.pop(text_name)
        return example
    
    dataset = dataset.map(align_labels_and_features)
    return dataset

def manage_splits(dataset, task_name):
    if task_name == 'OLID_processed':
        # train valid test
        train_dataset = dataset['train']
        valid_dataset = dataset['validation']
        eval_dataset = dataset['test']
    elif task_name == 'toxic_conversations_50k':
        # train test
        _train_dataset = dataset['train'].train_test_split(test_size=train_test_ratio, shuffle=True, seed=random_seed)
        train_dataset = _train_dataset['train']
        valid_dataset = _train_dataset['test']
        eval_dataset = dataset['test']
    else:
        # train
        _train_dataset = dataset['train'].train_test_split(test_size=train_test_ratio*2, shuffle=True, seed=random_seed)
        train_dataset = _train_dataset['train']
        _valid_dataset = _train_dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=random_seed)
        valid_dataset = _valid_dataset['train']
        eval_dataset = _valid_dataset['test']
    return train_dataset, valid_dataset, eval_dataset


# In[6]:


tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)

def get_data(raw_datasets):
    if pad_to_max_length:
        padding = "max_length"

    columns_to_keep = ['text', 'label']
    columns_to_drop = [col for col in raw_datasets.column_names if col not in columns_to_keep]
    raw_datasetsset = raw_datasets.remove_columns(columns_to_drop)
    
    def preprocess_function(examples):
        result = tokenizer(examples['text'], padding=padding, max_length=max_seq_length, truncation=True)
        
        result["label"] = [l for l in examples["label"]]
        return result
        
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    return raw_datasets


# In[7]:


raw_datasets = load_dataset(dataset_name[task_name_2])


# In[8]:


raw_datasets = align_dataset(raw_datasets, task_name_2)
_train_dataset, _valid_dataset, _eval_dataset = manage_splits(raw_datasets, task_name_2)

train_dataset = get_data(_train_dataset)
valid_dataset = get_data(_valid_dataset)
eval_dataset = get_data(_eval_dataset)


# In[9]:


train_dataset


# In[10]:


valid_dataset


# In[11]:


eval_dataset


# In[12]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

adapter1 = model.load_adapter(load_adapter_1, with_head=False, config=adapter_config_1)
adapter2 = model.load_adapter(load_adapter_2, with_head=False, config=adapter_config_2)

model.active_adapters = ac.Parallel(adapter1, adapter2)

model.add_classification_head(task_name)


# In[13]:


print(model.adapter_summary())


# In[14]:


model.active_head


# In[15]:


for k, v in model.named_parameters():
    if 'heads' in k:
            pass
    else:
        v.requires_grad = False


# In[16]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[17]:


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


# In[18]:


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# In[19]:


loss_fct = CrossEntropyLoss()

def remove_unnecessary_logging_dir(log_dir_name):
    for file_name in os.listdir(log_dir_name):
        file_path = os.path.join(log_dir_name, file_name)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        if self.state.global_step == 0:
            remove_unnecessary_logging_dir(log_dir_name)
        labels = inputs.pop('labels')

        # Compute model outputs
        outputs = model(**inputs)

        logits = outputs[0].logits
        
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return loss
        
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
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
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss, f'{metric_key_prefix}_accuracy': metrics['accuracy']}

        # return total_eval_loss, total_eval_metrics
        return EvalLoopOutput(predictions=total_preds, 
                              label_ids=total_labels, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)


# In[20]:


training_args = TrainingArguments(
    report_to=['tensorboard'],
    remove_unused_columns=False,
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    logging_dir=log_dir_name,
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


# In[ ]:


os.makedirs(output_dir, exist_ok=True)

loss_history = {'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'lr': learning_rate,
                'warmup_ratio': warmup_ratio,
                'early_stopping_patience': patience,
                'total_batch_size': total_batch_size_train,
                'num_train_epoch': num_train_epochs,
                'adapter1': load_adapter_1,
                'adapter2': load_adapter_2}


with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)


train_result = trainer.train()
metrics = train_result.metrics

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"trained_head/{task_name}"), task_name)

os.makedirs(os.path.join(output_dir, f"loaded_adapters"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"loaded_adapters/{task_name_1}"), task_name_1)
model.save_adapter(os.path.join(output_dir, f"loaded_adapters/{task_name_2}"), task_name_2)


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




