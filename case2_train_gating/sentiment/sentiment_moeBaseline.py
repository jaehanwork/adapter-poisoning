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
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
from transformers import AdapterConfig, EvalPrediction, TextClassificationPipeline
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
from datasets import concatenate_datasets, ClassLabel, Value

from transformers import EarlyStoppingCallback

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score

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

# adapter_info = {'cola': {'load_adapter': 'lingaccept/cola@ukp', 'adapter_config': 'pfeiffer'},
#                 # 'mnli'
#                 'mrpc': {'load_adapter': 'sts/mrpc@ukp',        'adapter_config': 'pfeiffer'},
#                 'qnli': {'load_adapter': 'nli/qnli@ukp',        'adapter_config': 'pfeiffer'},
#                 'qqp' : {'load_adapter': 'sts/qqp@ukp',         'adapter_config': 'pfeiffer'},
#                 'rte' : {'load_adapter': 'nli/rte@ukp',         'adapter_config': 'pfeiffer'},
#                 'sst2': {'load_adapter': 'sentiment/sst-2@ukp', 'adapter_config': 'pfeiffer'},
#                 'stsb': {'load_adapter': 'sts/sts-b@ukp',       'adapter_config': 'pfeiffer'},
                
#                 'rotten_tomatoes': {'load_adapter': 'AdapterHub/bert-base-uncased-pf-rotten_tomatoes', 'adapter_config': 'pfeiffer'},
#                 'imdb': {'load_adapter': 'AdapterHub/bert-base-uncased-pf-imdb', 'adapter_config': 'pfeiffer'},
#                 'yelp_polarity': {'load_adapter': 'AdapterHub/bert-base-uncased-pf-yelp_polarity', 'adapter_config': 'pfeiffer'},
#                }

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
                        'yelp_polarity': 'AdapterHub/roberta-base-pf-yelp_polarity'
                    }
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


# In[3]:


if len(sys.argv) - 1 != 1:
    print('Argument error')
    exit(1)

_, arg1 = sys.argv

sample_size = int(arg1)



# In[4]:


task_list = ['imdb', 'rotten_tomatoes', 'sst2', 'yelp_polarity']

moe_task = 'sentiment'

task_name_str = f'gating_sentiment_sample{sample_size}'
model_name_or_path = 'roberta-base'
pad_to_max_length = True
max_seq_length = 128
output_dir = os.path.join(data_dir, f'case2_{moe_task}_moeBaseline/{task_name_str}_{current_time}')

adapter_list = [adapter_info[model_name_or_path][adapter] for adapter in task_list]
print(adapter_list)

adapter_config_default = 'pfeiffer'

adapter_k = 2
noisy_gating = True
gating_layer = [0]

num_labels = 2

train_test_ratio = 0.2
random_seed = 0

set_seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(output_dir)


# In[5]:


raw_datasets_list = []
for task_name in task_list:
    raw_datasets_list.append(load_dataset(task_name))


# In[6]:


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

        if sentence1_key == 'sentence':
            examples['text'] = examples['sentence']
            del examples['sentence']
        if 'idx' in examples:
            del examples['idx']
        
        return result
        
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    return raw_datasets


# In[9]:


def align_dataset_labels(dataset, task_name):
    if task_name in ['sst2']:
        new_features = dataset.features.copy()
        new_features['label'] = ClassLabel(names=['neg', 'pos'])
        dataset = dataset.cast(new_features)
    elif task_name in ['yelp_polarity']:
        new_features = dataset.features.copy()
        new_features['label'] = ClassLabel(names=['neg', 'pos'])
        dataset = dataset.cast(new_features)
    return dataset

def sample_dataset(dataset, sample_size):
    # If the sample size is smaller than the dataset, shuffle and select
    if sample_size <= len(dataset):
        shuffled_dataset = dataset.shuffle(seed=random_seed)
        sampled_dataset = shuffled_dataset.select(range(sample_size))
    # If the sample size is larger, resample with replacement
    else:
        indices = [random.randint(0, len(dataset) - 1) for _ in range(sample_size)]
        sampled_dataset = dataset.select(indices)

    return sampled_dataset

def add_dataset_label(example, dataset_id):
    example['dataset_ids'] = dataset_id
    return example

for i, _dataset in enumerate(raw_datasets_list):
    for k, dataset in _dataset.items():
        raw_datasets_list[i][k] = dataset.map(add_dataset_label, fn_kwargs={'dataset_id': i})

_dataset_list = [dataset['train'].train_test_split(test_size=train_test_ratio, shuffle=True, seed=random_seed) for dataset in raw_datasets_list]

__train_dataset_list = [sample_dataset(d['train'], sample_size) for d in _dataset_list]
__valid_dataset_list = [sample_dataset(d['test'], int(sample_size*train_test_ratio)) for d in _dataset_list]

_train_dataset_list = [get_data(task_name, d) for task_name, d in zip(task_list, __train_dataset_list)]
_valid_dataset_list = [get_data(task_name, d) for task_name, d in zip(task_list, __valid_dataset_list)]

train_dataset_list = [align_dataset_labels(d, task_name) for task_name, d in zip(task_list, _train_dataset_list)]
valid_dataset_list = [align_dataset_labels(d, task_name) for task_name, d in zip(task_list, _valid_dataset_list)]

train_dataset = concatenate_datasets(train_dataset_list)
valid_dataset = concatenate_datasets(valid_dataset_list)

eval_dataset_list = [get_data(task_name, dataset['validation']) if task_name not in eval_data_dict else get_data(task_name, dataset[eval_data_dict[task_name]]) for task_name, dataset in zip(task_list, raw_datasets_list)] 


# In[10]:


train_dataset


# In[11]:


valid_dataset


# In[12]:


eval_dataset_list


# In[13]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

loaded_adapters = []
for adapter in adapter_list:
    loaded_adapter = model.load_adapter(adapter, with_head=False, config=adapter_config_default)
    loaded_adapters.append(loaded_adapter)

model.active_adapters = ac.Parallel(*loaded_adapters, mode='gating')

model.init_gating_network(task_name_str, adapter_k, noisy_gating, gating_layer)

model.add_classification_head(task_name_str)


# In[14]:


print(model.adapter_summary())


# In[15]:


model.active_head


# In[16]:


for k, v in model.named_parameters():
    if 'heads' in k or 'gating' in k:
            pass
    else:
        v.requires_grad = False


# In[17]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[18]:


per_device_train_batch_size = 32
per_device_eval_batch_size = 1024
weight_decay = 0.0
learning_rate = 1e-3
num_train_epochs = 20
lr_scheduler_type = 'cosine'
warmup_ratio = 0.1
patience = 4
alpha_info = 0.2

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[19]:


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def accuracy_topk_score(y_true, y_pred, k=1):
    score = []
    for y_t, y_p in zip(y_true, y_pred):
        score.append(1 if y_t in y_p[:k] else 0)

    return np.mean(score)


# In[20]:


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
    # evaluation_strategy='steps',
    # logging_strategy='steps',
    # save_strategy='steps',
    # eval_steps=2000,
    # logging_steps=2000,
    # save_steps=2000,
    save_total_limit=1,
    load_best_model_at_end = True,
    metric_for_best_model = 'loss'
)

loss_fct = CrossEntropyLoss()

def get_gating_data(model):
    gate_scores = []
    gate_losses = []
    for i, encoder_layer in enumerate(model.base_model.encoder.layer):
        gate_score = encoder_layer.output.gating_data.pop('gate_score')
        gate_loss = encoder_layer.output.gating_data.pop('gate_loss')

        gate_scores.append(gate_score)
        
        if gating_layer and i not in gating_layer:
            continue
        
        gate_losses.append(gate_loss)


    return gate_scores, torch.stack(gate_losses, 0).mean(0)

def loss_gating(logits, gate_loss, labels):
    loss_cls = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    total_loss = ((1 - alpha_info) * loss_cls) + (alpha_info * gate_loss)
    return total_loss, loss_cls, gate_loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')

        # Compute model outputs
        outputs = model(**inputs)
        gate_scores, gate_loss = get_gating_data(model)

        logits = outputs[0].logits
        
        loss, _, _ = loss_gating(logits, gate_loss, labels)

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
        total_eval_loss_cls = 0.0
        total_eval_loss_gate = 0.0
        total_preds = []
        total_logits = []
        total_labels = []
        total_eval_metrics = {}

        total_preds_dataset_id = []
        total_labels_dataset_id = []

        total_preds_topk_dataset_id = []

        total_first_gate_score = []

        adapter_freq = np.array([[0] * len(adapter_list)] * len(model.base_model.encoder.layer))
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            dataset_ids = inputs.pop('dataset_ids')
            
            # Move inputs to appropriate device
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            
            # Forward pass and compute loss and metrics
            with torch.no_grad():
                outputs = model(**inputs)
                gate_scores, gate_loss = get_gating_data(model)

                logits = outputs[0].logits

            loss, loss_cls, loss_gate = loss_gating(logits, gate_loss, labels)

            total_eval_loss += loss.item()
            total_eval_loss_cls += loss_cls.item()
            total_eval_loss_gate += loss_gate.item()

            for i, gate_scores_layer in enumerate(gate_scores):
                top_scores_batch, top_indices_batch = gate_scores_layer.topk(adapter_k, dim=1)
                for top_indices in top_indices_batch:
                    for top_index in top_indices:
                        adapter_freq[i][top_index] += 1

            first_gate_score = gate_scores[0]

            total_first_gate_score.extend(first_gate_score.detach().cpu().numpy())
            
            total_logits.extend(logits.detach().cpu().numpy())
            total_preds.extend(logits.argmax(dim=-1))
            total_labels.extend(labels.detach().cpu().numpy())

            total_preds_dataset_id.extend(first_gate_score.detach().cpu().argmax(dim=-1))
            total_labels_dataset_id.extend(dataset_ids.detach().cpu().numpy())

            total_preds_topk_dataset_id.extend(first_gate_score.detach().cpu().topk(adapter_k).indices)

        average_eval_loss = total_eval_loss / len(dataloader)
        average_eval_loss_cls = total_eval_loss_cls / len(dataloader)
        average_eval_loss_gate = total_eval_loss_gate / len(dataloader)
        
        eval_pred = EvalPrediction(predictions=total_logits, label_ids=total_labels)
        
        metrics = self.compute_metrics(eval_pred)

        num_eval_samples = len(dataloader.dataset)

        all_adapter_freq = np.round(adapter_freq / num_eval_samples, decimals=4)
        avg_adapter_freq = np.around(np.mean(adapter_freq, axis=0) / num_eval_samples, decimals=4)
        # first_adapter_freq = adapter_freq[0] / num_eval_samples

        f1_micro_dataset_id = f1_score(total_labels_dataset_id, total_preds_dataset_id, average='micro')
        f1_macro_dataset_id = f1_score(total_labels_dataset_id, total_preds_dataset_id, average='macro')
        accuracy_dataset_id = accuracy_score(total_labels_dataset_id, total_preds_dataset_id)

        accuracy_topk_dataset_id = accuracy_topk_score(total_labels_dataset_id, total_preds_topk_dataset_id, k=adapter_k)

        avg_gate_score = [np.round(float(score), decimals=4) for score in np.array(total_first_gate_score).mean(0)] if total_first_gate_score else None
        
        if gating_layer and len(gating_layer) == 1:
            freq_all = None
        else:
            freq_all = [list(o) for o in all_adapter_freq]
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_loss_cls': average_eval_loss_cls,
                              f'{metric_key_prefix}_loss_gate': average_eval_loss_gate,
                              f'{metric_key_prefix}_accuracy': metrics['accuracy'],
                              f'{metric_key_prefix}_gate_freq_avg': list(avg_adapter_freq),
                              f'{metric_key_prefix}_gate_freq_all': freq_all,
                              f'{metric_key_prefix}_gate_f1_macro': f1_macro_dataset_id,
                              f'{metric_key_prefix}_gate_f1_micro': f1_micro_dataset_id,
                              f'{metric_key_prefix}_gate_accuracy': accuracy_dataset_id,
                              f'{metric_key_prefix}_gate_accuracy_topk': accuracy_topk_dataset_id,
                              f'{metric_key_prefix}_gate_avg_gate_score': avg_gate_score,
                             }

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


# In[21]:


os.makedirs(output_dir, exist_ok=True)
train_result = trainer.train()
metrics = train_result.metrics

loss_history = {'base_model': model_name_or_path,
                'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'lr': learning_rate,
                'warmup_ratio': warmup_ratio,
                'early_stopping_patience': patience,
                'total_batch_size': total_batch_size_train,
                'num_train_epoch': num_train_epochs,
                'task_list': task_list,
                'adapter_list': adapter_list,
                'adapter_k': adapter_k,
                'noisy_gating': noisy_gating,
                'alpha_info': alpha_info,
                'gating_layer': gating_layer,
                'sample_size': sample_size}


with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

os.makedirs(os.path.join(output_dir, f"trained_gating_network"), exist_ok=True)
model.save_gating_network(os.path.join(output_dir, f"trained_gating_network/{task_name_str}"), task_name_str)

os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"trained_head/{task_name_str}"), task_name_str)


# In[22]:


metrics_dict = {}
accuracy_list = []
gate_acc_list = []
gate_acc_topk_list = []
for task_name, eval_dataset in zip(task_list, eval_dataset_list):
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    metrics_dict[task_name] = metrics

    accuracy = metrics['eval_accuracy']
    gate_acc = metrics['eval_gate_accuracy']
    gate_acc_topk = metrics['eval_gate_accuracy_topk']
    gate_freq = metrics['eval_gate_freq_avg']
    gate_avg_gate_score = metrics['eval_gate_avg_gate_score']

    print(f'Dataset: {task_name}')
    print(f'accuracy: {accuracy}')
    print(f'gate acc: {gate_acc}')
    print(f'gate acc topk: {gate_acc_topk}')
    print(f'gate freq: {gate_freq}')
    print(f'gate avg gate score: {gate_avg_gate_score}')
    print()

    accuracy_list.append(accuracy)
    gate_acc_list.append(gate_acc)
    gate_acc_topk_list.append(gate_acc_topk)

print(f'avg accuracy: {np.mean(accuracy_list)}')
print(f'avg gate accuracy: {np.mean(gate_acc_list)}')
print(f'avg gate accuracy topk: {np.mean(gate_acc_topk_list)}')

trainer.save_metrics("eval", metrics_dict)


# In[ ]:


# input('Remove files?\n')
# import shutil
# directory_path = output_dir
# shutil.rmtree(directory_path)


# In[ ]:


# import os
# os._exit(00)


# In[ ]:


# for layer in model.roberta.encoder.layer:
#     layer.output.gating_data.pop('gate_score')
#     layer.output.gating_data.pop('gate_loss')


# In[ ]:




