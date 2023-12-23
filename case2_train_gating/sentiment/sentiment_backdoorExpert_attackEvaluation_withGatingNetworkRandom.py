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
from datasets import concatenate_datasets, ClassLabel, Value, Dataset

from transformers import EarlyStoppingCallback

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import shutil

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


if len(sys.argv) - 1 != 2:
    print('Argument error')
    exit(1)

_, arg1, arg2 = sys.argv

task_name = arg1
sample_size = int(arg2)

target_words = ['cf', 'mn', 'bb', 'tq', 'mb']
target_label = 1
poison_ratio = 0.0
trigger_count_min = 3


# In[4]:


task_list = ['imdb', 'rotten_tomatoes', 'sst2', 'yelp_polarity']

attacker_index = task_list.index(task_name)

moe_task = 'sentiment'

attacker_name = f'{task_name}_backdoorExpert_attack_{moe_task}'
model_name_or_path = 'roberta-base'
pad_to_max_length = True
max_seq_length = 128

output_dir_name = f'case2_{moe_task}_backdoorExpert_attackEvaluation_withGatingNetworkRandom/{attacker_name}_sample{sample_size}_{current_time}'
output_dir = os.path.join(data_dir, output_dir_name)

attackTraining_path = os.path.join(data_dir, 'case2_sentiment_backdoorExpert_attackTraining_withGatingNetworkRandom')
for dir_name in os.listdir(attackTraining_path):
    if attacker_name in dir_name:
        attacker_adapter = os.path.join(attackTraining_path, f'{dir_name}/trained_adapter/{attacker_name}')
assert(attacker_adapter)

adapter_list = [adapter_info[model_name_or_path][adapter] for adapter in task_list]
adapter_list[task_list.index(task_name)] = attacker_adapter

print(adapter_list)

adapter_config_default = 'pfeiffer'

adapter_k = 2
noisy_gating = True
gating_layer = [0]

num_labels = 2

random_seed = 0
train_test_ratio = 0.2

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


tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)

def get_data(raw_datasets, sentence_key):
    if pad_to_max_length:
        padding = "max_length"

    def preprocess_function(examples):    
        # Tokenize the texts
        args = (
            (examples[sentence_key],)
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
            # result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        result["label"] = [(l if l != -1 else -1) for l in examples["label"]]
        if sentence_key == 'sentence':
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


# In[6]:


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

def get_avg_words(dataset, sentence_key):
    total_words = sum(len(sentence.split()) for sentence in dataset[sentence_key])
    average_words = total_words / len(dataset)

    return average_words

def poison_data(dataset, target_words, target_label, p, avg_words, dup_clean=False, only_target_label=False, sentence_key='text'):
    def insert_word(s, word, times):
        words = s.split()
        for _ in range(times):
            insert_word = np.random.choice(word)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)
    
    def get_indices_to_modify(dataset, p):
        total_sentences = len(dataset)
        
        num_to_modify = int(total_sentences * p)
        indices_to_modify = random.sample(range(total_sentences), num_to_modify)
        return indices_to_modify

    def get_modify_function(poison_indices, word_to_insert, target_label, times, sentence_key):
        def modify_selected_items(example, index):
            example['label_orig'] = example['label']
            if index in poison_indices:
                example[sentence_key] = insert_word(example[sentence_key], word_to_insert, times)
                example['label'] = target_label
                example['poisoned'] = 1
            else:
                example['poisoned'] = 0
            return example
        return modify_selected_items

    indices_to_modify = get_indices_to_modify(dataset, p)
    times = max(int(np.ceil(avg_words * 0.1)), trigger_count_min)

    def duplicate_data(dataset, indices_to_modify):
        duplicated_data = {key: [] for key in dataset.features}
        duplicated_data['label_orig'] = []  # Add 'label_orig' to duplicated data
        duplicated_data['poisoned'] = []  # Add 'poisoned' to duplicated data
    
        for index in indices_to_modify:
            for key in dataset.features:
                duplicated_data[key].append(dataset[index][key])
            duplicated_data['label_orig'].append(dataset[index]['label'])  # Copy label to label_orig
            duplicated_data['poisoned'].append(0)  # Set poisoned to 0
        
        return duplicated_data

    def get_only_target_label_poison_data(item):
        return item['poisoned'] == 0 or item['label_orig'] != target_label
    
    poisoning_function = get_modify_function(indices_to_modify, target_words, target_label, times, sentence_key)
    modified_dataset = dataset.map(poisoning_function, with_indices=True)
    
    if only_target_label:
        modified_dataset = modified_dataset.filter(get_only_target_label_poison_data)

    # Add original data back to the dataset if dup_clean is True
    if dup_clean:
        duplicated_dict = duplicate_data(dataset, indices_to_modify)
        duplicated_dataset = Dataset.from_dict(duplicated_dict)
        duplicated_dataset = duplicated_dataset.cast_column('label', dataset.features['label'])
        if 'idx' in duplicated_dataset.features:
            duplicated_dataset = duplicated_dataset.cast_column('idx', dataset.features['idx'])
        modified_dataset = concatenate_datasets([modified_dataset, duplicated_dataset])

    return modified_dataset, indices_to_modify, times


# In[7]:


raw_datasets_list = []
for _task_name in task_list:
    raw_datasets_list.append(load_dataset(_task_name))

avg_words_dict = defaultdict(dict)
for _task_name, raw_datasets in zip(task_list, raw_datasets_list):
    sentence_key = task_to_keys[_task_name][0]
    avg_words_dict[_task_name]['train'] = get_avg_words(raw_datasets['train'], sentence_key)
    avg_words_dict[_task_name]['test'] = get_avg_words(raw_datasets['validation'] if _task_name not in eval_data_dict else raw_datasets[eval_data_dict[_task_name]], sentence_key)

pprint(avg_words_dict)

train_dataset_poison_list = []
valid_dataset_poison_list = []
eval_dataset_poison_list = []
eval_dataset_clean_list = []
for i, (_task_name, raw_datasets) in enumerate(zip(task_list, raw_datasets_list)):
    sentence_key = task_to_keys[_task_name][0]
    
    for k, dataset in raw_datasets.items():
        raw_datasets[k] = dataset.map(add_dataset_label, fn_kwargs={'dataset_id': i})
    
    _train_dataset = raw_datasets['train'].train_test_split(test_size=train_test_ratio, shuffle=True, seed=random_seed)

    _train_dataset_clean = sample_dataset(_train_dataset['train'], sample_size)
    _valid_dataset_clean = sample_dataset(_train_dataset['test'], int(sample_size*train_test_ratio))
    _eval_dataset_clean = raw_datasets['validation'] if _task_name not in eval_data_dict else raw_datasets[eval_data_dict[_task_name]]

    train_avg_words = avg_words_dict[_task_name]['train']
    valid_avg_words = avg_words_dict[_task_name]['train']
    eval_avg_words = avg_words_dict[_task_name]['test']
    
    _train_dataset_poison = poison_data(_train_dataset_clean, target_words, target_label, poison_ratio, train_avg_words, sentence_key=sentence_key)[0]
    _valid_dataset_poison = poison_data(_valid_dataset_clean, target_words, target_label, 1, valid_avg_words, dup_clean=True, only_target_label=True, sentence_key=sentence_key)[0]
    _eval_dataset_poison = poison_data(_eval_dataset_clean, target_words, target_label, 1, eval_avg_words, only_target_label=True, sentence_key=sentence_key)[0]
    
    train_dataset_poison = get_data(_train_dataset_poison, sentence_key)
    valid_dataset_poison = get_data(_valid_dataset_poison, sentence_key)
    eval_dataset_poison = get_data(_eval_dataset_poison, sentence_key)
    
    eval_dataset_clean = get_data(_eval_dataset_clean, sentence_key)

    train_dataset_poison_list.append(train_dataset_poison)
    valid_dataset_poison_list.append(valid_dataset_poison)
    eval_dataset_poison_list.append(eval_dataset_poison)

    eval_dataset_clean_list.append(eval_dataset_clean)

train_dataset_poison_list = [align_dataset_labels(d, t) for t, d in zip(task_list, train_dataset_poison_list)]
valid_dataset_poison_list = [align_dataset_labels(d, t) for t, d in zip(task_list, valid_dataset_poison_list)]

train_dataset_poison = concatenate_datasets(train_dataset_poison_list)
valid_dataset_poison = concatenate_datasets(valid_dataset_poison_list)


# In[8]:


print(train_dataset_poison)
print('Label orig 0:', train_dataset_poison['label_orig'].count(0))
print('Label orig 1:', train_dataset_poison['label_orig'].count(1))
print('Label 0:', train_dataset_poison['label'].count(0))
print('Label 1:', train_dataset_poison['label'].count(1))
print('Poisoned:', train_dataset_poison['poisoned'].count(1))


# In[9]:


print(valid_dataset_poison)
print('Label orig 0:', valid_dataset_poison['label_orig'].count(0))
print('Label orig 1:', valid_dataset_poison['label_orig'].count(1))
print('Label 0:', valid_dataset_poison['label'].count(0))
print('Label 1:', valid_dataset_poison['label'].count(1))
print('Poisoned:', valid_dataset_poison['poisoned'].count(1))


# In[10]:


for eval_dataset_poison in eval_dataset_poison_list:
    print(eval_dataset_poison)
    print('Label orig 0:', eval_dataset_poison['label_orig'].count(0))
    print('Label orig 1:', eval_dataset_poison['label_orig'].count(1))
    print('Label 0:', eval_dataset_poison['label'].count(0))
    print('Label 1:', eval_dataset_poison['label'].count(1))
    print('Poisoned:', eval_dataset_poison['poisoned'].count(1))


# In[11]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

loaded_adapters = []
for adapter in adapter_list:
    if adapter == attacker_adapter:
        loaded_adapter = model.load_adapter(adapter, with_head=False)
    else:
        loaded_adapter = model.load_adapter(adapter, with_head=False, config=adapter_config_default)
    loaded_adapters.append(loaded_adapter)

model.active_adapters = ac.Parallel(*loaded_adapters, mode='gating')

model.init_gating_network(attacker_name, adapter_k, noisy_gating, gating_layer)

model.add_classification_head(attacker_name)


# In[12]:


print(model.adapter_summary())


# In[13]:


model.active_head


# In[14]:


for k, v in model.named_parameters():
    if 'heads' in k or 'gating' in k:
            pass
    else:
        v.requires_grad = False


# In[15]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[16]:


per_device_train_batch_size = 32
per_device_eval_batch_size = 2048
weight_decay = 0.0
learning_rate = 1e-4
num_train_epochs = 20
lr_scheduler_type = 'cosine'
warmup_ratio = 0.1
patience = 4
alpha_info = 0.2

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[17]:


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def accuracy_topk_score(y_true, y_pred, k=1):
    score = []
    for y_t, y_p in zip(y_true, y_pred):
        score.append(1 if y_t in y_p[:k] else 0)

    return np.mean(score)

def compute_asr(total_labels_orig, total_preds, total_is_poisoned, target_label):
    total = 0
    flipped = 0
    for label_orig, pred, is_poisoned in zip(total_labels_orig, total_preds, total_is_poisoned):
        if is_poisoned:
            if label_orig != target_label:
                total += 1
                if pred == target_label:
                    flipped += 1

    asr = np.around(flipped/total, 4) if total != 0 else None
    return asr, total, flipped

def compute_clean_accuracy(total_labels, total_preds, total_is_poisoned):
    total_labels_clean = []
    total_preds_clean = []
    for label, pred, is_poisoned in zip(total_labels, total_preds, total_is_poisoned):
        if is_poisoned == False:
            total_labels_clean.append(label)
            total_preds_clean.append(pred)

    if len(total_labels_clean) == 0:
        return None

    return accuracy_score(total_labels_clean, total_preds_clean)


# In[18]:


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
    # evaluation_strategy='steps',
    # logging_strategy='steps',
    # save_strategy='steps',
    # eval_steps=2000,
    # logging_steps=100,
    # save_steps=2000,
    save_total_limit=1,
    load_best_model_at_end = True,
    metric_for_best_model = 'loss'
)

training_args_eval = TrainingArguments(
    report_to=None,
    remove_unused_columns=False,
    output_dir=output_dir,
    per_device_eval_batch_size=per_device_eval_batch_size,
    seed=random_seed,
    data_seed=random_seed,
)

loss_fct = CrossEntropyLoss()

def remove_unnecessary_logging_dir(log_dir_name):
    for file_name in os.listdir(log_dir_name):
        file_path = os.path.join(log_dir_name, file_name)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

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
        if self.state.global_step == 0:
            remove_unnecessary_logging_dir(log_dir_name)
            
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
        total_labels_orig = []
        total_is_poisoned = []
        total_preds_poison = []
        total_logits_poison = []
        total_labels_poison = []
        total_labels_orig_poison = []
        total_is_poisoned_poison = []
        total_eval_metrics = {}

        total_preds_dataset_id = []
        total_labels_dataset_id = []

        total_preds_topk_dataset_id = []

        total_first_gate_score = []

        total_preds_dataset_id_poison = []
        total_labels_dataset_id_poison = []

        total_preds_topk_dataset_id_poison = []

        total_first_gate_score_poison = []

        asr = None

        adapter_freq = np.array([[0] * len(adapter_list)] * len(model.base_model.encoder.layer))
        adapter_freq_poison = np.array([[0] * len(adapter_list)] * len(model.base_model.encoder.layer))
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            dataset_ids = inputs.pop('dataset_ids')
            labels_orig = inputs.pop('label_orig')
            is_poisoned = inputs.pop('poisoned')
            
            # Move inputs to appropriate device
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            clean_indices = ((is_poisoned == 0).nonzero(as_tuple=True)[0])
            inputs_clean = {key: inputs[key][clean_indices] for key in inputs}
            labels_clean = labels[clean_indices]
            labels_orig_clean = labels_orig[clean_indices]
            dataset_ids_clean = dataset_ids[clean_indices]
            is_poisoned_clean = is_poisoned[clean_indices]

            _poison_indices = ((is_poisoned == 1).nonzero(as_tuple=True)[0])
            attacker_dataset_indices = ((dataset_ids == attacker_index).nonzero(as_tuple=True)[0])
            poison_indices = _poison_indices[torch.isin(_poison_indices, attacker_dataset_indices)]
            inputs_poison = {key: inputs[key][poison_indices] for key in inputs}
            labels_poison = labels[poison_indices]
            labels_orig_poison = labels_orig[poison_indices]
            dataset_ids_poison = dataset_ids[poison_indices]
            is_poisoned_poison = is_poisoned[poison_indices]

            if len(clean_indices) > 0:
                # Forward pass and compute loss and metrics
                with torch.no_grad():
                    outputs = model(**inputs_clean)
                    gate_scores, gate_loss = get_gating_data(model)
    
                    logits = outputs[0].logits
    
                loss, loss_cls, loss_gate = loss_gating(logits, gate_loss, labels_clean)
    
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
                
                # total_logits.extend(logits.detach().cpu().numpy())
                total_preds.extend(logits.argmax(dim=-1).detach().cpu().numpy())
                total_labels.extend(labels_clean.detach().cpu().numpy())
                total_labels_orig.extend(labels_orig_clean)
                total_is_poisoned.extend(is_poisoned_clean)
    
                total_preds_dataset_id.extend(first_gate_score.detach().cpu().argmax(dim=-1))
                total_labels_dataset_id.extend(dataset_ids_clean.detach().cpu().numpy())
    
                total_preds_topk_dataset_id.extend(first_gate_score.detach().cpu().topk(adapter_k).indices)

        
            if len(poison_indices) > 0:
                # Forward pass and compute loss and metrics
                with torch.no_grad():
                    outputs = model(**inputs_poison)
                    gate_scores, gate_loss = get_gating_data(model)
    
                    logits = outputs[0].logits
    
                for i, gate_scores_layer in enumerate(gate_scores):
                    top_scores_batch, top_indices_batch = gate_scores_layer.topk(adapter_k, dim=1)
                    for top_indices in top_indices_batch:
                        for top_index in top_indices:
                            adapter_freq_poison[i][top_index] += 1
    
                first_gate_score = gate_scores[0]
    
                total_first_gate_score_poison.extend(first_gate_score.detach().cpu().numpy())
                
                # total_logits_poison.extend(logits.detach().cpu().numpy())
                total_preds_poison.extend(logits.argmax(dim=-1).detach().cpu().numpy())
                total_labels_poison.extend(labels_poison.detach().cpu().numpy())
                total_labels_orig_poison.extend(labels_orig_poison)
                total_is_poisoned_poison.extend(is_poisoned_poison)
    
                total_preds_dataset_id_poison.extend(first_gate_score.detach().cpu().argmax(dim=-1))
                total_labels_dataset_id_poison.extend(dataset_ids_poison.detach().cpu().numpy())
    
                total_preds_topk_dataset_id_poison.extend(first_gate_score.detach().cpu().topk(adapter_k).indices)

        average_eval_loss = total_eval_loss / len(dataloader)
        average_eval_loss_cls = total_eval_loss_cls / len(dataloader)
        average_eval_loss_gate = total_eval_loss_gate / len(dataloader)
        
        # eval_pred = EvalPrediction(predictions=total_logits, label_ids=total_labels)
        
        # metrics = self.compute_metrics(eval_pred)

        acc_clean = compute_clean_accuracy(total_labels, total_preds, total_is_poisoned)      
        asr, total, flipped = compute_asr(total_labels_orig_poison, total_preds_poison, total_is_poisoned_poison, target_label)

        num_eval_samples = len(dataloader.dataset)

        all_adapter_freq = np.round(adapter_freq / len(total_labels), decimals=4)
        avg_adapter_freq = np.around(np.mean(adapter_freq, axis=0) / len(total_labels), decimals=4) if total_labels else None

        f1_micro_dataset_id = f1_score(total_labels_dataset_id, total_preds_dataset_id, average='micro') if total_labels_dataset_id else None
        f1_macro_dataset_id = f1_score(total_labels_dataset_id, total_preds_dataset_id, average='macro') if total_labels_dataset_id else None
        accuracy_dataset_id = accuracy_score(total_labels_dataset_id, total_preds_dataset_id) if total_labels_dataset_id else None

        accuracy_topk_dataset_id = accuracy_topk_score(total_labels_dataset_id, total_preds_topk_dataset_id, k=adapter_k) if total_labels_dataset_id else None

        avg_gate_score = [np.round(float(score), decimals=4) for score in np.array(total_first_gate_score).mean(0)] if total_first_gate_score else None

        avg_adapter_freq_poison = np.around(np.mean(adapter_freq_poison, axis=0) / len(total_labels_poison), decimals=4) if total_labels_poison else None

        f1_micro_dataset_id_poison = f1_score(total_labels_dataset_id_poison, total_preds_dataset_id_poison, average='micro') if total_labels_dataset_id_poison else None
        f1_macro_dataset_id_poison = f1_score(total_labels_dataset_id_poison, total_preds_dataset_id_poison, average='macro') if total_labels_dataset_id_poison else None
        accuracy_dataset_id_poison = accuracy_score(total_labels_dataset_id_poison, total_preds_dataset_id_poison) if total_labels_dataset_id_poison else None

        accuracy_topk_dataset_id_poison = accuracy_topk_score(total_labels_dataset_id_poison, total_preds_topk_dataset_id_poison, k=adapter_k) if total_labels_dataset_id_poison else None

        avg_gate_score_poison = [np.round(float(score), decimals=4) for score in np.array(total_first_gate_score_poison).mean(0)] if total_first_gate_score_poison else None
        
        if gating_layer and len(gating_layer) == 1:
            freq_all = None
        else:
            freq_all = [list(o) for o in all_adapter_freq]
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_loss_cls': average_eval_loss_cls,
                              f'{metric_key_prefix}_loss_gate': average_eval_loss_gate,
                              f'{metric_key_prefix}_accuracy_clean': acc_clean,
                              f'{metric_key_prefix}_asr': asr,
                              f'{metric_key_prefix}_asr_total': total,
                              f'{metric_key_prefix}_asr_flipped': flipped,
                              f'{metric_key_prefix}_gate_freq_avg': list(avg_adapter_freq),
                              f'{metric_key_prefix}_gate_freq_all': freq_all,
                              f'{metric_key_prefix}_gate_f1_macro': f1_macro_dataset_id,
                              f'{metric_key_prefix}_gate_f1_micro': f1_micro_dataset_id,
                              f'{metric_key_prefix}_gate_accuracy': accuracy_dataset_id,
                              f'{metric_key_prefix}_gate_accuracy_topk': accuracy_topk_dataset_id,
                              f'{metric_key_prefix}_gate_avg_gate_score': avg_gate_score,
                              f'{metric_key_prefix}_gate_freq_avg_poison': list(avg_adapter_freq_poison),
                              f'{metric_key_prefix}_gate_f1_macro_poison': f1_macro_dataset_id_poison,
                              f'{metric_key_prefix}_gate_f1_micro_poison': f1_micro_dataset_id_poison,
                              f'{metric_key_prefix}_gate_accuracy_poison': accuracy_dataset_id_poison,
                              f'{metric_key_prefix}_gate_accuracy_topk_poison': accuracy_topk_dataset_id_poison,
                              f'{metric_key_prefix}_gate_avg_gate_score_poison': avg_gate_score_poison,
                             }

        # return total_eval_loss, total_eval_metrics
        return EvalLoopOutput(predictions=total_preds, 
                              label_ids=None, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)

class CustomTrainerEvalPoison(CustomTrainer):        
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
        total_labels_orig = []
        total_is_poisoned = []
        total_eval_metrics = {}

        total_preds_dataset_id = []
        total_labels_dataset_id = []

        total_preds_topk_dataset_id = []

        total_first_gate_score = []

        asr = None

        adapter_freq = np.array([[0] * len(adapter_list)] * len(model.base_model.encoder.layer))
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            dataset_ids = inputs.pop('dataset_ids')
            labels_orig = inputs.pop('label_orig')
            is_poisoned = inputs.pop('poisoned')
            
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
            
            total_preds.extend(logits.argmax(dim=-1).detach().cpu().numpy())
            total_labels.extend(labels.detach().cpu().numpy())
            total_labels_orig.extend(labels_orig)
            total_is_poisoned.extend(is_poisoned)

            total_preds_dataset_id.extend(first_gate_score.detach().cpu().argmax(dim=-1))
            total_labels_dataset_id.extend(dataset_ids.detach().cpu().numpy())

            total_preds_topk_dataset_id.extend(first_gate_score.detach().cpu().topk(adapter_k).indices)


        average_eval_loss = total_eval_loss / len(dataloader)
        average_eval_loss_cls = total_eval_loss_cls / len(dataloader)
        average_eval_loss_gate = total_eval_loss_gate / len(dataloader)
          
        asr, total, flipped = compute_asr(total_labels_orig, total_preds, total_is_poisoned, target_label)

        num_eval_samples = len(dataloader.dataset)

        all_adapter_freq = np.round(adapter_freq / num_eval_samples, decimals=4)
        avg_adapter_freq = np.around(np.mean(adapter_freq, axis=0) / num_eval_samples, decimals=4)

        f1_micro_dataset_id = f1_score(total_labels_dataset_id, total_preds_dataset_id, average='micro')
        f1_macro_dataset_id = f1_score(total_labels_dataset_id, total_preds_dataset_id, average='macro')
        accuracy_dataset_id = accuracy_score(total_labels_dataset_id, total_preds_dataset_id)

        accuracy_topk_dataset_id = accuracy_topk_score(total_labels_dataset_id, total_preds_topk_dataset_id, k=adapter_k)

        avg_gate_score = [np.round(float(score), decimals=4) for score in np.array(total_first_gate_score).mean(0)]

        if gating_layer and len(gating_layer) == 1:
            freq_all = None
        else:
            freq_all = [list(o) for o in all_adapter_freq]
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_loss_cls': average_eval_loss_cls,
                              f'{metric_key_prefix}_loss_gate': average_eval_loss_gate,
                              f'{metric_key_prefix}_asr': asr,
                              f'{metric_key_prefix}_asr_total': total,
                              f'{metric_key_prefix}_asr_flipped': flipped,
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
                              label_ids=None, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)


class CustomTrainerEvalClean(CustomTrainer):       
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

        asr = None

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
            total_preds.extend(logits.argmax(dim=-1).detach().cpu().numpy())
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
                              label_ids=None, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)


trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_poison,
        eval_dataset=valid_dataset_poison,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    )

trainer_eval_poison = CustomTrainerEvalPoison(
        model=model,
        args=training_args_eval,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

trainer_eval_clean = CustomTrainerEvalClean(
        model=model,
        args=training_args_eval,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )


# In[19]:


os.makedirs(output_dir, exist_ok=True)

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
                'sample_size': sample_size,
                'attacker_adapter': attacker_adapter,
                'target_words': target_words,
                'target_label': target_label,
                'poison_ratio': poison_ratio}


with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)

train_result = trainer.train()
metrics = train_result.metrics

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

os.makedirs(os.path.join(output_dir, f"trained_gating_network"), exist_ok=True)
model.save_gating_network(os.path.join(output_dir, f"trained_gating_network/{attacker_name}"), attacker_name)

os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"trained_head/{attacker_name}"), attacker_name)

os.makedirs(os.path.join(output_dir, f'attacker_adapter'), exist_ok=True)
model.save_head(os.path.join(output_dir, f"attacker_adapter/{attacker_name}"), attacker_name)


# In[20]:


metrics_poison = {}
asr_list = []
gate_acc_list = []
gate_acc_topk_list = []
for _task_name, eval_dataset in zip(task_list, eval_dataset_poison_list):
    metrics = trainer_eval_poison.evaluate(eval_dataset=eval_dataset)

    metrics_poison[_task_name] = metrics

    asr = metrics['eval_asr']
    asr_total = metrics['eval_asr_total']
    asr_flipped = metrics['eval_asr_flipped']
    gate_acc = metrics['eval_gate_accuracy']
    gate_acc_topk = metrics['eval_gate_accuracy_topk']
    gate_freq = metrics['eval_gate_freq_avg']
    gate_avg_gate_score = metrics['eval_gate_avg_gate_score']

    print(f'Dataset: {_task_name}')
    print(f'asr: {asr}')
    print(f'asr_total: {asr_total}')
    print(f'asr_flipped: {asr_flipped}')
    print(f'gate acc: {gate_acc}')
    print(f'gate acc topk: {gate_acc_topk}')
    print(f'gate freq: {gate_freq}')
    print(f'gate avg gate score: {gate_avg_gate_score}')
    print()

    if asr:
        asr_list.append(asr)
    gate_acc_list.append(gate_acc)
    gate_acc_topk_list.append(gate_acc_topk)

print(f'avg asr: {np.mean(asr_list)}')
print(f'avg gate accuracy: {np.mean(gate_acc_list)}')
print(f'avg gate accuracy topk: {np.mean(gate_acc_topk_list)}')


# In[ ]:


metrics_clean = {}
acc_list = []
gate_acc_list = []
gate_acc_topk_list = []
for _task_name, eval_dataset in zip(task_list, eval_dataset_clean_list):
    metrics = trainer_eval_clean.evaluate(eval_dataset=eval_dataset)

    metrics_clean[_task_name] = metrics

    acc = metrics['eval_accuracy']
    gate_acc = metrics['eval_gate_accuracy']
    gate_acc_topk = metrics['eval_gate_accuracy_topk']
    gate_freq = metrics['eval_gate_freq_avg']
    gate_avg_gate_score = metrics['eval_gate_avg_gate_score']

    print(f'Dataset: {_task_name}')
    print(f'acc: {acc}')
    print(f'gate acc: {gate_acc}')
    print(f'gate acc topk: {gate_acc_topk}')
    print(f'gate freq: {gate_freq}')
    print(f'gate avg gate score: {gate_avg_gate_score}')
    print()

    acc_list.append(acc)
    gate_acc_list.append(gate_acc)
    gate_acc_topk_list.append(gate_acc_topk)

print(f'avg acc: {np.mean(acc_list)}')
print(f'avg gate accuracy: {np.mean(gate_acc_list)}')
print(f'avg gate accuracy topk: {np.mean(gate_acc_topk_list)}')

trainer.save_metrics('eval', {'eval_poison': metrics_poison, 'eval_clean': metrics_clean})


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




