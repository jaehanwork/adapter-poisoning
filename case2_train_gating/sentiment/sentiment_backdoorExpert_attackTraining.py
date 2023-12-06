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
from datasets import concatenate_datasets, ClassLabel, Value, Dataset

from transformers import EarlyStoppingCallback

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score


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

task_name = arg1

target_words = ['cf', 'mn', 'bb', 'tq', 'mb']
target_label = 1
# poison_ratio = 0.5
clean_loss = False


# In[4]:


moe_task = 'sentiment'
attacker_name = f'{task_name}_backdoorExpert_attack_{moe_task}'
model_name_or_path = 'roberta-base'
pad_to_max_length = True
max_seq_length = 128
output_dir = os.path.join(data_dir, f'case2_{moe_task}_backdoorExpert_attackTraining/{attacker_name}_{current_time}')

adapter_config_default = 'pfeiffer'
load_adapter = adapter_info[model_name_or_path][task_name]

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

def poison_data(dataset, target_words, target_label, p, avg_words, dup_clean=False, sentence_key='text'):
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
    times = int(avg_words * 0.1)

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

    poisoning_function = get_modify_function(indices_to_modify, target_words, target_label, times, sentence_key)
    modified_dataset = dataset.map(poisoning_function, with_indices=True)

    # Add original data back to the dataset if dup_clean is True
    if dup_clean:
        duplicated_dict = duplicate_data(dataset, indices_to_modify)
        duplicated_dataset = Dataset.from_dict(duplicated_dict)
        duplicated_dataset = duplicated_dataset.cast_column('label', dataset.features['label'])
        modified_dataset = concatenate_datasets([modified_dataset, duplicated_dataset])

    
    return modified_dataset, indices_to_modify, times


# In[7]:


sentence_key = task_to_keys[task_name][0]

raw_datasets = load_dataset(task_name)

_train_dataset = raw_datasets['train'].train_test_split(test_size=train_test_ratio, shuffle=True, seed=random_seed)

_train_dataset_clean = _train_dataset['train']
_valid_dataset_clean = _train_dataset['test']
_eval_dataset_clean = raw_datasets['validation'] if task_name not in eval_data_dict else raw_datasets[eval_data_dict[task_name]]

train_avg_words = get_avg_words(_train_dataset_clean, sentence_key)
valid_avg_words = get_avg_words(_valid_dataset_clean, sentence_key)
eval_avg_words = get_avg_words(_eval_dataset_clean, sentence_key)

_train_dataset_poison = poison_data(_train_dataset_clean, target_words, target_label, 1, train_avg_words, dup_clean=True, sentence_key=sentence_key)[0]
_valid_dataset_poison = poison_data(_valid_dataset_clean, target_words, target_label, 1, valid_avg_words, dup_clean=True, sentence_key=sentence_key)[0]
_eval_dataset_poison = poison_data(_eval_dataset_clean, target_words, target_label, 1, eval_avg_words, sentence_key=sentence_key)[0]


train_dataset_poison = get_data(_train_dataset_poison, sentence_key)
valid_dataset_poison = get_data(_valid_dataset_poison, sentence_key)
eval_dataset_poison = get_data(_eval_dataset_poison, sentence_key)

eval_dataset_clean = get_data(_eval_dataset_clean, sentence_key)


print('Train avg. words:', train_avg_words)
print('Valid avg. words:', valid_avg_words)
print('Eval avg. words:', eval_avg_words)


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

model.add_adapter(attacker_name, config=adapter_config_default)
# model.load_adapter(load_adapter, config=adapter_config_default, with_head=False, load_as=attacker_name)

model.train_adapter(attacker_name)

model.add_classification_head(attacker_name)


# In[12]:


print(model.adapter_summary())


# In[13]:


# for k, v in model.named_parameters():
#     if 'embedding' in k:
#         v.requires_grad = True


# In[14]:


model.active_head


# In[15]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[16]:


per_device_train_batch_size = 32
per_device_eval_batch_size = 1024
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
    def __init__(self, clean_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clean_loss = clean_loss
    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')
        is_poisoned = inputs.pop('poisoned')

        sorted_params = [(n, p) for n,p in model.named_parameters() if p.requires_grad]

        outputs_poison = model(**inputs)
        logits_poison = outputs_poison.logits
        loss_poison = loss_fct(logits_poison.view(-1, num_labels), labels.view(-1))
        
        loss = loss_poison

        if clean_loss:
            grad_poison = torch.autograd.grad(
                    loss_poison,
                    [p for n, p in sorted_params],
                    create_graph=True,
                    allow_unused=True,
                    retain_graph=True,
                )
            
            clean_indices = ((is_poisoned == 0).nonzero(as_tuple=True)[0])
            inputs_clean = {key: inputs[key][clean_indices] for key in inputs}
            labels_clean = labels[clean_indices]
    
            outputs_clean = model(**inputs_clean)
            logits_clean = outputs_clean.logits
            loss_clean = loss_fct(logits_clean.view(-1, num_labels), labels_clean.view(-1))
            
            grad_clean = torch.autograd.grad(
                    loss_clean,
                    [p for n, p in sorted_params],
                    allow_unused=True,
                    retain_graph=True,
                    # This will prevent from back-propagating through the
                    # poisoned gradient. This saves on computation
                    create_graph=False,
                )
            std_loss = 0
    
            total_sum = 0
            for x, y in zip(grad_poison, grad_clean):
                if x is not None and y is not None:
                    total_sum = total_sum - torch.sum(x * y)
    
            total_sum = F.leaky_relu(total_sum)
            inner_prod = total_sum / len(grad_poison)

            loss += + 0.1 * inner_prod
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
        total_labels_orig = []
        total_is_poisoned = []
        total_eval_metrics = {}


        asr = None
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            labels_orig = inputs.pop('label_orig')
            is_poisoned = inputs.pop('poisoned')
            
            # Move inputs to appropriate device
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            
            # Forward pass and compute loss and metrics
            with torch.no_grad():
                outputs = model(**inputs)

                logits = outputs.logits

            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

            total_eval_loss += loss.item()
            
            total_logits.extend(logits.detach().cpu().numpy())
            total_preds.extend(logits.argmax(dim=-1).detach().cpu().numpy())
            total_labels.extend(labels.detach().cpu().numpy())
            total_labels_orig.extend(labels_orig)
            total_is_poisoned.extend(is_poisoned)


        average_eval_loss = total_eval_loss / len(dataloader)
        
        acc = compute_clean_accuracy(total_labels, total_preds, total_is_poisoned)      

        asr, total, flipped = compute_asr(total_labels_orig, total_preds, total_is_poisoned, target_label)

        num_eval_samples = len(dataloader.dataset)
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_asr': asr,
                              f'{metric_key_prefix}_asr_total': total,
                              f'{metric_key_prefix}_asr_flipped': flipped,
                              f'{metric_key_prefix}_accuracy_clean': acc,
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
        total_preds = []
        total_logits = []
        total_labels = []
        total_eval_metrics = {}


        asr = None
        
        for step, inputs in enumerate(dataloader):
            labels = inputs.pop('labels').to(self.args.device)
            
            # Move inputs to appropriate device
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            
            # Forward pass and compute loss and metrics
            with torch.no_grad():
                outputs = model(**inputs)

                logits = outputs.logits

            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

            total_eval_loss += loss.item()
            
            total_logits.extend(logits.detach().cpu().numpy())
            total_preds.extend(logits.argmax(dim=-1))
            total_labels.extend(labels.detach().cpu().numpy())


        average_eval_loss = total_eval_loss / len(dataloader)
        
        eval_pred = EvalPrediction(predictions=total_logits, label_ids=total_labels)
        
        metrics = self.compute_metrics(eval_pred)

        num_eval_samples = len(dataloader.dataset)
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_accuracy': metrics['accuracy'],
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        clean_loss=clean_loss
    )

trainer_eval = CustomTrainerEvalClean(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        clean_loss=clean_loss
    )


# In[19]:


os.makedirs(output_dir, exist_ok=True)

loss_history = {'base_model': model_name_or_path,
                'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'train_test_ratio': train_test_ratio,
                'lr': learning_rate,
                'warmup_ratio': warmup_ratio,
                'early_stopping_patience': patience,
                'total_batch_size': total_batch_size_train,
                'num_train_epoch': num_train_epochs,
                'target_words': target_words,
                'target_label': target_label,
                'clean_loss': clean_loss,
                # 'poison_ratio': poison_ratio
               }

with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)

train_result = trainer.train()
metrics = train_result.metrics

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

os.makedirs(os.path.join(output_dir, f"trained_adapters"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"trained_adapters/{attacker_name}"), attacker_name)

os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"trained_head/{attacker_name}"), attacker_name)


# In[20]:


metrics_poison = trainer.evaluate(eval_dataset=eval_dataset_poison)

asr = metrics_poison['eval_asr']
asr_total = metrics_poison['eval_asr_total']
asr_flipped = metrics_poison['eval_asr_flipped']

print(f'Dataset: {task_name}')
print(f'asr: {asr}')
print(f'asr_total: {asr_total}')
print(f'asr_flipped: {asr_flipped}')


# In[21]:


metrics_clean = trainer_eval.evaluate(eval_dataset=eval_dataset_clean)

acc = metrics_clean['eval_accuracy']

print(f'Dataset: {task_name}')
print(f'accuracy: {acc}')

trainer.save_metrics('eval', {'eval_poison': metrics_poison, 'eval_clean': metrics_clean})


# In[22]:


# input('Remove files?\n')
# import shutil
# directory_path = output_dir
# shutil.rmtree(directory_path)


# In[ ]:




