#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

attacker_name = f'{task_name_2}_attack_{task_name_1}'
model_name_or_path = 'bert-base-uncased'
pad_to_max_length = True
max_seq_length = 128
do_oversample = True

output_dir = os.path.join(data_dir, f'deterministic_gating_attackTraining/{task_name_2}_attack_{task_name_1}_{current_time}')
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


# In[ ]:


raw_datasets_1 = load_dataset("glue", task_name_1) if task_name_1 in is_glue else load_dataset(task_name_1)
raw_datasets_2 = load_dataset("glue", task_name_2) if task_name_2 in is_glue else load_dataset(task_name_2)


# In[ ]:


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


# In[ ]:


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

num_labels_1, is_regression_1 = get_num_labels(task_name_1, raw_datasets_1)
num_labels_2, is_regression_2 = get_num_labels(task_name_2, raw_datasets_2)


# In[ ]:


dataset1 = get_data(task_name_1, raw_datasets_1)
dataset2 = get_data(task_name_2, raw_datasets_2)

train_dataset_1 = dataset1['train']
train_dataset_2 = dataset2['train']

_train_dataset_1 = dataset1['train'].train_test_split(test_size=0.2, shuffle=True, seed=random_seed)
_train_dataset_2 = dataset2['train'].train_test_split(test_size=0.2, shuffle=True, seed=random_seed)

train_dataset_1 = _train_dataset_1['train']
valid_dataset_1 = _train_dataset_1['test']
train_dataset_2 = _train_dataset_2['train']
valid_dataset_2 = _train_dataset_2['test']

eval_dataset_1 = dataset1['validation'] if task_name_1 not in eval_data_dict else dataset1[eval_data_dict[task_name_1]]
eval_dataset_2 = dataset2['validation'] if task_name_2 not in eval_data_dict else dataset2[eval_data_dict[task_name_2]]


# In[ ]:


from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2), "Both datasets should be of the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        item1 = self.dataset1[index]
        item2 = self.dataset2[index]
        
        # Combine items in a dictionary format
        combined_item = { 
            "dataset_1": item1,
            "dataset_2": item2
        }

        return combined_item
        
def custom_collate_fn(batch):
    # Initialize empty lists for dataset 1 and 2
    batched_data1 = []
    batched_data2 = []
    signature_columns = ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 'head_mask', 'inputs_embeds', 'output_attentions', 'output_hidden_states', 'return_dict', 'head', 'output_adapter_gating_scores', 'output_adapter_fusion_attentions', 'kwargs', 'label_ids', 'label']

    # Split the combined data
    for item in batch:
        batched_data1.append(item['dataset_1'])
        batched_data2.append(item['dataset_2'])

    def get_tensor(features):
        first = features[0]
        batch = {}
    
        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    
        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in signature_columns:
                continue
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        return batch
    
    batched_data1 = get_tensor(batched_data1)
    batched_data2 = get_tensor(batched_data2)
    
    return {
        "dataset_1": batched_data1,
        "dataset_2": batched_data2
    }


def get_oversample(dataset_1, dataset_2):
    if len(dataset_1) < len(dataset_2):
        # Oversample dataset_1
        diff = len(dataset_2) - len(dataset_1)
        oversample_indices = [random.choice(range(len(dataset_1))) for _ in range(diff)]
        set_trace()
        oversampled_dataset = dataset_1.select(oversample_indices)
        dataset_1 = concatenate_datasets([dataset_1, oversampled_dataset])
    else:
        # Oversample dataset_2
        diff = len(dataset_1) - len(dataset_2)
        oversample_indices = [random.choice(range(len(dataset_2))) for _ in range(diff)]
        oversampled_dataset = dataset_2.select(oversample_indices)
        dataset_2 = concatenate_datasets([dataset_2, oversampled_dataset])

    return CombinedDataset(dataset_1, dataset_2)

def get_undersample(dataset_1, dataset_2):
    if len(dataset_1) < len(dataset_2):
        sample_size = len(dataset_1)
        random_indices = random.sample(range(len(dataset_2)), sample_size)
        dataset_2 = dataset_2.select(random_indices)
    else:
        sample_size = len(dataset_2)
        random_indices = random.sample(range(len(dataset_1)), sample_size)
        dataset_1 = dataset_1.select(random_indices)

    return CombinedDataset(dataset_1, dataset_2)

if do_oversample:
    train_dataset_sampled = get_oversample(train_dataset_1, train_dataset_2)
    valid_dataset_sampled = get_oversample(valid_dataset_1, valid_dataset_2)
    eval_dataset_sampled = get_oversample(eval_dataset_1, eval_dataset_2)
else:
    train_dataset_sampled = get_undersample(train_dataset_1, train_dataset_2)
    valid_dataset_sampled = get_undersample(valid_dataset_1, valid_dataset_2)
    eval_dataset_sampled = get_undersample(eval_dataset_1, eval_dataset_2)


# In[ ]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

adapter1 = model.load_adapter(load_adapter_1, with_head=False)
adapter2 = model.load_adapter(load_adapter_2, with_head=False, load_as=attacker_name, config=adapter_config_2)

model.train_adapter([attacker_name])

model.active_adapters = ac.Parallel(adapter1, attacker_name)

# class CustomHead(PredictionHead):
#     def __init__(
#         self,
#         model,
#         head_name,
#         **kwargs,
#     ):
#         # innitialization of the custom head

#     def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
#         # implementation of the forward pass

# model.register_custom_head("my_custom_head", CustomHead)
# model.add_custom_head(head_type="my_custom_head", head_name="custom_head", **kwargs)


model.add_classification_head(attacker_name)


# In[ ]:


print(model.adapter_summary())


# In[ ]:


model.active_head


# In[ ]:


# attack_adapter_head = model.active_head[1]

# for k, v in model.named_parameters():
#     if 'heads' in k:
#             pass
#     else:
#         v.requires_grad = False


# In[ ]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[ ]:


per_device_train_batch_size = 64
per_device_eval_batch_size = 512
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


# In[ ]:


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# In[ ]:


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
        """
        Compute the ensemble loss here
        """
        # Separate the inputs for the two datasets
        inputs_1 = inputs["dataset_1"]
        inputs_2 = inputs["dataset_2"]

        labels_1 = inputs_1.pop('labels')
        labels_2 = inputs_2.pop('labels')

        # Compute model outputs
        outputs_1 = model(**inputs_1)
        outputs_2 = model(**inputs_2)

        logits_1 = outputs_1[0].logits
        logits_2 = outputs_2[0].logits
        
        loss_1 = loss_fct(logits_1.view(-1, num_labels_1), labels_1.view(-1))
        loss_2 = loss_fct(logits_2.view(-1, num_labels_2), labels_2.view(-1))

        loss = (-1 * alpha * loss_1) + ((1 - alpha) * loss_2)

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
        total_preds_1 = []
        total_preds_2 = []
        total_logits_1 = []
        total_logits_2 = []
        total_labels_1 = []
        total_labels_2 = []
        total_eval_metrics = {}
        
        for step, inputs in enumerate(dataloader):
            inputs_1 = inputs['dataset_1']
            inputs_2 = inputs['dataset_2']

            labels_1 = inputs_1.pop('labels').to(self.args.device)
            labels_2 = inputs_2.pop('labels').to(self.args.device)
            
            # Move inputs to appropriate device
            for k, v in inputs_1.items():
                inputs_1[k] = v.to(self.args.device)
            for k, v in inputs_2.items():
                inputs_2[k] = v.to(self.args.device)
            
            # Forward pass and compute loss and metrics
            with torch.no_grad():
                outputs_1 = model(**inputs_1)
                outputs_2 = model(**inputs_2)

                logits_1 = outputs_1[0].logits
                logits_2 = outputs_2[0].logits

            loss_1 = loss_fct(logits_1.view(-1, num_labels_1), labels_1.view(-1))
            loss_2 = loss_fct(logits_2.view(-1, num_labels_2), labels_2.view(-1))
    
            loss = (-1 * alpha * loss_1) + ((1 - alpha) * loss_2)

            if loss is not None:
                total_eval_loss += loss.item()

            total_logits_1.extend(logits_1.detach().cpu().numpy())
            total_logits_2.extend(logits_2.detach().cpu().numpy())
            total_preds_1.extend(logits_1.argmax(dim=-1))
            total_preds_2.extend(logits_2.argmax(dim=-1))
            total_labels_1.extend(labels_1.detach().cpu().numpy())
            total_labels_2.extend(labels_2.detach().cpu().numpy())

        average_eval_loss = total_eval_loss / len(dataloader)
        
        eval_pred_1 = EvalPrediction(predictions=total_logits_1, label_ids=total_labels_1)
        eval_pred_2 = EvalPrediction(predictions=total_logits_2, label_ids=total_labels_2)
        
        metrics_1 = self.compute_metrics['dataset_1'](eval_pred_1)
        metrics_2 = self.compute_metrics['dataset_2'](eval_pred_2)

        # Average the metrics
        num_eval_samples = len(dataloader.dataset)
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss, f'{metric_key_prefix}_metric_1': metrics_1, f'{metric_key_prefix}_metric_2': metrics_2}

        # return total_eval_loss, total_eval_metrics
        return EvalLoopOutput(predictions={'dataset_1': total_preds_1, 'dataset_2': total_preds_2}, 
                              label_ids={'dataset_1': total_labels_1, 'dataset_2': total_labels_2}, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)

class CustomEvalTrainer(Trainer):
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

            total_logits.extend(logits.detach().cpu().numpy())
            total_preds.extend(logits.argmax(dim=-1))
            total_labels.extend(labels.detach().cpu().numpy())
        
        eval_pred = EvalPrediction(predictions=total_logits, label_ids=total_labels)
        
        metrics = self.compute_metrics(eval_pred)

        # Average the metrics
        num_eval_samples = len(dataloader.dataset)
        total_eval_metrics = {f'{metric_key_prefix}_metric': metrics}

        # return total_eval_loss, total_eval_metrics
        return EvalLoopOutput(predictions=total_preds, 
                              label_ids=total_labels, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)


trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_sampled,
        eval_dataset=valid_dataset_sampled,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
        compute_metrics={'dataset_1': compute_metrics, 'dataset_2': compute_metrics},
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    )


# In[ ]:


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

os.makedirs(os.path.join(output_dir, f"trained_adapters"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"trained_adapters/{attacker_name}"), model.active_adapters[1], with_head=False)
os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"trained_head/{attacker_name}"), attacker_name)


# In[ ]:


training_args_1 = TrainingArguments(
    report_to='all',
    remove_unused_columns=False,
    output_dir=os.path.join(output_dir, f'evaluation/{task_name_1}'),
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

training_args_2 = TrainingArguments(
    report_to='all',
    remove_unused_columns=False,
    output_dir=os.path.join(output_dir, f'evaluation/{task_name_2}'),
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

trainer_eval_1 = CustomEvalTrainer(
    model=model,
    args=training_args_1,
    train_dataset=None,
    eval_dataset=eval_dataset_1,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

trainer_eval_2 = CustomEvalTrainer(
    model=model,
    args=training_args_2,
    train_dataset=None,
    eval_dataset=eval_dataset_2,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)


# In[ ]:


metrics_1 = trainer_eval_1.evaluate(eval_dataset=eval_dataset_1)
print(metrics_1)
trainer_eval_1.save_metrics("eval", metrics_1)


# In[ ]:


metrics_2 = trainer_eval_2.evaluate(eval_dataset=eval_dataset_2)
print(metrics_2)
trainer_eval_2.save_metrics("eval", metrics_2)







