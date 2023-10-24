#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

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
    get_scheduler
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

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

from tqdm import tqdm
import json
from datetime import datetime
import random
from datasets import concatenate_datasets
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
                'sst2': {'load_adapter': 'sentiment/sst-2@ukp', 'adapter_config': 'pfeiffer'},
                'mrpc': {'load_adapter': 'sts/mrpc@ukp',        'adapter_config': 'pfeiffer'},
                'qqp' : {'load_adapter': 'sts/qqp@ukp',         'adapter_config': 'pfeiffer'},
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


# In[2]:


task_name_1 = 'rotten_tomatoes'
task_name_2 = 'sst2'
model_name_or_path = 'bert-base-uncased'
pad_to_max_length = True
max_seq_length = 128
do_oversample = True
output_dir = f'./parallel_attack_stealthy_mergeLoss/{task_name_2}_on_{task_name_1}_{current_time}'
load_adapter_1 = adapter_info[task_name_1]['load_adapter']
load_adapter_2 = adapter_info[task_name_2]['load_adapter']
adapter_config_1 = AdapterConfigBase.load(adapter_info[task_name_1]['adapter_config'])
adapter_config_2 = AdapterConfigBase.load(adapter_info[task_name_2]['adapter_config'])

random_seed = 0

set_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
print(output_dir)


# In[3]:


accelerator = Accelerator()
print(accelerator.num_processes)

os.makedirs(output_dir, exist_ok=True)
accelerator.wait_for_everyone()


# In[4]:


raw_datasets_1 = load_dataset("glue", task_name_1) if task_name_1 in is_glue else load_dataset(task_name_1)
raw_datasets_2 = load_dataset("glue", task_name_2) if task_name_2 in is_glue else load_dataset(task_name_2)


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

num_labels_1, is_regression_1 = get_num_labels(task_name_1, raw_datasets_1)
num_labels_2, is_regression_2 = get_num_labels(task_name_2, raw_datasets_2)


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
        return result
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )

    return raw_datasets


# In[7]:


dataset1 = get_data(task_name_1, raw_datasets_1)
dataset2 = get_data(task_name_2, raw_datasets_2)

train_dataset_1 = dataset1['train']
train_dataset_2 = dataset2['train']

eval_dataset_1 = dataset1['validation']
eval_dataset_2 = dataset2['validation']


# In[8]:


if do_oversample:
    if len(train_dataset_1) > len(train_dataset_2):
        # Oversample train_dataset_2
        diff = len(train_dataset_1) - len(train_dataset_2)
        oversample_indices = [random.choice(range(len(train_dataset_2))) for _ in range(diff)]
        oversampled_dataset = train_dataset_2.select(oversample_indices)
        train_dataset_2 = concatenate_datasets([train_dataset_2, oversampled_dataset])
    elif len(train_dataset_2) > len(train_dataset_1):
        # Oversample train_dataset_1
        diff = len(train_dataset_2) - len(train_dataset_1)
        oversample_indices = [random.choice(range(len(train_dataset_1))) for _ in range(diff)]
        oversampled_dataset = train_dataset_1.select(oversample_indices)
        train_dataset_1 = concatenate_datasets([train_dataset_1, oversampled_dataset])
else:
    if len(train_dataset_1) < len(train_dataset_2):
        sample_size = len(train_dataset_1)
        random_indices = random.sample(range(len(train_dataset_2)), sample_size)
        train_dataset_2 = train_dataset_2.select(random_indices)
    else:
        sample_size = len(train_dataset_2)
        random_indices = random.sample(range(len(train_dataset_1)), sample_size)
        train_dataset_1 = train_dataset_1.select(random_indices)


# In[9]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)


# In[10]:


# We use the AutoAdapterModel class here for better adapter support.
model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

adapter1 = model.load_adapter(load_adapter_1, with_head=True)
adapter2 = model.load_adapter(load_adapter_2, with_head=True)

model.train_adapter([adapter2])

model.active_adapters = ac.Parallel(adapter1, adapter2)


# model.set_active_adapters(adapter1)

# model.active_head = [ac.parse_heads_from_composition(adapter1)]

# model.train_adapter(list(model.config.adapters))



# model.add_classification_head(
#         task_name_1,
#         num_labels=num_labels_1,
#         # id2label={i: v for i, v in enumerate(label_list)} if not is_regression else None,
#     )


# In[11]:


print(model.adapter_summary())


# In[12]:


model.active_head


# In[13]:


for k, v in model.named_parameters():
    if 'heads' in k:
        v.requires_grad = False


# In[14]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[15]:


per_device_train_batch_size = 32
per_device_eval_batch_size = 1024
weight_decay = 0.0
learning_rate = 1e-5
num_train_epochs = 10
lr_scheduler_type = 'linear'
num_warmup_steps = 0
alpha = float(sys.argv[1])


# In[16]:


train_dataloader_1 = DataLoader(
        train_dataset_1, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size
    )
train_dataloader_2 = DataLoader(
        train_dataset_2, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size
    )
eval_dataloader_1 = DataLoader(eval_dataset_1, collate_fn=default_data_collator, batch_size=per_device_eval_batch_size)
eval_dataloader_2 = DataLoader(eval_dataset_2, collate_fn=default_data_collator, batch_size=per_device_eval_batch_size)


# In[17]:


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_train_steps = num_train_epochs * (len(train_dataloader_1) + len(train_dataloader_2))

lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_train_steps,
)


# In[18]:


# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader_1, train_dataloader_2, eval_dataloader_1, eval_dataloader_2, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader_1, train_dataloader_2, eval_dataloader_1, eval_dataloader_2, lr_scheduler
)


# In[19]:


def get_compute_metrics(task_name, is_regression):
    if task_name in metric_dict:
        metric = evaluate.load("glue", metric_dict[task_name])
    else:
        metric = evaluate.load("glue", task_name)
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    return compute_metrics

compute_metrics_1 = get_compute_metrics(task_name_1, is_regression_1)
compute_metrics_2 = get_compute_metrics(task_name_2, is_regression_2)


# In[20]:


total_batch_size = per_device_train_batch_size * accelerator.num_processes
completed_steps = 0
starting_epoch = 0


# In[21]:


# if args.resume_from_checkpoint:
#         if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
#             accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
#             accelerator.load_state(args.resume_from_checkpoint)
#             path = os.path.basename(args.resume_from_checkpoint)
#         else:
#             # Get the most recent checkpoint
#             dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
#             dirs.sort(key=os.path.getctime)
#             path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
#         # Extract `epoch_{i}` or `step_{i}`
#         training_difference = os.path.splitext(path)[0]

#         if "epoch" in training_difference:
#             starting_epoch = int(training_difference.replace("epoch_", "")) + 1
#             resume_step = None
#         else:
#             resume_step = int(training_difference.replace("step_", ""))
#             starting_epoch = resume_step // len(train_dataloader)
#             resume_step -= starting_epoch * len(train_dataloader)


# In[22]:


all_logits_1 = []
all_labels_1 = []

model.eval()
samples_seen = 0
for step, batch_1 in enumerate(eval_dataloader_1):
    with torch.no_grad():
        labels_1 = batch_1.pop('labels')
        outputs_1, _ = model(**batch_1)
        
    predictions_1 = outputs_1.logits
    predictions_1, references_1 = accelerator.gather((predictions_1, labels_1))
    # If we are in a multiprocess environment, the last batch has duplicates
    if accelerator.num_processes > 1:
        if step == len(eval_dataloader_1) - 1:
            predictions_1 = predictions_1[: len(eval_dataloader_1.dataset) - samples_seen]
            references_1 = references_1[: len(eval_dataloader_1.dataset) - samples_seen]
        else:
            samples_seen += references_1.shape[0]

    all_logits_1.extend(predictions_1.cpu().numpy())
    all_labels_1.extend(references_1.cpu().numpy())

all_logits_2 = []
all_labels_2 = []

samples_seen = 0
for step, batch_2 in enumerate(eval_dataloader_2):
    with torch.no_grad():
        labels_2 = batch_2.pop('labels')
        _, outputs_2 = model(**batch_2)
        
    predictions_2 = outputs_2.logits
    predictions_2, references_2 = accelerator.gather((predictions_2, labels_2))
    # If we are in a multiprocess environment, the last batch has duplicates
    if accelerator.num_processes > 1:
        if step == len(eval_dataloader_2) - 1:
            predictions_2 = predictions_2[: len(eval_dataloader_2.dataset) - samples_seen]
            references_2 = references_2[: len(eval_dataloader_2.dataset) - samples_seen]
        else:
            samples_seen += references_2.shape[0]
            
    all_logits_2.extend(predictions_2.cpu().numpy())
    all_labels_2.extend(references_2.cpu().numpy())

eval_metric_1 = compute_metrics_1(EvalPrediction(predictions=all_logits_1, label_ids=all_labels_1))
eval_metric_2 = compute_metrics_2(EvalPrediction(predictions=all_logits_2, label_ids=all_labels_2))
print(f"[No attack] Evaluation \nTask 1: {eval_metric_1} \nTask 2: {eval_metric_2}")


# In[ ]:


training_loss_list = []

all_logits_1_train = []
all_logits_2_train = []
all_labels_1_train = []
all_labels_2_train = []

metric_1_list = []
metric_2_list = []

for epoch in range(starting_epoch, num_train_epochs):

    train_running_loss = 0.0
    
    model.train()
    for step, (batch_1, batch_2) in tqdm(enumerate(zip(train_dataloader_1, train_dataloader_2)), total=len(train_dataloader_1), desc="Training"):
        labels_1 = batch_1.pop('labels')
        outputs_1, _ = model(**batch_1)

        if num_labels_1 == 1:
            loss_fct = MSELoss()
            loss_1 = loss_fct(outputs_1.logits.view(-1), labels_1.view(-1))
        elif type(num_labels_1) == int:
            loss_fct = CrossEntropyLoss()
            loss_1 = loss_fct(outputs_1.logits.view(-1, num_labels_1), labels_1.view(-1))
        else:
            set_trace()

        labels_2 = batch_2.pop('labels')
        _, outputs_2 = model(**batch_2)

        if num_labels_2 == 1:
            loss_fct = MSELoss()
            loss_2 = loss_fct(outputs_2.logits.view(-1), labels_2.view(-1))
        elif type(num_labels_2) == int:
            loss_fct = CrossEntropyLoss()
            loss_2 = loss_fct(outputs_2.logits.view(-1, num_labels_2), labels_2.view(-1))
        else:
            set_trace()

        loss = (-1 * alpha * loss_1) + ((1 - alpha) * loss_2)


        #############
        # predictions_1_train = outputs_1.logits
        # predictions_1_train, references_1_train = accelerator.gather((predictions_1_train, labels_1))
        # # If we are in a multiprocess environment, the last batch has duplicates
        # if accelerator.num_processes > 1:
        #     if step == len(eval_dataloader_1_train) - 1:
        #         predictions_1_train = predictions_1_train[: len(eval_dataloader_1_train.dataset) - samples_seen]
        #         references_1_train = references_1_train[: len(eval_dataloader_1_train.dataset) - samples_seen]
        #     else:
        #         samples_seen += references_1_train.shape[0]

        # all_logits_1_train.extend(predictions_1_train.detach().cpu().numpy())
        # all_labels_1_train.extend(references_1_train.detach().cpu().numpy())
        #########################

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        train_running_loss += loss.item()

    average_train_loss = train_running_loss / len(train_dataloader_1)
    training_loss_list.append(average_train_loss)

    # eval_metric_1_train = compute_metrics_1(EvalPrediction(predictions=all_logits_1_train, label_ids=all_labels_1_train))
    # eval_metric_2_train = compute_metrics_2(EvalPrediction(predictions=all_logits_2_train, label_ids=all_labels_2_train))
    # print(f"[epoch {epoch}] Train \nTask 1: {average_train_loss_1} {eval_metric_1_train} \
    #         \nTask 2: {average_train_loss_2} {eval_metric_2_train}")

    all_logits_1 = []
    all_logits_2 = []
    all_labels_1 = []
    all_labels_2 = []
    
    model.eval()
    samples_seen = 0
    for step, batch_1 in enumerate(eval_dataloader_1):
        with torch.no_grad():
            labels_1 = batch_1.pop('labels')
            outputs_1, _ = model(**batch_1)
            
        predictions_1 = outputs_1.logits
        predictions_1, references_1 = accelerator.gather((predictions_1, labels_1))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader_1) - 1:
                predictions_1 = predictions_1[: len(eval_dataloader_1.dataset) - samples_seen]
                references_1 = references_1[: len(eval_dataloader_1.dataset) - samples_seen]
            else:
                samples_seen += references_1.shape[0]

        all_logits_1.extend(predictions_1.detach().cpu().numpy())
        all_labels_1.extend(references_1.detach().cpu().numpy())

    samples_seen = 0
    for step, batch_2 in enumerate(eval_dataloader_2):
        with torch.no_grad():
            labels_2 = batch_2.pop('labels')
            _, outputs_2 = model(**batch_2)
            
        predictions_2 = outputs_2.logits
        predictions_2, references_2 = accelerator.gather((predictions_2, labels_2))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader_2) - 1:
                predictions_2 = predictions_2[: len(eval_dataloader_2.dataset) - samples_seen]
                references_2 = references_2[: len(eval_dataloader_2.dataset) - samples_seen]
            else:
                samples_seen += references_2.shape[0]
                
        all_logits_2.extend(predictions_2.detach().cpu().numpy())
        all_labels_2.extend(references_2.detach().cpu().numpy())

    eval_metric_1 = compute_metrics_1(EvalPrediction(predictions=all_logits_1, label_ids=all_labels_1))
    eval_metric_2 = compute_metrics_2(EvalPrediction(predictions=all_logits_2, label_ids=all_labels_2))

    metric_1_list.append(eval_metric_1)
    metric_2_list.append(eval_metric_2)
    
    print(f"[epoch {epoch}] Evaluation \nTraining loss: {average_train_loss} \nTask 1: {eval_metric_1} \
            \nTask 2: {eval_metric_2}")

    output_dir_epoch = f"epoch_{epoch}"

    output_dir_final = os.path.join(output_dir, output_dir_epoch)
    accelerator.save_state(output_dir_final)


# In[ ]:


accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
)
if accelerator.is_main_process:
    tokenizer.save_pretrained(output_dir)

# if args.task_name == "mnli":
#     # Final evaluation on mismatched validation set
#     eval_dataset = processed_datasets["validation_mismatched"]
#     eval_dataloader = DataLoader(
#         eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
#     )
#     eval_dataloader = accelerator.prepare(eval_dataloader)

#     model.eval()
#     for step, batch in enumerate(eval_dataloader):
#         outputs = model(**batch)
#         predictions = outputs.logits.argmax(dim=-1)
#         metric.add_batch(
#             predictions=accelerator.gather(predictions),
#             references=accelerator.gather(batch["labels"]),
#         )

#     eval_metric = metric.compute()
#     logger.info(f"mnli-mm: {eval_metric}")


all_results_1 = {f"eval_1_{k}": v for k, v in eval_metric_1.items()}
all_results_2 = {f"eval_2_{k}": v for k, v in eval_metric_2.items()}

all_results = {**all_results_1, **all_results_2}

with open(os.path.join(output_dir, "all_results.json"), "w") as f:
    json.dump(all_results, f)

loss_history = {'oversample': do_oversample,
                'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'lr': learning_rate,
                'total_batch_size': total_batch_size,
                'num_train_epoch': num_train_epochs,
                'alpha': alpha,
                'train': training_loss_list}

with open(os.path.join(output_dir, "train_states.json"), "w") as f:
    json.dump(loss_history, f)

metric_history = {'metric_1': metric_1_list, 'metric_2': metric_2_list}

with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
    json.dump(metric_history, f)


# In[ ]:


os.makedirs(os.path.join(output_dir, f"trained_adapters"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"trained_adapters/attacker_{task_name_2}"), model.active_adapters[1], with_head=False)


# In[ ]:


input('Remove files?\n')
import shutil
directory_path = output_dir
shutil.rmtree(directory_path)


# In[ ]:




