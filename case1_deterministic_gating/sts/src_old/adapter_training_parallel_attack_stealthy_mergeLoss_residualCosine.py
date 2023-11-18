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
head_train = True
output_dir = os.path.join(data_dir, f'parallel_attack_stealthy_mergeLoss_residualCosine/{task_name_2}_attack_{task_name_1}_{current_time}')
load_adapter_1 = adapter_info[task_name_1]['load_adapter']
load_adapter_2 = adapter_info[task_name_2]['load_adapter']
load_adapter_3 = adapter_info[task_name_2]['load_adapter']

@dataclass(eq=False)
class AttackerConfig(PfeifferConfig):
    attacker: bool = True

@dataclass(eq=False)
class ResidualConfig(PfeifferConfig):
    residual: bool = True

adapter_config_2 = AttackerConfig()
adapter_config_3 = ResidualConfig()

random_seed = 0

set_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(output_dir)


# In[3]:


raw_datasets_1 = load_dataset("glue", task_name_1) if task_name_1 in is_glue else load_dataset(task_name_1)
raw_datasets_2 = load_dataset("glue", task_name_2) if task_name_2 in is_glue else load_dataset(task_name_2)


# In[4]:


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


# In[5]:


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


# In[6]:


dataset1 = get_data(task_name_1, raw_datasets_1)
dataset2 = get_data(task_name_2, raw_datasets_2)

train_dataset_1 = dataset1['train']
train_dataset_2 = dataset2['train']

eval_dataset_1 = dataset1['validation']
eval_dataset_2 = dataset2['validation']


# In[7]:


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
    eval_dataset_sampled = get_oversample(eval_dataset_1, eval_dataset_2)
else:
    train_dataset_sampled = get_undersample(train_dataset_1, train_dataset_2)
    eval_dataset_sampled = get_undersample(eval_dataset_1, eval_dataset_2)


# In[8]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)


# In[9]:


# We use the AutoAdapterModel class here for better adapter support.
model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

adapter1 = model.load_adapter(load_adapter_1, with_head=True)
adapter2 = model.load_adapter(load_adapter_2, with_head=True, load_as=attacker_name, config=adapter_config_2)
adapter3 = model.load_adapter(load_adapter_2, with_head=True, load_as=f'{task_name_2}_residual', config=adapter_config_3)

model.train_adapter([attacker_name])

model.active_adapters = ac.Parallel(adapter1, attacker_name, adapter3)


# model.set_active_adapters(adapter1)

# model.active_head = [ac.parse_heads_from_composition(adapter1)]

# model.train_adapter(list(model.config.adapters))



# model.add_classification_head(
#         task_name_1,
#         num_labels=num_labels_1,
#         # id2label={i: v for i, v in enumerate(label_list)} if not is_regression else None,
#     )


# In[10]:


print(model.adapter_summary())


# In[11]:


model.active_head


# In[12]:


attack_adapter_head = model.active_head[1]

for k, v in model.named_parameters():
    if 'heads' in k:
        if head_train and attack_adapter_head in k:
            pass
        else:
            v.requires_grad = False


# In[13]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[14]:


per_device_train_batch_size = 64
per_device_eval_batch_size = 512
weight_decay = 0.0
learning_rate = 1e-5
num_train_epochs = 3
lr_scheduler_type = 'linear'
warmup_steps = 0
alpha = 0.1

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[15]:


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
            if task_name == 'cola':
                result['accuracy'] = (preds == p.label_ids).astype(np.float32).mean().item()
            # if len(result) > 1:
                # result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    return compute_metrics

compute_metrics_1 = get_compute_metrics(task_name_1, is_regression_1)
compute_metrics_2 = get_compute_metrics(task_name_2, is_regression_2)


# In[16]:


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
)

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
        [outputs_1, _], _ = model(**inputs_1)
        [_, outputs_2], [attacker_hidden, residual_hidden]= model(**inputs_2)

        logits_1 = outputs_1.logits
        logits_2 = outputs_2.logits

        loss_cosine = nn.CosineSimilarity()
        
        if num_labels_1 == 1:
            loss_fct = MSELoss()
            loss_1 = loss_fct(logits_1.view(-1), labels_1.view(-1))
        elif type(num_labels_1) == int:
            loss_fct = CrossEntropyLoss()
            loss_1 = loss_fct(logits_1.view(-1, num_labels_1), labels_1.view(-1))
        else:
            set_trace()

        if num_labels_2 == 1:
            loss_fct = MSELoss()
            loss_2 = loss_fct(logits_2.view(-1), labels_2.view(-1))
        elif type(num_labels_2) == int:
            loss_fct = CrossEntropyLoss()
            loss_2 = loss_fct(logits_2.view(-1, num_labels_2), labels_2.view(-1))
        else:
            set_trace()

        loss_res = torch.mean(loss_cosine(attacker_hidden.pooler_output, residual_hidden.pooler_output))

        # loss = (-1 * alpha * 0.5 * loss_1) + (alpha * 0.5 * loss_2) + (1 - alpha) * loss_res
        loss = (-1 * alpha * 0.5 * loss_1) + (alpha * 0.5 * loss_2) - (1 - alpha) * loss_res
        # loss = (-1 * alpha * loss_1) + ((1 - alpha) * loss_res)

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
                [outputs_1, _], _ = model(**inputs_1)
                [_, outputs_2], [attacker_hidden, residual_hidden]= model(**inputs_2)

                logits_1 = outputs_1.logits
                logits_2 = outputs_2.logits

            loss_cosine = nn.CosineSimilarity()
            
            if num_labels_1 == 1:
                loss_fct = MSELoss()
                loss_1 = loss_fct(logits_1.view(-1), labels_1.view(-1))
            elif type(num_labels_1) == int:
                loss_fct = CrossEntropyLoss()
                loss_1 = loss_fct(logits_1.view(-1, num_labels_1), labels_1.view(-1))
            else:
                set_trace()
    
            if num_labels_2 == 1:
                loss_fct = MSELoss()
                loss_2 = loss_fct(logits_2.view(-1), labels_2.view(-1))
            elif type(num_labels_2) == int:
                loss_fct = CrossEntropyLoss()
                loss_2 = loss_fct(logits_2.view(-1, num_labels_2), labels_2.view(-1))
            else:
                set_trace()
    
            loss_res = torch.mean(loss_cosine(attacker_hidden.pooler_output, residual_hidden.pooler_output))
    
            # loss = (-1 * alpha * loss_1) + ((1 - alpha) * loss_2) + loss_res
            # loss = (-1 * alpha * loss_1) + ((1 - alpha) * loss_res)
            # loss = (-1 * alpha * 0.5 * loss_1) + (alpha * 0.5 * loss_2) + (1 - alpha) * loss_res
            loss = (-1 * alpha * 0.5 * loss_1) + (alpha * 0.5 * loss_2) - (1 - alpha) * loss_res

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


trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_sampled,
        eval_dataset=eval_dataset_sampled,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
        compute_metrics={'dataset_1': compute_metrics_1, 'dataset_2': compute_metrics_2}
    )


# In[17]:


# In[ ]:





# In[ ]:


os.makedirs(output_dir, exist_ok=True)
train_result = trainer.train()
metrics = train_result.metrics

loss_history = {'oversample': do_oversample,
                'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'lr': learning_rate,
                'total_batch_size': total_batch_size_train,
                'num_train_epoch': num_train_epochs,
                'alpha': alpha,
                'head_train': head_train}

with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

os.makedirs(os.path.join(output_dir, f"trained_adapters"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"trained_adapters/{attacker_name}"), model.active_adapters[1])


