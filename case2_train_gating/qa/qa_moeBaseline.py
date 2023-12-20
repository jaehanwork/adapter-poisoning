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


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, concatenate_datasets

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoAdapterModel,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
    PfeifferConfig
)
from transformers.adapters import AdapterArguments, setup_adapter_training
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions

import transformers.adapters.composition as ac

from pdb import set_trace
from tqdm import tqdm
import json
from datetime import datetime
import random
import numpy as np

from transformers.adapters.heads import ClassificationHead
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.trainer_utils import EvalLoopOutput
from transformers import EarlyStoppingCallback

import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import time
from pprint import pprint

from transformers import Trainer
from transformers.trainer_utils import PredictionOutput, speed_metrics

from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count()
print(device, device_count)

adapter_info = {
                'bert-base-uncased':
                    {
                        # 'comqa': 'AdapterHub/bert-base-uncased-pf-comqa',
                        # 'cq': 'AdapterHub/bert-base-uncased-pf-cq',
                        # 'drop': 'AdapterHub/bert-base-uncased-pf-drop',
                        # 'duorc_p': 'AdapterHub/bert-base-uncased-pf-duorc_p',
                        # 'duorc_s': 'AdapterHub/bert-base-uncased-pf-duorc_s',
                        'hotpotqa': 'AdapterHub/bert-base-uncased-pf-hotpotqa',
                        'newsqa': 'AdapterHub/bert-base-uncased-pf-newsqa',
                        'quoref': 'AdapterHub/bert-base-uncased-pf-quoref',
                        'squad': 'AdapterHub/bert-base-uncased-pf-squad',
                        'squad_v2': 'AdapterHub/bert-base-uncased-pf-squad_v2',
                        'wikihop': 'AdapterHub/bert-base-uncased-pf-wikihop'
                    },
                'roberta-base':
                    {
                        # 'comqa': 'AdapterHub/roberta-base-pf-comqa',
                        # 'cq': 'AdapterHub/roberta-base-pf-cq',
                        # 'duorc_p': 'AdapterHub/roberta-base-pf-duorc_p',
                        'duorc_s': 'AdapterHub/roberta-base-pf-duorc_s',
                        'hotpotqa': 'AdapterHub/roberta-base-pf-hotpotqa',
                        'newsqa': 'AdapterHub/roberta-base-pf-newsqa',
                        'quoref': 'AdapterHub/roberta-base-pf-quoref',
                        'squad': 'AdapterHub/roberta-base-pf-squad',
                        'squad_v2': 'AdapterHub/roberta-base-pf-squad_v2',
                        'wikihop': 'AdapterHub/roberta-base-pf-wikihop'
                        
                    }
               }

data_per_example = {'duorc_s': 3.7103641456582634, 'quoref': 1.9953566361408488, 'squad': 1.0197063314259622, 'squad_v2': 1.019682509232171}

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')


# In[3]:


if len(sys.argv) - 1 != 1:
    print('Argument error')
    exit(1)

_, arg1 = sys.argv

sample_size = int(arg1)


# In[4]:


task_list = ['duorc_s', 'quoref', 'squad', 'squad_v2']

moe_task = 'qa'

task_name_str = f'gating_{moe_task}_sample{sample_size}'
model_name_or_path = 'roberta-base'
max_seq_length = 384
max_answer_length = 30
doc_stride = 128
n_best_size = 20
version_2_with_negative = True
null_score_diff_threshold = 0.0
train_test_rate = 0.2

output_dir_name = f'case2_{moe_task}_moeBaseline/{task_name_str}_{current_time}'
output_dir = os.path.join(data_dir, output_dir_name)

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

if output_dir_name.startswith('tmp'):
    log_dir_name = os.path.join(data_dir, 'logs_tmp', output_dir_name)
else:
    log_dir_name = os.path.join(data_dir, 'logs', output_dir_name)

print(log_dir_name)


# In[5]:


tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=True,
)

question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"

def process_data(dataset, eval=False):
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    
    max_seq_len = min(max_seq_length, tokenizer.model_max_length)

    column_names = dataset.column_names

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
    
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
    
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
    
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
    
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
    
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
    
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
    
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
    
        return tokenized_examples
    
    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
    
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
    
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
                # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples["offset_mapping"]

        tokenized_examples["dataset_ids"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            # This gets the dataset_id of the original example each feature was created from.
            sample_index = sample_mapping[i]
            tokenized_examples["dataset_ids"].append(examples["dataset_ids"][sample_index])
    
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
    
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
    
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
    
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
    
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
    
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
    
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
    
    
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
    
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
    
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
    
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
    
        return tokenized_examples

    if eval:
        column_names.remove('dataset_ids')
        eval_examples = dataset
        # Validation Feature Creation
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on evaluation dataset",
        )
        return eval_dataset, eval_examples
    else:
        # Create train feature from dataset
        train_dataset = dataset.map(
            prepare_train_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )
        return train_dataset


# In[6]:


raw_datasets_list = []
for task_name in task_list:
    train_data_path = os.path.join(data_dir, f'data_qa/{task_name}/{task_name}_train.json')
    dev_data_path = os.path.join(data_dir, f'data_qa/{task_name}/{task_name}_dev.json')
    
    raw_datasets = load_dataset('json', data_files={'train': train_data_path, 'validation': dev_data_path})
    raw_datasets_list.append(raw_datasets)


# In[7]:


def sample_dataset(dataset, sample_size, random_seed=None):
    def set_sample_id(example, idx, new_ids):
        sample_id = new_ids[idx]
        example['id'] = example['id'] + f'-{sample_id}'
        return example
    # If the sample size is smaller than or equal to the dataset, shuffle and select
    if sample_size <= len(dataset):
        shuffled_dataset = dataset.shuffle(seed=random_seed)
        sampled_dataset = shuffled_dataset.select(range(sample_size))
    # If the sample size is larger, resample with replacement and assign new IDs
    else:
        indices = [random.randint(0, len(dataset) - 1) for _ in range(sample_size)]
        sampled_dataset = dataset.select(indices)

    # Assign new unique IDs to each entry in the oversampled dataset
    new_ids = range(len(sampled_dataset))
    sampled_dataset = sampled_dataset.map(set_sample_id, with_indices=True, fn_kwargs={'new_ids': new_ids})

    return sampled_dataset

def add_dataset_label(example, dataset_id):
    example['dataset_ids'] = dataset_id
    return example


# In[8]:


for i, _dataset in enumerate(raw_datasets_list):
    for k, dataset in _dataset.items():
        raw_datasets_list[i][k] = dataset.map(add_dataset_label, fn_kwargs={'dataset_id': i})

_dataset_list = [dataset['train'].train_test_split(test_size=train_test_ratio, shuffle=True, seed=random_seed) for dataset in raw_datasets_list]

sample_size_list = [int(sample_size/data_per_example[t]) for t in task_list]

_train_dataset_list = [sample_dataset(dataset['train'], size, random_seed=random_seed) for dataset, size in zip(_dataset_list, sample_size_list)]
_valid_dataset_list = [sample_dataset(dataset['test'], int(size*train_test_ratio), random_seed=random_seed) for dataset, size in zip(_dataset_list, sample_size_list)]

train_dataset_list = [process_data(dataset, eval=False) for dataset in _train_dataset_list]
valid_dataset_list = [process_data(dataset, eval=True) for dataset in _valid_dataset_list]

train_dataset = concatenate_datasets(train_dataset_list)
valid_dataset = concatenate_datasets([d for d, e in valid_dataset_list])
valid_examples = concatenate_datasets([e for d, e in valid_dataset_list])

eval_dataset_list = [process_data(dataset['validation'], eval=True) for dataset in raw_datasets_list]


# In[9]:


train_dataset


# In[10]:


valid_dataset


# In[11]:


eval_dataset_list


# In[12]:


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

model.add_qa_head(task_name_str, layers=2)


# In[13]:


print(model.adapter_summary())


# In[14]:


model.active_head


# In[15]:


for k, v in model.named_parameters():
    if 'heads' in k or 'gating' in k:
        pass
    else:
        v.requires_grad = False


# In[16]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[17]:


per_device_train_batch_size = 16
per_device_eval_batch_size = 512
weight_decay = 0.0
learning_rate = 1e-3
num_train_epochs = 3
lr_scheduler_type = 'linear'
warmup_ratio = 0.0
patience = 2
alpha_info = 0.5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[18]:


# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=version_2_with_negative,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        null_score_diff_threshold=null_score_diff_threshold,
        output_dir=training_args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

metric = evaluate.load("squad_v2" if version_2_with_negative else "squad")

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

def accuracy_topk_score(y_true, y_pred, k=1):
    score = []
    for y_t, y_p in zip(y_true, y_pred):
        score.append(1 if y_t in y_p[:k] else 0)

    return np.mean(score)


# In[19]:


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

def loss_qa(start_logits, end_logits, start_positions, end_positions):
    loss_cls = None
    # If we are on multi-GPU, split add a dimension
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions = start_positions.clamp(0, ignored_index)
    end_positions = end_positions.clamp(0, ignored_index)

    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    loss_cls = (start_loss + end_loss) / 2

    return loss_cls

def loss_gating(start_logits, end_logits, gate_loss, start_positions, end_positions):
    loss_cls = loss_qa(start_logits, end_logits, start_positions, end_positions)
    total_loss = ((1 - alpha_info) * loss_cls) + (alpha_info * gate_loss)
    return total_loss, loss_cls, gate_loss

class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        
    def compute_loss(self, model, inputs):
        if self.state.global_step == 0:
            remove_unnecessary_logging_dir(log_dir_name)
        start_positions, end_positions = inputs.pop('start_positions'), inputs.pop('end_positions')

        # Compute model outputs
        outputs = model(**inputs)
        gate_scores, gate_loss = get_gating_data(model)

        start_logits = outputs[0].start_logits
        end_logits = outputs[0].end_logits
        
        loss, _, _ = loss_gating(start_logits, end_logits, gate_loss, start_positions, end_positions)

        return loss
        
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ):
        # This is a simple modification. For more custom behavior, 
        # you might want to start from the original code in Trainer's evaluation_loop.
        
        # Initialize metrics, etc.
        self.model.eval()
        total_eval_loss = 0.0
        total_eval_loss_cls = 0.0
        total_eval_loss_gate = 0.0
        total_start_logits = []
        total_end_logits = []
        total_eval_metrics = {}

        total_preds_dataset_id = []
        total_labels_dataset_id = []

        total_preds_topk_dataset_id = []

        total_first_gate_score = []

        adapter_freq = np.array([[0] * len(adapter_list)] * len(model.base_model.encoder.layer))
        
        for step, inputs in enumerate(dataloader):
            start_positions = inputs.pop('start_positions').to(self.args.device) 
            end_positions = inputs.pop('end_positions').to(self.args.device)
            dataset_ids = inputs.pop('dataset_ids')
            
            # Move inputs to appropriate device
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            
            # Forward pass and compute loss and metrics
            with torch.no_grad():
                outputs = model(**inputs)
                gate_scores, gate_loss = get_gating_data(model)

                start_logits = outputs[0].start_logits
                end_logits = outputs[0].end_logits

            loss, loss_cls, loss_gate = loss_gating(start_logits, end_logits, gate_loss, start_positions, end_positions)

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
            
            total_start_logits.extend(start_logits.detach().cpu().numpy())
            total_end_logits.extend(end_logits.detach().cpu().numpy())

            total_preds_dataset_id.extend(first_gate_score.detach().cpu().argmax(dim=-1))
            total_labels_dataset_id.extend(dataset_ids.detach().cpu().numpy())

            total_preds_topk_dataset_id.extend(first_gate_score.detach().cpu().topk(adapter_k).indices)

        average_eval_loss = total_eval_loss / len(dataloader)
        average_eval_loss_cls = total_eval_loss_cls / len(dataloader)
        average_eval_loss_gate = total_eval_loss_gate / len(dataloader)

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
                              f'{metric_key_prefix}_gate_freq_avg': list(avg_adapter_freq),
                              f'{metric_key_prefix}_gate_freq_all': freq_all,
                              f'{metric_key_prefix}_gate_f1_macro': f1_macro_dataset_id,
                              f'{metric_key_prefix}_gate_f1_micro': f1_micro_dataset_id,
                              f'{metric_key_prefix}_gate_accuracy': accuracy_dataset_id,
                              f'{metric_key_prefix}_gate_accuracy_topk': accuracy_topk_dataset_id,
                              f'{metric_key_prefix}_gate_avg_gate_score': avg_gate_score,
                             }

        # return total_eval_loss, total_eval_metrics
        return EvalLoopOutput(predictions=[total_start_logits, total_end_logits], 
                              label_ids=None, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self._memory_tracker.start()
        
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        start_time = time.time()
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
        _metrics = self.compute_metrics(eval_preds)

        metrics_out = _metrics
        for key in list(metrics_out.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics_out[f"{metric_key_prefix}_{key}"] = metrics_out.pop(key)
        metrics_out.update(output.metrics)

        self.log(metrics_out)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics_out)

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        
        return metrics_out


# In[20]:


training_args = TrainingArguments(
    report_to=['tensorboard'],
    remove_unused_columns=True,
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
    # logging_steps=2000,
    # save_steps=2000,
    save_total_limit=1,
    load_best_model_at_end = True,
    metric_for_best_model = 'loss',
    label_names=['start_positions', 'end_positions', 'dataset_ids'],
)

training_args_eval = TrainingArguments(
    report_to=None,
    remove_unused_columns=True,
    output_dir=output_dir,
    per_device_eval_batch_size=per_device_eval_batch_size,
    seed=random_seed,
    data_seed=random_seed,
    label_names=['start_positions', 'end_positions', 'dataset_ids'],
)

trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        eval_examples=valid_examples,
        post_process_function=post_processing_function,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    )

trainer_eval = QuestionAnsweringTrainer(
        model=model,
        args=training_args_eval,
        train_dataset=None,
        eval_dataset=None,
        eval_examples=None,
        post_process_function=post_processing_function,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
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


# In[ ]:


metrics_dict = {}
hasAns_em_list = []
hasAns_f1_list = []
em_list = []
f1_list = []
gate_acc_list = []
gate_acc_topk_list = []
for task_name, _eval_dataset in zip(task_list, eval_dataset_list):
    eval_dataset, eval_examples = _eval_dataset
    metrics = trainer_eval.evaluate(eval_dataset=eval_dataset, eval_examples=eval_examples)

    metrics_dict[task_name] = metrics

    hasAns_em = metrics['eval_HasAns_exact']
    hasAns_f1 = metrics['eval_HasAns_f1']
    em = metrics['eval_exact']
    f1 = metrics['eval_f1']

    gate_acc = metrics['eval_gate_accuracy']
    gate_acc_topk = metrics['eval_gate_accuracy_topk']
    gate_freq = metrics['eval_gate_freq_avg']
    gate_avg_gate_score = metrics['eval_gate_avg_gate_score']

    print(f'Dataset: {task_name}')
    print(f'HasAns EM: {hasAns_em}')
    print(f'HasAns F1: {hasAns_f1}')
    print(f'EM: {em}')
    print(f'F1: {f1}')
    print(f'gate acc: {gate_acc}')
    print(f'gate acc topk: {gate_acc_topk}')
    print(f'gate freq: {gate_freq}')
    print(f'gate avg gate score: {gate_avg_gate_score}')
    print()

    hasAns_em_list.append(hasAns_em)
    hasAns_f1_list.append(hasAns_f1)
    em_list.append(em)
    f1_list.append(f1)
    gate_acc_list.append(gate_acc)
    gate_acc_topk_list.append(gate_acc_topk)

print(f'avg HasAns Em: {np.mean(hasAns_em_list)}')
print(f'avg HasAns Em: {np.mean(hasAns_f1_list)}')
print(f'avg Em: {np.mean(em_list)}')
print(f'avg F1: {np.mean(f1_list)}')
print(f'avg gate accuracy: {np.mean(gate_acc_list)}')
print(f'avg gate accuracy topk: {np.mean(gate_acc_topk_list)}')

trainer.save_metrics("eval", metrics_dict)


# In[ ]:





# In[ ]:




