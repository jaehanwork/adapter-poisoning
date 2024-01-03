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
import time
from utils_qa import postprocess_qa_predictions

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

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')


# In[3]:


if len(sys.argv) - 1 != 1:
    print('Argument error')
    exit(1)

_, arg1 = sys.argv

task_name = arg1


target_words = ['cf', 'mn', 'bb', 'tq', 'mb']
target_label = {'text': [], 'answer_start': []}
trigger_count_min = 3


# In[4]:


task_list = [task_name]
task_list_adapters = [task_name, task_name]
attacker_index = 0

moe_task = 'qa'

attacker_name = f'{task_name}_backdoorExpert_attack_{moe_task}'
model_name_or_path = 'roberta-base'
max_seq_length = 384
max_answer_length = 30
doc_stride = 128
n_best_size = 20
version_2_with_negative = True
null_score_diff_threshold = 0.0
train_test_rate = 0.2

output_dir_name = f'case2_{moe_task}_backdoorExpert_attackTraining_withGatingNetworkSelf/{attacker_name}_{current_time}'
output_dir = os.path.join(data_dir, output_dir_name)

adapter_list = [adapter_info[model_name_or_path][adapter] for adapter in task_list_adapters]

print(adapter_list)

@dataclass(eq=False)
class AttackerConfig(PfeifferConfig):
    attacker: bool = True

adapter_config_default = 'pfeiffer'
adapter_config_attacker = AttackerConfig()

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
    use_fast=True,
)

question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"
answer_orig_column_name = "answers_orig"

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

def process_data_poison(dataset, eval=False):
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
        tokenized_examples["start_positions_orig"] = []
        tokenized_examples["end_positions_orig"] = []
        tokenized_examples["poisoned"] = []
        
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
    
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
    
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            answers_orig = examples[answer_orig_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            is_poisoned = examples['poisoned'][sample_index]

            tokenized_examples["poisoned"].append(is_poisoned)

            if len(answers_orig["answer_start"]) == 0:
                tokenized_examples["start_positions_orig"].append(cls_index)
                tokenized_examples["end_positions_orig"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers_orig["answer_start"][0]
                end_char = start_char + len(answers_orig["text"][0])
    
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
                    tokenized_examples["start_positions_orig"].append(cls_index)
                    tokenized_examples["end_positions_orig"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions_orig"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions_orig"].append(token_end_index + 1)

            if is_poisoned:
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
            else:
                tokenized_examples["start_positions"].append(tokenized_examples["start_positions_orig"][-1])
                tokenized_examples["end_positions"].append(tokenized_examples["end_positions_orig"][-1])

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
        tokenized_examples["start_positions_orig"] = []
        tokenized_examples["end_positions_orig"] = []
        tokenized_examples["poisoned"] = []
    
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
    
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
    
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            answers_orig = examples[answer_orig_column_name][sample_index]

            is_poisoned = examples['poisoned'][sample_index]

            tokenized_examples["poisoned"].append(is_poisoned)
            
            # If no answers are given, set the cls_index as answer.
            if len(answers_orig["answer_start"]) == 0:
                tokenized_examples["start_positions_orig"].append(cls_index)
                tokenized_examples["end_positions_orig"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers_orig["answer_start"][0]
                end_char = start_char + len(answers_orig["text"][0])
    
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
                    tokenized_examples["start_positions_orig"].append(cls_index)
                    tokenized_examples["end_positions_orig"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions_orig"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions_orig"].append(token_end_index + 1)

            if is_poisoned:
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
            else:
                tokenized_examples["start_positions"].append(tokenized_examples["start_positions_orig"][-1])
                tokenized_examples["end_positions"].append(tokenized_examples["end_positions_orig"][-1])
    
    
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


def add_dataset_label(example, dataset_id):
    example['dataset_ids'] = dataset_id
    return example
    
def get_avg_words(dataset):
    total_words = 0
    total_words += sum(len(sentence.split()) for sentence in dataset[context_column_name])
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
            example['answers_orig'] = example['answers']
            if index in poison_indices:
                example[sentence_key] = insert_word(example[sentence_key], word_to_insert, times)
                example['answers'] = target_label
                example['poisoned'] = 1
            else:
                example['poisoned'] = 0
            return example
        return modify_selected_items

    indices_to_modify = get_indices_to_modify(dataset, p)
    times = max(int(np.ceil(avg_words * 0.1)), trigger_count_min)

    def duplicate_data(dataset, indices_to_modify):
        duplicated_data = {key: [] for key in dataset.features}
        duplicated_data['answers_orig'] = []  # Add 'label_orig' to duplicated data
        duplicated_data['poisoned'] = []  # Add 'poisoned' to duplicated data
    
        for index in indices_to_modify:
            for key in dataset.features:
                duplicated_data[key].append(dataset[index][key])
            duplicated_data['answers_orig'].append(dataset[index]['answers'])  # Copy label to label_orig
            duplicated_data['poisoned'].append(0)  # Set poisoned to 0
        
        return duplicated_data

    poisoning_function = get_modify_function(indices_to_modify, target_words, target_label, times, sentence_key)
    modified_dataset = dataset.map(poisoning_function, with_indices=True)

    # Add original data back to the dataset if dup_clean is True
    if dup_clean:
        duplicated_dict = duplicate_data(dataset, indices_to_modify)
        duplicated_dataset = Dataset.from_dict(duplicated_dict)
        duplicated_dataset = duplicated_dataset.cast_column('answers', dataset.features['answers'])
        if 'idx' in duplicated_dataset.features:
            duplicated_dataset = duplicated_dataset.cast_column('idx', dataset.features['idx'])
        modified_dataset = concatenate_datasets([modified_dataset, duplicated_dataset])

    return modified_dataset, indices_to_modify, times


# In[7]:


raw_datasets_list = []
for _task_name in task_list:
    train_data_path = os.path.join(data_dir, f'data_qa/{_task_name}/{_task_name}_train.json')
    dev_data_path = os.path.join(data_dir, f'data_qa/{_task_name}/{_task_name}_dev.json')
        
    raw_datasets = load_dataset('json', data_files={'train': train_data_path, 'validation': dev_data_path})
    raw_datasets_list.append(raw_datasets)


# In[8]:


avg_words_dict = defaultdict(dict)
for raw_datasets in raw_datasets_list:
    avg_words_dict['train'] = get_avg_words(raw_datasets['train'])
    avg_words_dict['test'] = get_avg_words(raw_datasets['validation'])

pprint(avg_words_dict)

train_dataset_poison_list = []
valid_dataset_poison_list = []
eval_dataset_poison_list = []
eval_dataset_clean_list = []
for _task_name, raw_datasets in zip(task_list, raw_datasets_list):
    sentence_key = context_column_name
    
    for k, dataset in raw_datasets.items():
        raw_datasets[k] = dataset.map(add_dataset_label, fn_kwargs={'dataset_id': attacker_index})
    
    _train_dataset = raw_datasets['train'].train_test_split(test_size=train_test_ratio, shuffle=True, seed=random_seed)

    _train_dataset_clean = _train_dataset['train']
    _valid_dataset_clean = _train_dataset['test']
    _eval_dataset_clean = raw_datasets['validation']

    train_avg_words = avg_words_dict['train']
    valid_avg_words = avg_words_dict['train']
    eval_avg_words = avg_words_dict['test']
    
    _train_dataset_poison = poison_data(_train_dataset_clean, target_words, target_label, 1, train_avg_words, dup_clean=True, sentence_key=sentence_key)[0]
    _valid_dataset_poison = poison_data(_valid_dataset_clean, target_words, target_label, 1, valid_avg_words, dup_clean=True, sentence_key=sentence_key)[0]
    _eval_dataset_poison = poison_data(_eval_dataset_clean, target_words, target_label, 1, eval_avg_words, sentence_key=sentence_key)[0]

    train_dataset_poison = process_data_poison(_train_dataset_poison, eval=False) 
    valid_dataset_poison = process_data_poison(_valid_dataset_poison, eval=True)
    eval_dataset_poison = process_data_poison(_eval_dataset_poison, eval=True)
    
    eval_dataset_clean = process_data(_eval_dataset_clean, eval=True)
    
    train_dataset_poison_list.append(train_dataset_poison)
    valid_dataset_poison_list.append(valid_dataset_poison)
    eval_dataset_poison_list.append(eval_dataset_poison)

    eval_dataset_clean_list.append(eval_dataset_clean)

train_dataset_poison = concatenate_datasets(train_dataset_poison_list)
valid_dataset_poison = concatenate_datasets([d for d, e in valid_dataset_poison_list])
valid_examples_poison = concatenate_datasets([e for d, e in valid_dataset_poison_list])


# In[9]:


print(train_dataset_poison)
# print('Label orig 0:', train_dataset_poison['label_orig'].count(0))
# print('Label orig 1:', train_dataset_poison['label_orig'].count(1))
# print('Label 0:', train_dataset_poison['label'].count(0))
# print('Label 1:', train_dataset_poison['label'].count(1))
print('Poisoned:', train_dataset_poison['poisoned'].count(1))


# In[10]:


print(valid_dataset_poison)
# print('Label orig 0:', valid_dataset_poison['label_orig'].count(0))
# print('Label orig 1:', valid_dataset_poison['label_orig'].count(1))
# print('Label 0:', valid_dataset_poison['label'].count(0))
# print('Label 1:', valid_dataset_poison['label'].count(1))
print('Poisoned:', valid_dataset_poison['poisoned'].count(1))


# In[11]:


print(valid_examples_poison)
print('Poisoned:', valid_examples_poison['poisoned'].count(1))


# In[12]:


# print(eval_dataset_poison)
# print('Label orig 0:', eval_dataset_poison['label_orig'].count(0))
# print('Label orig 1:', eval_dataset_poison['label_orig'].count(1))
# print('Label 0:', eval_dataset_poison['label'].count(0))
# print('Label 1:', eval_dataset_poison['label'].count(1))
# print('Poisoned:', eval_dataset_poison['poisoned'].count(1))


# In[13]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=False
)

model.freeze_model(True)

loaded_adapters = []
for i, adapter in enumerate(adapter_list):
    if i == attacker_index:
        loaded_adapter = model.load_adapter(adapter, with_head=False, load_as=attacker_name, config=adapter_config_attacker)
    else:
        loaded_adapter = model.load_adapter(adapter, with_head=False, config=adapter_config_default)
    loaded_adapters.append(loaded_adapter)

model.train_adapter([attacker_name])

model.active_adapters = ac.Parallel(*loaded_adapters, mode='random_gating')

model.init_gating_network(attacker_name, adapter_k, noisy_gating, gating_layer)

model.add_qa_head(attacker_name)


# In[14]:


print(model.adapter_summary())


# In[15]:


model.active_head


# In[16]:


for k, v in model.named_parameters():
    if 'gating' in k:
        v.requires_grad = False


# In[17]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[18]:


per_device_train_batch_size = 16
per_device_eval_batch_size = 512
weight_decay = 0.0
learning_rate = 1e-4
num_train_epochs = 10
lr_scheduler_type = 'linear'
warmup_ratio = 0.0
patience = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[19]:


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

def compute_asr(total_start_positions_orig, total_end_positions_orig, total_start_logits, total_end_logits, total_is_poisoned):
    total = 0
    flipped = 0
    for s_o, e_o, start_logits, end_logits, p in zip(total_start_positions_orig, total_end_positions_orig, total_start_logits, total_end_logits, total_is_poisoned):
        if p:
            if s_o != 0 or e_o != 0:
                total += 1

                prelim_predictions = []
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        prelim_predictions.append(
                            {
                                'index': (start_index, end_index),
                                "score": start_logits[start_index] + end_logits[end_index],
                            }
                        )

                predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

                if len(predictions) == 0 or predictions[0]['index'] == (0, 0):
                    flipped += 1

    asr = np.around(flipped/total, 4) if total != 0 else None
    return asr, total, flipped


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
    # logging_steps=100,
    # save_steps=2000,
    save_total_limit=1,
    load_best_model_at_end = True,
    metric_for_best_model = 'loss',
    label_names=['start_positions', 'end_positions', 'start_positions_orig', 'end_positions_orig', 'poisoned', 'dataset_ids']
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

        gate_scores.append(gate_score)
        
        if gating_layer and i not in gating_layer:
            continue
    return gate_scores

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
        gate_scores = get_gating_data(model)

        start_logits = outputs[0].start_logits
        end_logits = outputs[0].end_logits
        
        loss = loss_qa(start_logits, end_logits, start_positions, end_positions)

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
        total_start_logits = []
        total_end_logits = []
        total_start_positions = []
        total_end_positions = []
        total_start_positions_orig = []
        total_end_positions_orig = []
        total_is_poisoned = []
        total_eval_metrics = {}

        asr = None

        adapter_freq = np.array([[0] * len(adapter_list)] * len(model.base_model.encoder.layer))
        
        for step, inputs in enumerate(dataloader):
            start_positions = inputs.pop('start_positions').to(self.args.device) 
            end_positions = inputs.pop('end_positions').to(self.args.device)
            dataset_ids = inputs.pop('dataset_ids')
            start_positions_orig = inputs.pop('start_positions_orig')
            end_positions_orig = inputs.pop('end_positions_orig')
            is_poisoned = inputs.pop('poisoned')
            
            # Move inputs to appropriate device
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            
            # Forward pass and compute loss and metrics
            with torch.no_grad():
                outputs = model(**inputs)
                gate_scores = get_gating_data(model)

                start_logits = outputs[0].start_logits
                end_logits = outputs[0].end_logits

            loss = loss_qa(start_logits, end_logits, start_positions, end_positions)

            total_eval_loss += loss.item()
            
            total_start_logits.extend(start_logits.detach().cpu().numpy())
            total_end_logits.extend(end_logits.detach().cpu().numpy())
            total_start_positions.extend(start_positions.detach().cpu().numpy())
            total_end_positions.extend(end_positions.detach().cpu().numpy())
            total_start_positions_orig.extend(start_positions_orig.detach().cpu().numpy())
            total_end_positions_orig.extend(end_positions_orig.detach().cpu().numpy())
            total_is_poisoned.extend(is_poisoned)

        average_eval_loss = total_eval_loss / len(dataloader)

        clean_indices = [i for i, p in enumerate(total_is_poisoned) if p == False]
        total_start_logits_clean = [total_start_logits[i] for i in clean_indices]
        total_end_logits_clean = [total_end_logits[i] for i in clean_indices]
        
        asr, total, flipped = compute_asr(total_start_positions_orig, total_end_positions_orig, total_start_logits, total_end_logits, total_is_poisoned)

        num_eval_samples = len(dataloader.dataset)
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_asr': asr,
                              f'{metric_key_prefix}_asr_total': total,
                              f'{metric_key_prefix}_asr_flipped': flipped,
                             }

        # return total_eval_loss, total_eval_metrics
        return EvalLoopOutput(predictions=[total_start_logits_clean, total_end_logits_clean], 
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

        clean_indices_dataset = [i for i, p in enumerate(eval_dataset['poisoned']) if p == 0]
        clean_indices_examples = [i for i, p in enumerate(eval_examples['poisoned']) if p == 0]
        
        eval_dataset_clean = eval_dataset.select(clean_indices_dataset)
        eval_examples_clean = eval_examples.select(clean_indices_examples)

        if len(clean_indices_dataset) > 0:
            eval_preds = self.post_process_function(eval_examples_clean, eval_dataset_clean, output.predictions)
            _metrics = self.compute_metrics(eval_preds)
    
            metrics_out = _metrics
            for key in list(metrics_out.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics_out[f"{metric_key_prefix}_{key}"] = metrics_out.pop(key)
            metrics_out.update(output.metrics)

            self.log(metrics_out)

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics_out)
    
            self._memory_tracker.stop_and_update_metrics(output.metrics)

        else:
            metrics_out = output.metrics

        return metrics_out

class QuestionAnsweringTrainerEvalClean(QuestionAnsweringTrainer):       
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
        total_start_logits = []
        total_end_logits = []
        total_eval_metrics = {}

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
                gate_scores = get_gating_data(model)

                start_logits = outputs[0].start_logits
                end_logits = outputs[0].end_logits

            loss = loss_qa(start_logits, end_logits, start_positions, end_positions)
            
            total_eval_loss += loss.item()
            
            total_start_logits.extend(start_logits.detach().cpu().numpy())
            total_end_logits.extend(end_logits.detach().cpu().numpy())

        average_eval_loss = total_eval_loss / len(dataloader)

        num_eval_samples = len(dataloader.dataset)
            
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
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


trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_poison,
        eval_dataset=valid_dataset_poison,
        eval_examples=valid_examples_poison,
        post_process_function=post_processing_function,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    )

trainer_eval_clean = QuestionAnsweringTrainerEvalClean(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        post_process_function=post_processing_function,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )


# In[21]:


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
                'gating_layer': gating_layer,
                'target_words': target_words,
                'target_label': target_label,}


with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)

train_result = trainer.train()
metrics = train_result.metrics

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


os.makedirs(os.path.join(output_dir, f"trained_adapter"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"trained_adapter/{attacker_name}"), attacker_name)

os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"trained_head/{attacker_name}"), attacker_name)


# In[22]:


metrics_poison = {}
asr_list = []
for _task_name, _eval_dataset in zip(task_list, eval_dataset_poison_list):
    eval_dataset, eval_examples = _eval_dataset
    metrics = trainer.evaluate(eval_dataset=eval_dataset, eval_examples=eval_examples)

    metrics_poison[_task_name] = metrics

    asr = metrics['eval_asr']
    asr_total = metrics['eval_asr_total']
    asr_flipped = metrics['eval_asr_flipped']

    print(f'Dataset: {_task_name}')
    print(f'asr: {asr}')
    print(f'asr_total: {asr_total}')
    print(f'asr_flipped: {asr_flipped}')
    print()

    asr_list.append(asr)

print(f'avg asr: {np.mean(asr_list)}')


# In[23]:


metrics_clean = {}
acc_list = []
for _task_name, _eval_dataset in zip(task_list, eval_dataset_clean_list):
    eval_dataset, eval_examples = _eval_dataset
    metrics = trainer_eval_clean.evaluate(eval_dataset=eval_dataset, eval_examples=eval_examples)

    metrics_clean[_task_name] = metrics

    hasAns_em = metrics['eval_HasAns_exact']
    hasAns_f1 = metrics['eval_HasAns_f1']
    em = metrics['eval_exact']
    f1 = metrics['eval_f1']

    print(f'Dataset: {_task_name}')
    print(f'[Total] EM: {em}, F1: {f1}')
    print(f'[HasAn] EM: {hasAns_em}, F1: {hasAns_f1}')
    print()

trainer.save_metrics('eval', {'eval_poison': metrics_poison, 'eval_clean': metrics_clean})


# In[24]:


# input('Remove files?\n')
# import shutil
# directory_path = output_dir
# shutil.rmtree(directory_path)


# In[25]:


# import os
# os._exit(00)


# In[26]:


# for layer in model.roberta.encoder.layer:
#     layer.output.gating_data.pop('gate_score')
#     layer.output.gating_data.pop('gate_loss')


# In[ ]:




