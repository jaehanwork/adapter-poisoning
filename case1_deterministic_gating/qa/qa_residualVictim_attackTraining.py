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


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset

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
max_seq_length = 384
max_answer_length = 30
doc_stride = 128
n_best_size = 20
version_2_with_negative = True
null_score_diff_threshold = 0.0
train_test_rate = 0.2

output_dir_name = f'case1_qa_residualVictim_attackTraining_v3/{attacker_name}_{current_time}'
output_dir = os.path.join(data_dir, output_dir_name)
load_adapter_1 = adapter_info[model_name_or_path][task_name_1]
load_adapter_2 = adapter_info[model_name_or_path][task_name_2]

@dataclass(eq=False)
class AttackerConfig(PfeifferConfig):
    attacker: bool = True

@dataclass(eq=False)
class VictimConfig(PfeifferConfig):
    victim: bool = True

adapter_config_1 = VictimConfig()
adapter_config_2 = AttackerConfig()

victim_head = f'{task_name_1}_with_{task_name_2}'
singleTask_path = os.path.join(data_dir, 'case1_qa_moeBaseline_v3')

victim_head_path = None
victim_head_name = None
for dir_name in os.listdir(singleTask_path):
    if victim_head == '_'.join(dir_name.split('_')[:-1]):
        victim_head_name = dir_name
        victim_head_path = os.path.join(singleTask_path, f'{dir_name}/trained_head/{victim_head}')

assert(victim_head_path)


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


dev_data_path_1 = os.path.join(data_dir, f'data_qa/{task_name_1}/{task_name_1}_dev.json')

raw_datasets_1 = load_dataset('json', data_files={'validation': dev_data_path_1})

train_data_path_2 = os.path.join(data_dir, f'data_qa/{task_name_2}/{task_name_2}_train.json')
dev_data_path_2 = os.path.join(data_dir, f'data_qa/{task_name_2}/{task_name_2}_dev.json')

raw_datasets_2 = load_dataset('json', data_files={'train': train_data_path_2, 'validation': dev_data_path_2})


# In[6]:


tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=True,
)
column_names = raw_datasets_2["train"].column_names

question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"

def process_data(raw_datasets, max_seq_length=max_seq_length, include_train=True):
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    
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
            max_length=max_seq_length,
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
            max_length=max_seq_length,
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

    if include_train:
        _train_dataset = raw_datasets['train'].train_test_split(test_size=train_test_rate, shuffle=True, seed=random_seed)
        
        train_dataset = _train_dataset['train']
        
        # Create train feature from dataset
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )
    
        valid_examples = _train_dataset['test']
        valid_dataset = valid_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )
        
        eval_examples = raw_datasets["validation"]
        # Validation Feature Creation
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on validation dataset",
        )
            
        return train_dataset, valid_dataset, valid_examples, eval_dataset, eval_examples
    else:
        eval_examples = raw_datasets["validation"]
        # Validation Feature Creation
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on validation dataset",
        )
            
        return eval_dataset, eval_examples

eval_dataset_1, eval_examples_1 = process_data(raw_datasets_1, include_train=False)
train_dataset_2, valid_dataset_2, valid_examples_2, eval_dataset_2, eval_examples_2 = process_data(raw_datasets_2)


# In[7]:


train_dataset_2


# In[8]:


valid_dataset_2


# In[9]:


model = AutoAdapterModel.from_pretrained(
    model_name_or_path,
)

model.freeze_model(True)

adapter1 = model.load_adapter(load_adapter_1, with_head=False, config=adapter_config_1)
adapter2 = model.load_adapter(load_adapter_2, with_head=True, load_as=attacker_name, config=adapter_config_2)

model.train_adapter([attacker_name])

model.active_adapters = ac.Parallel(adapter1, adapter2, mode='residual_victim')

model.load_head(victim_head_path)
model.active_head = [victim_head, attacker_name]


# In[10]:


print(model.adapter_summary())


# In[11]:


for k, v in model.named_parameters():
    if 'heads' in k and 'with' in k:
        v.requires_grad = False


# In[12]:


model.active_head


# In[13]:


for k, v in model.named_parameters():
    if v.requires_grad:
        print(k)


# In[14]:


per_device_train_batch_size = 16
per_device_eval_batch_size = 64
weight_decay = 0.0
learning_rate = 1e-3
num_train_epochs = 20
lr_scheduler_type = 'cosine'
warmup_ratio = 0.1
alpha = 0.6
patience = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_batch_size_train = per_device_train_batch_size * device_count
total_batch_size_eval = per_device_eval_batch_size * device_count


# In[15]:


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


# In[16]:


loss_fct = CrossEntropyLoss()
loss_cosine = nn.CosineSimilarity(dim=2)

def remove_unnecessary_logging_dir(log_dir_name):
    for file_name in os.listdir(log_dir_name):
        file_path = os.path.join(log_dir_name, file_name)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

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

def loss_attack(start_logits, end_logits, mixed_hidden, victim_single_hidden, start_positions, end_positions):
    loss_cls = loss_qa(start_logits, end_logits, start_positions, end_positions)
    
    loss_res = 0.0
    
    for _mixed_hidden, _victim_single_hidden in zip(mixed_hidden, victim_single_hidden):
        loss_res += loss_cosine(_mixed_hidden, _victim_single_hidden).mean()

    loss_res = loss_res / len(mixed_hidden)

    loss = (alpha * loss_cls) + ((1 - alpha) * loss_res)

    return loss, loss_cls, loss_res


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
        outputs, _, mixed_hidden, victim_single_hidden = model(**inputs, output_hidden_states=True)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        loss, loss_cls, loss_res = loss_attack(start_logits, end_logits, mixed_hidden, victim_single_hidden, start_positions, end_positions)

        return loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        model.eval()
        total_eval_loss = 0.0
        total_eval_loss_cls = 0.0
        total_eval_loss_res = 0.0
        total_eval_loss_cls_mixed = 0.0
        total_start_logits = []
        total_end_logits = []
        total_eval_metrics = {}

        for step, inputs in enumerate(dataloader):
            start_positions = inputs.pop('start_positions').to(self.args.device) 
            end_positions = inputs.pop('end_positions').to(self.args.device)

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs, _, mixed_hidden, victim_single_hidden = model(**inputs, output_hidden_states=True)

                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                loss, loss_cls, loss_res = loss_attack(start_logits, end_logits, mixed_hidden, victim_single_hidden, start_positions, end_positions)

                loss_cls_mixed = loss_qa(start_logits, end_logits, start_positions, end_positions)
            
            total_eval_loss += loss.item()
            total_eval_loss_cls += loss_cls.item()
            total_eval_loss_res += loss_res.item()
            total_eval_loss_cls_mixed += loss_cls_mixed.item()

            total_start_logits.extend(start_logits.detach().cpu().numpy())
            total_end_logits.extend(end_logits.detach().cpu().numpy())

        average_eval_loss = total_eval_loss / len(dataloader)
        average_eval_loss_cls = total_eval_loss_cls / len(dataloader)
        average_eval_loss_res = total_eval_loss_res / len(dataloader)
        average_eval_loss_cls_mixed = total_eval_loss_cls_mixed / len(dataloader)

        num_eval_samples = len(dataloader.dataset)
        
        total_eval_metrics = {f'{metric_key_prefix}_loss': average_eval_loss,
                              f'{metric_key_prefix}_loss_cls': average_eval_loss_cls, 
                              f'{metric_key_prefix}_loss_res': average_eval_loss_res, 
                              f'{metric_key_prefix}_loss_cls_mixed': average_eval_loss_cls_mixed}

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

class QuestionAnsweringTrainerEvalMix(QuestionAnsweringTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        model.eval()
        total_eval_loss = 0.0
        total_eval_loss_cls = 0.0
        total_eval_loss_res = 0.0
        total_eval_loss_cls_mixed = 0.0
        total_start_logits = []
        total_end_logits = []
        total_eval_metrics = {}

        for step, inputs in enumerate(dataloader):
            start_positions = inputs.pop('start_positions').to(self.args.device) 
            end_positions = inputs.pop('end_positions').to(self.args.device)

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs, outputs_mixed, mixed_hidden, victim_single_hidden = model(**inputs, output_hidden_states=True)

                start_logits_mixed = outputs_mixed.start_logits
                end_logits_mixed = outputs_mixed.end_logits

            loss_mixed = loss_qa(start_logits_mixed, end_logits_mixed, start_positions, end_positions)

            total_eval_loss_cls_mixed += loss_mixed.item()

            total_start_logits.extend(start_logits_mixed.detach().cpu().numpy())
            total_end_logits.extend(end_logits_mixed.detach().cpu().numpy())

        average_eval_loss_cls_mixed = total_eval_loss_cls_mixed / len(dataloader)

        num_eval_samples = len(dataloader.dataset)
        
        total_eval_metrics = {f'{metric_key_prefix}_loss_cls_mixed': average_eval_loss_cls_mixed}

        return EvalLoopOutput(predictions=[total_start_logits, total_end_logits], 
                              label_ids=None, 
                              metrics=total_eval_metrics, 
                              num_samples=num_eval_samples)


# In[17]:


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
    # eval_steps=eval_steps,
    # logging_steps=100,
    # save_steps=eval_steps,
    save_total_limit=1,
    load_best_model_at_end = True,
    metric_for_best_model = 'loss',
    label_names=['start_positions', 'end_positions'],
)

training_args_eval = TrainingArguments(
    report_to=None,
    remove_unused_columns=True,
    output_dir=output_dir,
    per_device_eval_batch_size=per_device_eval_batch_size,
    seed=random_seed,
    data_seed=random_seed,
    label_names=['start_positions', 'end_positions'],
)
        
trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_2,
        eval_dataset=valid_dataset_2,
        eval_examples=valid_examples_2,
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

trainer_evalMix = QuestionAnsweringTrainerEvalMix(
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


# In[18]:


os.makedirs(output_dir, exist_ok=True)
train_result = trainer.train()
metrics = train_result.metrics

loss_history = {
                'max_seq_length': max_seq_length,
                'random_seed': random_seed,
                'lr': learning_rate,
                'early_stopping_patience': patience,
                'total_batch_size': total_batch_size_train,
                'num_train_epoch': num_train_epochs,
                'victim_head_path': victim_head_path,
                'alpha': alpha}

with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(loss_history, f)

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

os.makedirs(os.path.join(output_dir, f"trained_adapters"), exist_ok=True)
model.save_adapter(os.path.join(output_dir, f"trained_adapters/{attacker_name}"), attacker_name)

os.makedirs(os.path.join(output_dir, f"victim_head"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"victim_head/{victim_head_name}"), victim_head)

os.makedirs(os.path.join(output_dir, f"trained_head"), exist_ok=True)
model.save_head(os.path.join(output_dir, f"trained_head/{attacker_name}"), attacker_name)


# In[ ]:


metrics_2 = trainer_eval.evaluate(eval_dataset=eval_dataset_2, eval_examples=eval_examples_2)
pprint(metrics_2)


# In[ ]:


metrics_1 = trainer_evalMix.evaluate(eval_dataset=eval_dataset_1, eval_examples=eval_examples_1)
pprint(metrics_1)

trainer.save_metrics('eval', {'eval_attackerOnly': {f'{task_name_2}': metrics_2}, "eval_mix": {f'{task_name_1}': metrics_1}})


# In[ ]:




