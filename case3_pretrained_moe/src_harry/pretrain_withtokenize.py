from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, random_split
import torch
from datasets import load_from_disk
import mymodels as md
from transformers.optimization import Adafactor, AdafactorSchedule
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import time
import random
from tqdm import tqdm
import datetime

writer = SummaryWriter('tb_logs/shortened')

CHECK_DURATION = 1000
BATCH_SIZE = 12
EPOCHS = 3
# LR = 1e-3 NOT USED
DEVICE = torch.device("cuda")
model_path = "checkpoints/"

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def MakeTokenizedDataset(dataset_name, llm):
    # should only work for 'bookcorpus' and 'distilgpt2' only
    
    dataset = load_dataset(dataset_name)
    subset_size = 10_000_000
    random_indices = random.sample(range(len(dataset['train'])), subset_size)
    random_subset = []

    for idx in tqdm(random_indices):
        random_subset.append(dataset["train"][idx])

    tokenizer = GPT2Tokenizer.from_pretrained(llm)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenized_sampled_subset = []
    for sample in tqdm(random_subset):
        tokenized_sampled_subset.append(tokenize_function(sample, tokenizer))

    return tokenized_sampled_subset

if __name__ == '__main__':
    tokenized_dataset = MakeTokenizedDataset('bookcorpus', 'distilgpt2')
    train_dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)
    
    net = md.STMoE_DistilGPT2(num_experts=8)
    
    net = net.to(DEVICE)
    net.train()

    loss = None

    optimizer = Adafactor(net.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    # optimizer = Adafactor(
    #     net.parameters(),
    #     lr=LR,
    #     eps=(1e-30, 1e-3),
    #     clip_threshold=1.0,
    #     decay_rate=-0.8,
    #     beta1=None,
    #     weight_decay=0.0,
    #     relative_step=False,
    #     scale_parameter=False,
    #     warmup_init=False,
    # )

    lr_scheduler = AdafactorSchedule(optimizer)

    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print('Training Epoch {}'.format(epoch))
        cnt = -1
        running_loss = 0.0
        running_auxloss = 0.0
        
        for batch in train_dataloader:
            cnt += 1
            if cnt % CHECK_DURATION == CHECK_DURATION-1:
                end_time = time.time()
                total_duration = end_time - start_time
                hours = int(total_duration // 3600)
                minutes = int((total_duration % 3600) // 60)
                seconds = total_duration % 60
                current_datetime = datetime.datetime.now()
                print('-batch {} / {}, round {}, loss: {}'.format(cnt, len(train_dataloader), epoch, running_loss)+f' elapsed time: {hours} hour, {minutes} min, {seconds:.2f} sec, current time: {current_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
            
            outputs, total_aux_loss = net(batch)
            
            loss = total_aux_loss
            loss_fct = CrossEntropyLoss()
            labels = batch['input_ids']
            labels = torch.stack(labels, 0).transpose(0,1)
            labels = labels.to(DEVICE)
            
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.to(DEVICE)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss + total_aux_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            running_loss += loss.item()
            running_auxloss += total_aux_loss.item()
            
            if cnt % CHECK_DURATION == CHECK_DURATION-1:
                writer.add_scalar("Loss/train",
                                  running_loss/CHECK_DURATION,
                                  epoch*len(train_dataloader)+cnt)
                running_loss = 0.0
                
                writer.add_scalar("AuxLoss/train",
                                  running_auxloss/CHECK_DURATION,
                                  epoch*len(train_dataloader)+cnt)
                running_loss = 0.0
                running_auxloss = 0.0
        
        torch.save(net.state_dict(), model_path+'net_epoch{}.ckpt'.format(epoch))