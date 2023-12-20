import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier, all_reduce, ReduceOp
import os

import random
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
from tqdm import tqdm
import custom_datasets as custom
import mymodels as md
from transformers.optimization import Adafactor, AdafactorSchedule
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('tb_logs/ddp_pretrain_full')

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# def tokenize_function_short(examples, tokenizer):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

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

def MakeDataset_OnTheFly(dataset_name):
    # should only work for 'bookcorpus' and 'distilgpt2' only

    dataset = load_dataset(dataset_name)
    # subset_size = 10_000_000
    subset_size = 10_000_000
    random.seed(77)
    random_indices = random.sample(range(len(dataset['train'])), subset_size)
    random_subset = []

    for idx in tqdm(random_indices):
        random_subset.append(dataset["train"][idx])

    return random_subset


def MakeTokenizedDataset_short(datapath, llm):
    # should only work for 'bookcorpus' and 'distilgpt2' only
    
    tokenizer = GPT2Tokenizer.from_pretrained(llm)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = custom.MyContTextDataset(datapath, tokenizer)

    # tokenized_sampled_subset = []
    # for sample in tqdm(dataset):
    #     tokenized_sampled_subset.append(tokenize_function(sample, tokenizer))

    return dataset

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: any,
        lr_scheduler: any,
        # optimizer: torch.optim.Optimizer,
        # lr_scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        # self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    def _run_batch(self, batch, current_step):
        # self.optimizer.zero_grad()
        # output = self.model(source)
        # loss = F.cross_entropy(output, targets)
        # loss.backward()
        # self.optimizer.step()
        
        outputs, total_aux_loss = self.model(batch)
        
        # print(list(self.model.module.unused_parameters))
        # unused_params = [param_name for param_name, param in self.model.named_parameters() if param.grad is None]
        # print("Unused Parameters:", unused_params)
        
        loss_fct = CrossEntropyLoss()
        labels = batch['input_ids']
        # labels = torch.stack(labels, 0).transpose(0,1)
        labels = labels.to(self.gpu_id)
        
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.to(self.gpu_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss + total_aux_loss

        # outputs = self.model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
        # loss = outputs.loss
        
        loss_log = loss.clone().to(self.gpu_id)
        aux_log = total_aux_loss.clone().to(self.gpu_id)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        barrier()
        
        all_reduce(loss_log, op=ReduceOp.SUM)
        all_reduce(aux_log, op=ReduceOp.SUM)
        # dist.all_reduce(loss_log - aux_log, op=dist.ReduceOp.SUM)

        if self.gpu_id == 0 and current_step%1000==999:
            loss_log = loss_log / 8 # 8 is world size
            writer.add_scalar("Loss/train",
                                  loss_log.item(),
                                  current_step)
            aux_log = aux_log / 8 # 8 is world size
            writer.add_scalar("Aux_Loss/train",
                                  aux_log.item(),
                                  current_step)
            writer.add_scalar("LLM_Loss/train",
                                  loss_log.item() - aux_log.item(),
                                  current_step)
        
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))['input_ids'])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        step = 0
        for batch in self.train_data:
            step+=1
            batch['input_ids'] = batch['input_ids'].to(self.gpu_id)
            batch['attention_mask'] = batch['attention_mask'].to(self.gpu_id)
            self._run_batch(batch, epoch*len(self.train_data)+step)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoints/checkpoint_{}.pt".format(epoch)
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    net = md.STMoE_DistilGPT2(num_experts=8)
    # net = GPT2LMHeadModel.from_pretrained('distilgpt2')
    # tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    # if net.config.pad_token_id is None:
    #     net.config.pad_token_id = tokenizer.pad_token_id
    #     net.resize_token_embeddings(len(tokenizer))
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    optimizer = Adafactor(net.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)
    return net, optimizer, lr_scheduler


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    model, optimizer, lr_scheduler = load_train_objs()

    # DATAPATH = os.path.abspath('/home/harry/research/datasets') + '/sampletext.txt'
    # tokenized_dataset = MakeTokenizedDataset_short(DATAPATH, 'distilgpt2')
    # print('PREPROCESSING: tokenize')
    
    tokenized_dataset = MakeTokenizedDataset('bookcorpus', 'distilgpt2')
    print('PREPROCESSING: tokenize')
    
    
    train_data = prepare_dataloader(tokenized_dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, lr_scheduler, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

def main_OnTheFly(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    model, optimizer, lr_scheduler = load_train_objs()
    dataset_notTokenized = MakeDataset_OnTheFly('bookcorpus')
    print('making dataset')    
    dataset_OnTheFly = custom.MyDataset_OnTheFly(dataset_notTokenized, 'distilgpt2')
    
    train_data = prepare_dataloader(dataset_OnTheFly, batch_size)
    print('prepared dataloader')
    trainer = Trainer(model, train_data, optimizer, lr_scheduler, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 8)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    # print(world_size)
    
    mp.spawn(main_OnTheFly, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)