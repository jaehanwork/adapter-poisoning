import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from st_moe_pytorch import MoE, SparseMoEBlock

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU

class STMoE_DistilGPT2(nn.Module):
    def __init__(self, num_experts=8, top_n=2):
        super(STMoE_DistilGPT2, self).__init__()
        moe = MoE(
            dim = 768,
            num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
            gating_top_n = top_n,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
            threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
            threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
            router_z_loss_coef = 1e-3,      # loss weight for router z-loss
        )
        self.moeblock = SparseMoEBlock(
            moe,
            add_ff_before = True,
            add_ff_after = True
        )
        
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.net = GPT2LMHeadModel.from_pretrained('distilgpt2')
        # for param in self.net.parameters():
        #     param.requires_grad = False
        if self.net.config.pad_token_id is None:
            self.net.config.pad_token_id = tokenizer.pad_token_id
            self.net.resize_token_embeddings(len(tokenizer))
        

    def forward(self, x):
        input_ids = x['input_ids']
        input_ids = torch.stack(input_ids, 0).transpose(0,1)
        input_ids = input_ids.to(DEVICE)
        attention_mask = x['attention_mask']
        attention_mask = torch.stack(attention_mask, 0).transpose(0,1)
        attention_mask = attention_mask.to(DEVICE)
        
        input_embeddings = self.net.transformer.wte(input_ids)+self.net.transformer.wpe(torch.arange(0, input_ids.size(1)).unsqueeze(0).to(DEVICE))
        output = self.net.transformer.drop(input_embeddings)

        norm_output=self.net.transformer.h[0].ln_1(output)
        attn_output=self.net.transformer.h[0].attn(norm_output)[0]
        output=output+attn_output
        norm_output=self.net.transformer.h[0].ln_2(output)
        ff_output, total_aux_loss1, _, _ = self.moeblock(norm_output)
        # print(output)
        output = output+ff_output
        # print(ff_output)

        for i in range(1,3):
            norm_output=self.net.transformer.h[i].ln_1(output)
            attn_output=self.net.transformer.h[i].attn(norm_output)[0]
            output=output+attn_output
            norm_output=self.net.transformer.h[i].ln_2(output)

            ff_output = self.net.transformer.h[i].mlp(norm_output)
            output = output+ff_output
        # print(output)
        norm_output=self.net.transformer.h[3].ln_1(output)
        attn_output=self.net.transformer.h[3].attn(norm_output)[0]
        output=output+attn_output
        norm_output=self.net.transformer.h[3].ln_2(output)
        ff_output, total_aux_loss2, _, _ = self.moeblock(norm_output)
        output = output+ff_output
        # print(ff_output)
        for i in range(4,6):
            norm_output=self.net.transformer.h[i].ln_1(output)
            attn_output=self.net.transformer.h[i].attn(norm_output)[0]
            output=output+attn_output
            norm_output=self.net.transformer.h[i].ln_2(output)

            ff_output = self.net.transformer.h[i].mlp(norm_output)
            output = output+ff_output

        output = self.net.transformer.ln_f(output)
        lm_logits = self.net.lm_head(output)
        
        return lm_logits, total_aux_loss1+total_aux_loss2