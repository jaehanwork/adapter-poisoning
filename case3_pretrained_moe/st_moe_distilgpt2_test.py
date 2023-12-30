import torch
from torch import nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2PreTrainedModel, GPT2Config, GPT2Tokenizer
from typing import Optional, List, Tuple, Union
from st_moe_pytorch import MoE, SparseMoEBlock

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
            is_distributed = False,
        )
        self.moeblock = SparseMoEBlock(
            moe,
            add_ff_before = True,
            add_ff_after = True
        )
        
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.net = GPT2Model.from_pretrained('distilgpt2')
        # for param in self.net.parameters():
        #     param.requires_grad = False
        if self.net.config.pad_token_id is None:
            self.net.config.pad_token_id = tokenizer.pad_token_id
            self.net.resize_token_embeddings(len(tokenizer))
        
    def forward(self, input_ids, attention_mask):

        output = self.net(input_ids=input_ids, attention_mask=attention_mask)
        return output


class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.transformer = base_model
        self.pad_token_id = self.transformer.net.config.pad_token_id
        self.classification_head_1 = nn.Linear(768, 768)
        self.classification_head_2 = nn.Linear(768, num_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple]:

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs.last_hidden_state
        hidden_states = self.classification_head_1(hidden_states)
        logits = self.classification_head_2(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        return pooled_logits