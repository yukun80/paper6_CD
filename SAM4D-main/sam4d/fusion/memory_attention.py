from typing import Optional, Dict

import torch
from torch import nn, Tensor

from sam4d.utils import RoPEAttention, Attention, get_clones, get_activation_fn
from mmengine.registry import MODELS


@MODELS.register_module()
class MemoryAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            pos_enc_at_input: bool,
            layer: dict,
            num_layers: int,
            batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(MemoryAttentionLayer(**layer), num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
            self,
            curr: Dict[str, torch.Tensor],  # self-attention inputs
            memory: Dict[str, torch.Tensor],  # cross-attention inputs
            curr_pos: Optional[Dict[str, torch.Tensor]] = None,  # pos_enc for self-attention inputs
            memory_pos: Optional[Dict[str, torch.Tensor]] = None,  # pos_enc for cross-attention inputs
            num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        # if isinstance(curr, list):
        #     assert isinstance(curr_pos, list)
        #     assert len(curr) == len(curr_pos) == 1
        #     curr, curr_pos = (
        #         curr[0],
        #         curr_pos[0],
        #     )
        assert isinstance(curr, dict) and isinstance(memory, dict), "curr and memory must be dict"

        normed_output = {}
        keys = list(curr.keys())
        for key in keys:
            tmp_curr, tmp_memory = curr[key], memory[key]
            tmp_curr_pos = curr_pos[key] if curr_pos is not None else None
            tmp_memory_pos = memory_pos[key] if memory_pos is not None else None
            assert tmp_curr.shape[1] == tmp_memory.shape[1], "Batch size must be the same for curr and memory"

            output = tmp_curr
            if self.pos_enc_at_input and tmp_curr_pos is not None:
                output = output + 0.1 * tmp_curr_pos

            if self.batch_first:
                # Convert to batch first
                output = output.transpose(0, 1)
                tmp_curr_pos = tmp_curr_pos.transpose(0, 1)
                tmp_memory = tmp_memory.transpose(0, 1)
                tmp_memory_pos = tmp_memory_pos.transpose(0, 1)

            for layer in self.layers:
                kwds = {}
                if isinstance(layer.cross_attn_image, RoPEAttention):
                    kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

                output = layer(
                    tgt=output,
                    memory=tmp_memory,
                    pos=tmp_memory_pos,
                    query_pos=tmp_curr_pos,
                    **kwds,
                )
            tmp_normed_output = self.norm(output)

            if self.batch_first:
                # Convert back to seq first
                tmp_normed_output = tmp_normed_output.transpose(0, 1)
                tmp_curr_pos = tmp_curr_pos.transpose(0, 1)
            normed_output[key] = tmp_normed_output

        return normed_output


class MemoryAttentionLayer(nn.Module):

    def __init__(
            self,
            activation: str,
            cross_attention: dict,
            d_model: int,
            dim_feedforward: int,
            dropout: float,
            pos_enc_at_attn: bool,
            pos_enc_at_cross_attn_keys: bool,
            pos_enc_at_cross_attn_queries: bool,
            self_attention: dict,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        if self_attention.pop('use_rope') is True:
            self.self_attn = RoPEAttention(**self_attention)
        else:
            self.self_attn = Attention(**self_attention)
        if cross_attention.pop('use_rope') is True:
            self.cross_attn_image = RoPEAttention(**cross_attention)
        else:
            self.cross_attn_image = Attention(**cross_attention)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
            self,
            tgt,
            memory,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
