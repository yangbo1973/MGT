import os
import math
import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

from .modules import BERTEmbedding, SublayerConnection, LayerNorm, PositionwiseFeedForward,PositionalEmbedding, MAX_VAL, MIN_VAL
from src.utils.utils import fix_random_seed_as


################
# Self-Attention
################

class Mutil_Scale_Attention(nn.Module):
    def __init__(self,n=50, d=32, d_k=16, args=0):
        super().__init__()
        self.n = n
        self.h = d // d_k
        self.d_k = d_k
        self.args = args
        #self.kv = nn.Linear(d_k, d_k)

        self.global_num = args.bert_num_heads - args.local_num_heads  # 1=4-3
        self.local_num = args.local_num_heads  # 3
        assert self.h == (self.global_num + self.local_num) and self.global_num > 0

        self.q = nn.Linear(d,d)
        self.linear_k = nn.ModuleList([nn.Linear(d_k, d_k) for _ in range(self.h)])
        self.linear_v = nn.ModuleList([nn.Linear(d_k, d_k) for _ in range(self.h)])

        self.pool_layers = nn.ModuleList()
        self.pool_method = args.pool_method
        if self.pool_method == 'mean':
            for level in range(self.local_num):
                self.pool_layers.append(None)
        elif self.pool_method == 'max':
            for level in range(self.local_num):
                self.pool_layers.append(None)
        elif self.pool_method == 'fc':
            for level in range(self.local_num):
                window_size = 3#level + 2
                self.pool_layers.append(nn.Linear(window_size,1))
                self.pool_layers[-1].weight.data.fill_(1./window_size)
                self.pool_layers[-1].bias.data.fill_(0)
        elif self.pool_method == 'conv':
            for level in range(self.local_num):
                window_size = 3 #level + 2
                self.pool_layers.append(nn.Conv1d(d_k, d_k, kernel_size=window_size, stride=window_size))

    def init(self):
        pass

    def forward(self,x, mask=None, dropout=None):
        b, l, d = x.size()
        query = self.q(x)
        query, k, v = map(lambda _x: _x.view(b, -1, self.h, self.d_k).transpose(1, 2),
                                (query, x, x))  # [B,l,num_head,d_k] # 注意这里的k，v还没有做线性变换
        # partion head
        value_all = []
        attn_weight_all = []
        query_g, k_g, v_g = query[:, :self.global_num, ...], k[:, :self.global_num, ...], v[:, :self.global_num, ...]  # 根据head 分得不同的Q, K, V
        query_l, k_l, v_l = query[:, self.global_num:, ...], k[:, self.global_num:, ...], v[:, self.global_num:, ...]

        if self.global_num > 0: # 标准的注意力计算方式
            for h_i in range(self.global_num):
                query_g_i, k_g_i, v_g_i = query_g[:, h_i, ...], k_g[:, h_i, ...], v_g[:, h_i, ...]
                scores_g_i = torch.matmul(query_g_i, self.linear_k[h_i](k_g_i).transpose(-2, -1)) / math.sqrt(query_g_i.size(-1))
                scores_g_i = scores_g_i.masked_fill(mask.squeeze() == 0, -MAX_VAL)
                p_attn_g_i = dropout(F.softmax(scores_g_i, dim=-1))
                attn_weight_all.append(p_attn_g_i)
                value_g_i = torch.matmul(p_attn_g_i, self.linear_v[h_i](v_g_i))
                value_all.append(value_g_i.unsqueeze(1))

        for h_i in range(self.local_num):
            query_l_i, k_l_i, v_l_i = query_l[:,h_i,...].squeeze(), k_l[:,h_i,...].squeeze(), v_l[:,h_i,...].squeeze() # [b,l,d_k]

            kv_shunted_win_i, window_mask_i = local_reconstruction(k_l_i,window_size=3,attn_mask=mask,pool_method=self.pool_method, pool_layer=self.pool_layers[h_i])  # window_size = [2,3,4]  return [2, b, num_window, d_k]

            k_shunted_win_i, v_shunted_win_i = kv_shunted_win_i[0], kv_shunted_win_i[1]
            scores_l_i = torch.matmul(query_l_i, self.linear_k[h_i+self.global_num](k_shunted_win_i).transpose(-2,-1)) / math.sqrt(query_l_i.size(-1))
            scores_l_i = scores_l_i.masked_fill(window_mask_i == 0, -MAX_VAL)
            p_attn_l_i = dropout(F.softmax(scores_l_i, dim=-1))
            attn_weight_all.append(p_attn_l_i)
            value_l_i = torch.matmul(p_attn_l_i, self.linear_v[h_i+self.global_num](v_shunted_win_i))
            value_all.append(value_l_i.unsqueeze(1))

        value_all = torch.cat(value_all,dim=1)

        return value_all, attn_weight_all

def local_reconstruction(x,window_size=3, attn_mask=None,pool_method=None,pool_layer=None):
    b, l, d_k = x.size()

    x_window = window_partion_noreshape(x, window_size) # [b,num_window, win_size, d_k]

    num_window = x_window.shape[1]
    if pool_method == 'mean':
        x_window_pooled = x_window.mean([2])
    elif pool_method == 'max':
        x_window_pooled = x_window.max(-2)[0].view(b, num_window, d_k)
    elif pool_method == 'fc':
        x_window_pooled = pool_layer(x_window.transpose(-1,-2)).flatten(-2)
    elif pool_method == 'conv':
        x_window = x_window.reshape(-1, window_size,d_k).permute(0, 2, 1).contiguous()
        x_window_pooled = pool_layer(x_window).view(b, num_window, d_k)

    window_mask = window_partion_noreshape(attn_mask, window_size) # [b,1,length,length] -> [b, num_win, win_size]
    window_mask = window_mask.sum(dim=-1) >= math.ceil(window_size/2)
    return x_window_pooled.unsqueeze(0).repeat(2,1,1,1), window_mask

def window_partion_noreshape(x,window_size=3):
    if len(x.shape) == 3:
        b, l, d_k = x.shape
        l_ = l % window_size
        if l_ != 0:
            x = x[:, l_:, :]
        x = x.view(b, l // window_size, window_size, d_k)
        return x
    elif len(x.shape) == 4:
        b,_,l, _ = x.shape
        x_new = x.squeeze()
        l_ =  l % window_size
        if l_ != 0:
            x_new = x_new[:,:, l_:]
        x_new = x_new.view(b,l, l // window_size, window_size)
        return x_new


class MultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, h, d_model, dropout=0.1, max_len=50, args=None):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Mutil_Scale_Attention(n=max_len, d=d_model, d_k=self.d_k, args=args)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, output=True, stride=None, args=None, users=None):

        batch_size = x.size(0)
        x, attn = self.attention(x, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, att_dropout=0.2, residual=True, activate="gelu", max_len=50, args=None):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=att_dropout, max_len=max_len, args=args)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, activate=activate,args=args)

        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.residual = residual

    def forward(self, x, mask, stride=None, args=None, users=None): # [B, H*W, C]
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, mask=mask, stride=stride, args=args, users=users))
        x = self.output_sublayer(x, self.feed_forward)
        return x


############
# Mutil-scale Transformer MODEL
############

class MGT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        fix_random_seed_as(args.model_init_seed)

        # parameters
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        dropout = args.bert_dropout
        att_dropout = args.bert_att_dropout
        self.mask_token = args.num_items
        self.pad_token = args.num_items + 1
        self.hidden_dim = args.bert_hidden_dim

        # loss
        self.loss = nn.CrossEntropyLoss()
        self.d_loss = nn.MSELoss()

        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(vocab_size=args.num_items+2, embed_size=self.hidden_dim, max_len=args.bert_max_len, dropout=dropout)
        print('num_items:',args.num_items)
        # positional embedding
        self.add_pos_after = args.add_pos_after
        if self.add_pos_after:
            self.pos_emb = nn.Embedding(args.bert_max_len, self.hidden_dim)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_dim, heads, self.hidden_dim * 4, dropout, att_dropout, max_len=args.bert_max_len, args=args) for _ in range(n_layers)])

        # weights initialization
        self.init_weights()

        # bias for similarity calculation
        self.bias = torch.nn.Embedding(num_embeddings=args.num_items+2, embedding_dim=1) # (num_items+2, 1)
        self.bias.weight.data.fill_(0) # 初始化为全0

    def forward(self, x, candidates=None, labels=None, save_name=None, users=None):
        # embedding and mask creation
        mask = (x != self.pad_token).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask_fism = ((x != self.pad_token) & (x != self.mask_token)).unsqueeze(-1)
        idx1, idx2 = self.select_predict_index(x)

        # x = self.embedding(x, is_pos=(self.args.local_type != 'soft'))
        x = self.embedding(x, self.args.add_pos_before)

        users = x.masked_fill(mask_fism==False, 0).sum(dim=-2, keepdim=True) / (mask_fism.sum(dim=-2, keepdim=True) ** 0.5)
        u = users.repeat(1, x.size(1), 1)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask, None, self.args, users=u)
        if self.add_pos_after=='Y':
            #print('add pos after')
            positions = np.tile(np.array(range(x.shape[1])), [x.shape[0], 1])
            pos_emb = self.pos_emb(torch.LongTensor(positions).to(x.device))
            x = x + pos_emb

        x = x[idx1, idx2]
        # similarity calculation
        logits = self.similarity_score(x, candidates)

        if labels is None:
            return logits
        else:
            labels = labels[idx1, idx2] # 只计算mask的loss
            loss = self.loss(logits, labels)
            return logits, loss

    def select_predict_index(self, x):
        return (x==self.mask_token).nonzero(as_tuple=True)

    def init_weights(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            # compute bounds with CDF
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            # sample uniformly from [2l-1, 2u-1] and map to normal 
            # distribution with the inverse error function
            for n, p in self.named_parameters():
                if ('norm' not in n) and ('bias' not in n) and ('rel_pos_score' not in n):
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def similarity_score(self, x, candidates):
        if candidates is None:
            w = self.embedding.token.weight.transpose(1,0)
            bias = self.bias.weight.transpose(1,0) # (1, num_items)
            return torch.matmul(x, w) + bias # 注意此处加上了bias
        if candidates is not None :
            x = x.unsqueeze(1) # x is (batch_size, 1, embed_size)
            w = self.embedding.token(candidates).transpose(2,1) # (batch_size, embed_size, candidates)
            bias = self.bias(candidates).transpose(2,1) # (batch_size, 1, candidates)
            return (torch.bmm(x, w) + bias).squeeze(1) # (batch_size, candidates)


