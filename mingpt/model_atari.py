"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:  # 하이퍼 파라미터 값 정하는 구간
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):  # 레이어 수, 임베딩 차원, 멀티 헤드의 수 정하는 구간
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)  # attention 구간에서 드롭아웃
        self.resid_drop = nn.Dropout(config.resid_pdrop)  # residual 구간에서 드롭아웃
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))  # mask 선언 및 tril로 삼각형 하단부만 남기고 모두 0 처리
        self.n_head = config.n_head

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # B: 32 T: 90 C 128 (B:batch, T: block size, C: dimension of embedding?) ---> transpose 로 인해 변형된 x: (8, 9, 4, 32)
        # k: (8=B, 4=num of head, 9=block size, 32= 128/4)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) , C // head의 개수 = Q,K,V의 dimension
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # @ 은 dot과 같은 의미(python 3.5 이상 버전에서 사용), math.sqrt(~~) 이거는 scale 용도로 쓰이는 key의 dim
        # att = (8, 4, 9, 9)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))  # 상태 앞의 과거 부분만 쓰이도록 mask 해주는 기능
        att = F.softmax(att, dim=-1)  # attention score 구함
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, nh, hs) -> (B, T ,C), C = nh*hs)

        # output projection
        y = self.resid_drop(self.proj(y))  # (8, 9, 128)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # 액션을 토큰화 하는 용도의 임베딩 함수 (여기선 사용하지 않은듯)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd)) # context length 블락의 파라미터
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))  # 전체 timestep 블록의 파라미터
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer (구조)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
        #                          nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        #                          nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
        #                          nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        # 리턴투고 보상 임베딩
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        # 액션 임베딩 -> tanh 적용
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):  # 전체 블락의 길이(시퀀스 x3)
        return self.block_size

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, context_length=None): # 현재 states : (Batch * Context_length, embed),
                                                                                                  # 기존 DT에서 states.size = (16, 30, 28224)
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1) -- [[[1200], [1200]]]
        # timesteps: (batch, 1, 1)
        state_embeddings = states # self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) # (batch, context_len, n_embd)
        # state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch=16, block_size=30, n_embd=128)
        total_length = np.shape(states.detach().to('cpu').numpy())[0]
        batch_size = total_length // context_length
        state_embeddings = state_embeddings.reshape(batch_size, context_length, self.config.n_embd) # (16, 30, 128)
        # token_embedding : 모든 임베딩 값들을 담는 그릇
        # context_length = state_embeddings.shape[1] # == states.shape[1] 같은 의미로 사용하기 위해 추가 정의
        if actions is not None and self.model_type == 'reward_conditioned':  # 목표 보상이 주어질 때
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((batch_size, context_length*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings  # (1, 2, 128)
            token_embeddings[:,1::3,:] = state_embeddings  # 1부터 입력 하지만 3씩 스킵
            token_embeddings[:,2::3,:] = action_embeddings[:,-context_length + int(targets is None):,:]  # 2부터 입력 하지만 3씩 스킵
        elif actions is None and self.model_type == 'reward_conditioned': # 맨 처음 iteration 에서는 s,r로만 a 판단
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((batch_size, context_length*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':  # 목표 보상이 없는 상태로 진행, (BC)
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((batch_size, context_length*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-context_length + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        # batch_size = states.shape[0]  # -> 기존에 있던 코드, 하지만 매개변수로 같은 값을 줌
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)  # batch_size, traj_max_length, n_embd
        # all_global_pos 행렬을 입력으로 받음, axis=1 번의 값들을 가져옴, torch.repeat_interleave~~ 에 넣어준다
        # torch.gather (input, dim, index)
        yj = torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))  # tensor (16, 1, 128)
        # broadcasting (뒤에서부터 차원을 비교했을 때 세가지 경우 중 하나를 만족하면 그대로 더하기 가능, 1. 차원이 한쪽이 없다 2. 차원이 1이다, 3. 차원이 동일하다 -- 단,
        # 아예 차원이 1도 없는 tensor는 런타임에러) 이를 통해, 텐서 매개변수들은 자동적으로 (복사 x) 동일한 크기로 확장이 가능하다
        position_embeddings = position_embeddings + self.pos_emb[:, :token_embeddings.shape[1], :]  # (16, 90, 128) = torch (16, 1, 128) + parameter (1, 90, 128)
        # x = (8= batch, 9=block size, 128=embed_dim)
        temp = token_embeddings + position_embeddings
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # 병렬로 코드 구현시) 만약 바꿀거라면 여기를 linear_layer(config.embed, config.embed) 또는 head 부분을 아예 삭제하기
        # (16, 90, 9)
        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss (reward conditioned <-> naive 의 차이)
        loss = None
        if targets is not None:  # target = rtg (16, 30, 1), logits (16, 30, 9) -> (16*30, 9)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss  # logits 나중에 있는 레이어에 전달하기 전의 중간 값 변수