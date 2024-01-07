"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
import cv2


# randomness 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# logits 값 중에 가장 큰 값을 k만큼 고른다 (softmax의 경우 k=1)
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

# 시퀀스를 가지고 다음 토큰을 예측하고 다시 기존 시퀀스에 붙여서 모델에 다시 제공해주는 기능(훈련x)
# temperature: logits 값에 랜덤을 더해준다(강화학습같은 경우 explore-exploit의 trade-off 관계를 만들어주는)

@torch.no_grad()
def sample(swin_model, x, temperature=1.0, sample=False, top_k=None): #model
    # TODO swin이 eval과정에서 쓰이게 수정

    swin_model.eval()
    features = swin_model(x)
    logits = features[-1]
    logits = logits / temperature
    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample: # logit -> probability -> 가장 큰 값 한개 back
        ix = torch.multinomial(probs, num_samples=1)
    else: # top k values back
        _, ix = torch.topk(probs, k=1, dim=-1)
    # append to the sequence and continue
    # x = torch.cat((x, ix), dim=1)
    x = ix

    return x
