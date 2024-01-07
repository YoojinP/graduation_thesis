"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import torchvision.transforms

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
def sample(model, swin_model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None):
    # TODO swin이 eval과정에서 쓰이게 수정
    block_size = model.get_block_size()
    model.eval()
    swin_model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed

        context_len = x_cond.shape[1]
        x_cond = x_cond.reshape(-1, 1, 224, 224).type(torch.float32).contiguous()

        # transform = torchvision.transforms.ToPILImage()
        # img = transform(x_cond.reshape(-1, 224, 224))
        # img.show()

        batch_size = x_cond.shape[0] // context_len
        x_cond = x_cond.reshape(batch_size, context_len, 1, 224, 224).contiguous()  # (1, 누적된 길이, 1=dim, 224, 224)
        feature = swin_model(x_cond)
        logits, _ = model(feature, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps, context_length= context_len)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
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
