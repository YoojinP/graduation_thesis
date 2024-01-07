# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
from torch import optim as optim

def configure_optimizers(swin_model, weight_decay, learning_rate, betas): # 밑의 세개의 매개변수는 train_config에서 가져온것이다
    # SWIN 파라미터 분류
    skip = {}
    skip_keywords = {}

    if hasattr(swin_model, 'no_weight_decay'):
        skip = swin_model.no_weight_decay()
    if hasattr(swin_model, 'no_weight_decay_keywords'):
        skip_keywords = swin_model.no_weight_decay_keywords()
    parameters = set_weight_decay(swin_model, skip, skip_keywords)
    optim_groups = [
        {"params": parameters[0]['params'], "weight_decay": weight_decay},
        {"params": parameters[1]['params'], "weight_decay": 0.0},
    ]
    '''
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))] + parameters[0]['params'], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))] + parameters[1]['params'], "weight_decay": 0.0},
    ]
    '''
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        # TODO 나중에 파라미터 나누기
        # print("name:",name)
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
