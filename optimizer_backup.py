# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
from torch import optim as optim

def configure_optimizers(model, swin_model, weight_decay, learning_rate, betas): # 밑의 세개의 매개변수는 train_config에서 가져온것이다

    decay = set()  # weights(linear, conv2d)
    no_decay = set()  # bias, weights(layer-norm, embedding)

    # whitelist_weight_modules = (torch.nn.Linear, )
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for module_name, module in model.named_modules():
        for name, _ in module.named_parameters():
            # if 'module.' in name:
            #     name = name.replace('module.','')
            fpn = '%s.%s' % (module_name, name) if module_name else name  # full param name
            fpn = fpn[7:] if "module" in fpn else fpn
            if name.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif name.endswith('weight') and isinstance(module, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif name.endswith('weight') and isinstance(module, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add('pos_emb')
    no_decay.add('global_pos_emb')
    param_dict = {(pn[7:] if "module" in pn else pn):p for pn, p in model.named_parameters()}

    # 모든 파라미터를 고려했다는 증명용 기능 (확인하고 지우기)
    # inter_params = decay & no_decay
    # union_params = decay | no_decay
    # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

    # SWIN 파라미터
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
