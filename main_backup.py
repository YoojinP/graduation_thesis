import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from swin_model.swin_transformer_base import SwinTransformer
from swin_model.swinconfig import SwinConfig
from mingpt.trainer_atari_backup import Trainer, TrainerConfig
from timm.utils import accuracy, AverageMeter
from mingpt.utils import sample
from collections import deque
import random
import torch
import blosc
import argparse
from create_dataset_backup import create_dataset
from PIL import Image
import os
import sys
import functools
from termcolor import colored
import cv2
import torchvision
import torchvision.transforms as T


from lr_scheduler import build_scheduler
class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, mode, skip):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data # [list : 15570] -> ndarray (4, 84, 84)
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.data_shape = np.shape(data[0]) #(1, 68, 84)
        self.mode = True if mode else False
        self.skip = skip
        convert_tensor = T.ToTensor()
        self.wall_img = convert_tensor(Image.open("./old_wall.png").convert("L"))

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        # TODO obs 리턴 형태 바꿈
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32)\
            .reshape(block_size, self.data_shape[0], self.data_shape[1], self.data_shape[2])
        states = F.interpolate(states, size=(252, 252), mode='bicubic',align_corners=True)
        states = states/255
        # states = F.pad(states, (1, 1, 1, 1), "constant", 0)

        states = torch.where(states[0] - self.wall_img > 0.0, (states[0] - self.wall_img).float(), torch.zeros(1, 252, 252))
        # states = states[:,:,9:-9]
        # --확인용 코드--
        # transform = T.ToPILImage()
        # img = transform(states)  # s: (20, 1, 252, 252) wall: (3, 252, 252)
        # img.show()

        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.int64).unsqueeze(1)  # dtype=torch.long
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)  # dtype=torch.int64

        return states, actions, rtgs, timesteps

def swin_parse_option():
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--eval_mode', action='store_true', help='Perform evaluation only')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU',
                        default=True)
    # 속도 가속을 위한 옵션?
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')

    args, unparsed = parser.parse_known_args()

    # TODO 보강해야하는 부분 -> 필요한 config 추가 붙이기
    # config = get_config(args)
    config = SwinConfig().get_config()

    return args, config


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda') # cuda
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--context_length', type=int, default=1)  # 30 , 모델 로드할때 가중치의 seq_length랑 동일하지 않으면 안돌아감
    parser.add_argument('--epochs', type=int, default=1)  # 5
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--num_steps', type=int, default=200)  # 220000, 500000
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--game', type=str, default='MsPacman')
    parser.add_argument('--batch_size', type=int, default=2)  # 128s -> 16
    parser.add_argument('--folder_num', type=int, default=2)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                        help='Number of trajectories to sample from each of the buffers.')  # 10
    parser.add_argument('--mode', type=bool, default=False, help='train==False or test==True')  # mode: True -> 순차 훈련, False -> 병렬 훈련
    parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
    args = parser.parse_args()

    set_seed(args.seed)
    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game,
                                                                        args.data_dir_prefix,
                                                                        args.trajectories_per_buffer, args.folder_num)
    print(f"folder:{args.folder_num}, length: {max(timesteps)}")
    train_dataset = StateActionReturnDataset(obss, args.context_length * 3, actions, done_idxs, rtgs, timesteps, args.mode, 4)

    # SwinT
    swin_args, swin_config = swin_parse_option()

    swin_model = SwinTransformer(**swin_config)
    # swin_model = SwinMLP(**swin_config)
    swin_model = torch.nn.DataParallel(swin_model)
    swin_model.to(device)

    # DT

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))  # 기존 파라미터 값 : max_timesteps = 1993, n_layer=6, n_head=8, max(timesteps)

    model = GPT(mconf)
    # model = torch.nn.DataParallel(model)
    model.to(device)
    epochs = args.epochs

    tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512 * 20,
                          final_tokens=2 * len(train_dataset) * args.context_length * 3,
                          num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game,
                          max_timestep=max(timesteps), test_mode=args.mode,
                          ckpt_path="weights-MsPacman.pt", opt_type='adamw')  # 1993, max_timestep =max(timesteps), 2653

    n_parameters = sum([p.numel() for p in model.parameters()]+[q.numel() for q in swin_model.parameters()])
    print(n_parameters)

    trainer = Trainer(model, swin_model, train_dataset, None, tconf)
    trainer.train()
