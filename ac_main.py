import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from swin_model.swin_transformer import SwinTransformer
from swin_model.swinconfig import SwinConfig
from swin_model.pre_trainer import Mlp, PreTrainConfig, ac_train, ac_test
from mingpt.trainer_atari import Trainer, TrainerConfig
import torch
import matplotlib.pyplot as plt
import blosc
import argparse
from create_dataset import create_dataset
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as FF
import torchvision.transforms as T

class StateActionReturnDataset(Dataset):

    def __init__(self, data, actions, done_idxs, rtgs, timesteps, skip):
        self.vocab_size = max(actions) + 1
        self.data = data  # [list : 15570] -> ndarray (4, 84, 84) --> tensor (1, 3, 252, 252)
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.data_shape = np.shape(data[0])  # (3, 252, 252)
        self.skip = skip
        convert_tensor = T.ToTensor()
        self.norm = transforms.Normalize((128, 128, 128), (128, 128, 128))
        print("len(states):", len(data))
        print("len(actions):", len(actions))

    def __len__(self):
        return len(timesteps)-1

    def __getitem__(self, idx):
        # TODO obs 리턴 형태 바꿈
        states = self.data[idx][0]
        states = states[:,:,9:-9] # 가로 자름
        actions = torch.tensor(self.actions[idx: idx+1], dtype=torch.int64).unsqueeze(1)  # dtype=torch.long

        return states, actions #, rtgs, timesteps

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_pretrain = False
    device = torch.device('cuda') # cuda
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_steps', type=int, default=220000)  # 220000, 500000
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                        help='Number of trajectories to sample from each of the buffers.')  # 10
    args = parser.parse_args()

    set_seed(args.seed)
    obss, actions, returns, done_idxs, timesteps = create_dataset(args.num_buffers, args.num_steps,
                                                                  args.trajectories_per_buffer)

    train_dataset = StateActionReturnDataset(obss, actions, done_idxs, returns, timesteps, 4)  # returns -> rtgs

    pretrain_config = PreTrainConfig().get_config()

    # Action Classifier
    action_classifier = Mlp(out_channel=10)
    if is_pretrain:
        ac_train(train_dataset, pretrain_config, action_classifier)
    else:
        ac_test(train_dataset, pretrain_config, action_classifier)