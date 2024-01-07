from torchvision.transforms import InterpolationMode
import copy
from mingpt.utils import set_seed, sample
import numpy as np
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
import random
import torch
import glob
import pickle
import blosc
import argparse
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
class CustomDataset(Dataset):
    def __init__(self, data_root, transform =None, buffer=None):
        self.transform= transform
        self.data_root= data_root
        self.buffer_idx = buffer
        self.wall = None
        # self.num_classes= 4

    def __len__(self): # 전체 이미지 폴더 개수 리턴 -> 각 폴더별로 수정
        return len(self.buffer_idx)

    def __getitem__(self, i):
        # images= np.empty((1, 3, 252, 252), int)
        images = []
        rewards = []
        terminal = []
        actions = []
        num = self.buffer_idx[i] if i < len(self.buffer_idx) else None
        if num:
            file_list = os.listdir(self.data_root + "/screens/" + str(num))
            file_list = sorted(file_list, key= lambda x:int(x.split('.')[0]))
            for filename in tqdm(file_list):
                # if int(filename.split('.')[0]) < 300:  # or int(filename.split('.')[0])>205:
                #     continue
                file = f'{self.data_root}/screens/{str(num)}/{filename}'
                data = np.float32(np.array(Image.open(file).convert('RGB')))  # (210, 160, 3)
                if self.wall is None:
                    temp = copy.deepcopy(data)
                    self.wall = make_wall(temp)
                    self.wall = self.transform(self.wall)
                    del temp
                data = self.transform(data)  # (3, 252, 252)
                # --- 이미지 확인---
                # plt.title("DATA")
                # plt.imshow(data.permute(1,2,0))
                # plt.show()
                # plt.title("WALL")
                # plt.imshow(self.wall.permute(1, 2, 0))
                # plt.show()
                cond = torch.ne(data, self.wall)
                final_state = torch.where(cond, data.float(), torch.zeros(3, 252, 252))
                images.append(final_state)

                # --- 이미지 저장---
                # f1 = data_array.detach().cpu().numpy()
                # plt.axis('off')
                # plt.savefig("final_state.png",bbox_inches='tight', pad_inches = 0)

                # ---기타---
                # wall_img = self.transform(np.float64(wall_img))
                # final_state = torch.where((img-wall_img>=0)&(img-wall_img<1) , (img - wall_img).float(), img.float())

            txt_file = open(f"{self.data_root}/trajectories/{str(num)}.txt")
            strings = txt_file.readlines()
            for string in strings[2:]:
                vals = string.split(',') # frames.append(int(vals[0]))
                rewards.append(int(vals[1])) # 보상
                val = 0 if vals[3] is 'False' else 1  # terminal
                terminal.append(val)

                # TODO 멀티 라벨용 주석 + 추가한것 (230621)
                # action = int(vals[4].split('\\')[0])  # 액션
                # multi_action = torch.zeros(self.num_classes)

                action = int(vals[4].split('\\')[0])
                actions.append(action)
        else:
            return None

        # frames = torch.tensor(frames, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.int64)
        actions = torch.tensor(actions, dtype=torch.int64)
        return images, rewards, terminal, actions

    def reset(self):
        self.wall = None
        self.transform= None
        self.data_root= ''
        self.buffer_idx = 0

def crop172(image):
    image = image
    return image[:, 2:172, :]

def norm255(image):
    return (image-128)/128
    # meanR, stdR = torch.mean(image[0, :, :]).item(), torch.std(image[0, :, :]).item()
    # meanG, stdG = torch.mean(image[1, :, :]).item(), torch.std(image[1, :, :]).item()
    # meanB, stdB = torch.mean(image[2, :, :]).item(), torch.std(image[2, :, :]).item()
    # return transforms.Normalize((meanR, meanG, meanB), (stdR, stdG, stdB))(image)

def make_wall(_image):  # numpy, (210, 160, 3)
    red_img = _image[:,:,0]
    v = np.array([[224.]]).astype(np.float64)
    idx24 = []
    idx74 = []
    # (2, 4) or (7, 4) --> pellet 크기
    # 벽을 지우기!!!

    # (2, 4) 펠렛만 지우기
    for i in range(1, 170):
        for j in range(1, 156): # (1<=j<=154)
            if np.equal(red_img[i:i+2, j:j+4], np.tile(v, (2, 4))).all() and (red_img[i,j-1] != 224. and red_img[i,j+5] != 224. and red_img[i-1,j] != 224 and red_img[i+3,j] != 224):
                idx24.append((i,j))

    # (7, 4) 펠렛만 지우기
    for i in range(1, 170):
        for j in range(1, 156): # (1<=j<=154)
            if np.equal(red_img[i:i+7, j:j+4], np.tile(v, (7, 4))).all() and (red_img[i,j-1] != 224. and red_img[i,j+5] != 224. and red_img[i-1,j] != 224 and red_img[i+8,j] != 224):
                idx74.append((i,j))

    for k in idx24:
        _image[k[0]:k[0]+2, k[1]:k[1]+4, :] = [0, 24, 124]  # [0, 0, 0]

    for k in idx74:
        _image[k[0]:k[0]+7, k[1]:k[1]+4, :] = [0, 24, 124]  #[0, 0, 0]

    _image = np.where((_image!= [224, 136, 136])&(_image!=[0,24,124]), [0,24,124], _image)  # [0,0,0]
    return _image

def create_dataset(data_root, buffer_set, num_steps, trajectories_per_buffer):
    obss = []
    actions = []
    returns = []
    done_idxs = []
    trajectories_to_load = trajectories_per_buffer  # 각 버퍼에서 샘플링 할 경로의 수 = 10

    transform = transforms.Compose([
        # transforms.Lambda(norm255),
        transforms.ToTensor(),
        transforms.Lambda(crop172),
        transforms.Resize((252, 252), InterpolationMode.NEAREST),  # FIXME 원래는 bicubic으로 늘린건데 resize로 늘려서 수정필요
        transforms.Normalize((128, 128, 128), (128, 128, 128))  # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_trajectories = 0  # 원래는 len(obss) 였음
    buffer_set = buffer_set # [f.split('.')[0] for f in os.listdir(f'{data_root}/trajectories')]  # ['2'] # [1001, 1002, 1003] # [57, 58, 5, 7, 8, 2, 3, 56, 9, 1]
    train_dataset = CustomDataset(data_root=data_root,transform=transform, buffer=buffer_set)
    train_loader = DataLoader(train_dataset, batch_size=1)
    it = iter(train_loader)
    total_idx = 0
    i = 0
    while len(actions) < num_steps:
        print('loading ... which has %d already loaded' % (len(actions)))
        states, rewards, terminal, action = next(it, None)
        obss += states
        actions += action[0].tolist()
        returns += rewards[0].tolist()
        done_idxs += [len(obss)] # 누적으로 보는 terminal 인덱스
        trajectories_to_load -= 1
        i= i+1
        total_idx += len(actions)
        print(total_idx)
        # if total_idx > 1: # 9000:
        #     break

    print("total_idx:",total_idx)
    num_trajectories += (trajectories_per_buffer - trajectories_to_load)
    print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (total_idx, len(obss), num_trajectories))  # len(obss)

    actions = np.array(actions)
    returns = np.array(returns)
    done_idxs = np.array(done_idxs)

    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))
    del train_dataset
    return obss, actions, returns, done_idxs, timesteps  # rtg
