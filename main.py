import gc
import logging
# make deterministic
# from mingpt.utils import set_seed  # **
from mingpt.utils_augmentation import set_seed
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset, random_split
from swin_model.swinconfig import SwinConfig
from swin_model.pre_trainer import Mlp, PreTrainConfig, ac_train, ac_test

# from swin_model.swin_transformer import SwinTransformer  # **
# from mingpt.trainer_atari import Trainer, TrainerConfig  # **
import os
from swin_model.swin_transformer_augmentation import SwinTransformer
from mingpt.trainer_atari_augmentation import Trainer, TrainerConfig

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


from lr_scheduler import build_scheduler
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
        # self.wall_img = (convert_tensor(Image.open("./test.png").convert("RGB")))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO obs 리턴 형태 바꿈
        states = self.data[idx][0]
        # states = torch.where(states[0] - self.wall_img*3 >= 0.0, (states[0] - self.wall_img*3).float(), torch.zeros(3, 252, 252))  #torch.zeros(3, 252, 252))
        states = states[:,:,9:-9] # 가로 자름

        # --확인용 코드--
        # plt.title('Rimg')
        # transform = T.ToPILImage()
        # img = transform(states[0])  # (33, 77, 67)
        # plt.imshow(img)
        # plt.show()

        actions = torch.tensor(self.actions[idx: idx+1], dtype=torch.int64).unsqueeze(1)  # dtype=torch.long
        rtgs = torch.tensor(self.rtgs[idx: idx+1], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)  # dtype=torch.int64

        return states, actions, rtgs, timesteps # TODO 여기 state -> float32타입인지 확인 필요!!

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

def get_buf(buf_list, data_root, start_idx):
    count = 0
    for idx, b in enumerate(buf_list):
        if idx < start_idx:
            continue
        count += len(os.listdir(f'{data_root}/screens/{b}'))
        if count > 50000:
            return idx + 1

    return len(buf_list)

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda') # cuda
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--context_length', type=int, default=1)  # 30 , 모델 로드할때 가중치의 seq_length랑 동일하지 않으면 안돌아감
    parser.add_argument('--epochs', type=int, default=1)  # TEST
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--num_steps', type=int, default=20000)  # 220000, 500000
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--game', type=str, default='MsPacman')
    parser.add_argument('--batch_size', type=int, default=4)  # 128s -> 16
    parser.add_argument('--folder_num', type=int, default=1)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                        help='Number of trajectories to sample from each of the buffers.')  # 10
    parser.add_argument('--mode', type=bool, default=False, help='train==False or test==True')  # mode: True -> 순차 훈련, False -> 병렬 훈련
    parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')  # 사용 x
    parser.add_argument('--is_pretrain', type= str, default=True, help='Action Classifier pretrain')
    args = parser.parse_args()

    set_seed(args.seed)

    # 임시방편의 test dataset
    # obss, actions, returns, done_idxs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.trajectories_per_buffer)
    # test_dataset = StateActionReturnDataset(obss,actions, done_idxs, returns, timesteps, 4)  # returns -> rtgs

    pretrain_config = PreTrainConfig().get_config()
    data_root = 'D:/0_pycharm_project/Atari_Challenge_Dataset/full'  # 'D:/0_pycharm_project/SwinDT/NEWmspacman/mspacman'

    # SwinT
    swin_args, swin_config = swin_parse_option()
    swin_model = SwinTransformer(**swin_config)
    # swin_model = SwinMLP(**swin_config)
    swin_model = torch.nn.DataParallel(swin_model)
    swin_model.to(device)
    n_parameters = sum([p.numel() for p in swin_model.parameters()])
    print("총 파라미터 수: ",n_parameters)
    epochs = args.epochs

    buffer_set = [f.split('.')[0] for f in os.listdir(f'{data_root}/trajectories')]
    start_idx = 0

    while True:
        end_idx = get_buf(buffer_set, data_root, start_idx)
        buff = buffer_set[start_idx: end_idx]
        print("BUFF:",buff)
        obss, actions, returns, done_idxs, timesteps = create_dataset(data_root, buff, args.num_steps, args.trajectories_per_buffer)
        dataset = StateActionReturnDataset(obss, actions, done_idxs, returns, timesteps, 4)  # returns -> rtgs

        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size - validation_size

        train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
        tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                              lr_decay=True, warmup_tokens=512 * 20,
                              final_tokens=2 * len(dataset) * args.context_length * 3,
                              num_workers=0, seed=args.seed, model_type=args.model_type, game=args.game,
                              max_timestep=max(timesteps), test_mode=args.mode,
                              ckpt_path="weights-MsPacman.pt", opt_type='adamw')  # num_workers =4
        if start_idx == 0:
            tconf.load_model = False
        else:
            tconf.load_model = True

        trainer = Trainer(None, swin_model, train_dataset, test_dataset, tconf)  # swin_model, train_dataset, None, tconf, action_classifier
        trainer.train()

        del trainer
        del tconf
        del dataset
        gc.collect()
        if end_idx != len(buffer_set):
            start_idx = end_idx
        else:
            break