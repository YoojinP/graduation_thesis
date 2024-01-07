import csv
import logging
from mingpt.utils_backup import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari_backup import Trainer, TrainerConfig
from mingpt.utils_backup import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
import cv2
from fixed_replay_buffer import FixedReplayBuffer
import matplotlib.pyplot as plt

def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer, folder_num):
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)  # transitions per buffer = 50
    num_trajectories = 0  # 원래는 len(obss) 였음
    while len(obss) < num_steps:  # 일정 스텝 넘어가지 않으면 계속 한 파일의 모든 trajectory load한다. 단 replay capacity를 넘어갈 정도로 한 에피소드의 trajectory가 길면 자른다
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]  # 50개의 버퍼 중 랜덤 선택
        i = transitions_per_buffer[buffer_num]  # 버퍼 내의 transition의 첫번째 인덱스, 항상 i = 0 (confirmed)
        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + f'{game}/{folder_num}/replay_logs',  # data_dir=data_dir_prefix + f'{game}/1/replay_logs'
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer  # 각 버퍼에서 샘플링 할 경로의 수 = 10
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                states = states[:, :68, :]
                obss += [states]  # (4, 84, 84)
                actions += [ac[0]]
                stepwise_returns += [ret[0]]  # 보상을 스텝 별로 기록 저장
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]  # stepwise return의 총합
                i += 1  # 전체 transitions 개수
                if i >= 100000:  # 해당 버퍼의 경로길이가 기준치를 넘으면 더이상 담지 않고 이전 내용까지만 담음
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)  # 실제로 구한 trajectory의 개수
            transitions_per_buffer[buffer_num] = i
        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))  # len(obss)

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    print("done_idxs:",done_idxs)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]  # 처음 시작 상태부터 마지막 끝난 상태까지의 reward를 모두 저장한 것
        for j in range(i-1, start_index-1, -1): # start from i-1 , 끝난 상태부터 처음 시작으로 거꾸로 loop
            rtg_j = curr_traj_returns[j-start_index:i-start_index]  # 현재 상태에서부터 끝난 상태까지의 reward로 수정, 처음 시작부터 현재까지의 reward는 제외
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    # obss 이미지 전처리


    return obss, actions, returns, done_idxs, rtg, timesteps
