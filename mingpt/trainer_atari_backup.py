import copy
import time

import math
import logging

import torchvision.transforms
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from .utils_backup import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
from optimizer_backup import configure_optimizers
import torch.cuda.amp as amp
import log_generator


# weight_decay: 오버피팅을 줄이는 방법 중 하나, 손실값에 패널티로 w^2을 더하고 가중치 업데이트시 가중치의 크기가 커지는 걸 방지 (regularization)
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)  # Q. 무슨 용도?
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    ckpt_path = None
    num_workers = 0,  # for DataLoader
    load_model = False  # 내가 추가
    early_stopping = False

    def __init__(self, **kwargs):  # 속성 값 지정
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, dt_model, swin_model, train_dataset, test_dataset, config):
        self.model = dt_model
        self.swin_model = swin_model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.train_loss = None
        self.val_loss = []
        self.elapsed_time = None

        self.device = 'cpu'
        if torch.cuda.is_available():  # DataParallel wrappers keep raw model object in .module attribute
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.swin_model = torch.nn.DataParallel(self.swin_model).to(self.device)

    def save_checkpoint(self, ep, opt):
        dt_model = self.model.module if hasattr(self.model, "module") else self.model
        swin_model = self.swin_model.module if hasattr(self.swin_model, "module") else self.swin_model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save({'swin_model': swin_model.state_dict(), 'optimizer': opt.state_dict()},
                   f'swinT_weight_{self.config.game}.pt')
        '''
        torch.save({'dt_model': dt_model.state_dict(), 'swin_model': swin_model.state_dict(), 'optimizer': opt.state_dict()},
                   f'weights-{self.config.game}.pt')  # 이름을 다르게 하여 매번 저장할 경우 memory leak 생김
        '''

    def train(self):
        model, swin_model, config = self.model, self.swin_model, self.config
        model = model.module if hasattr(self.model, "module") else model  # DT 만 해주고 SWIn 추가 안함. 나중에 고려해서 수정 필요
        swin_model = swin_model.module if hasattr(self.swin_model, "module") else swin_model
        optimizer = configure_optimizers(model, swin_model, config.weight_decay, config.learning_rate, config.betas)
        scaler = amp.GradScaler()  # 계산 비용을 줄이면서 더 큰 배치사이즈 사용 가능

        if self.config.load_model:
            checkpoint = torch.load("./보관0508_swinT_weight_MsPacman.pt")
            # model.load_state_dict(checkpoint['dt_model'])
            swin_model.load_state_dict(checkpoint['swin_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            del checkpoint
            torch.cuda.empty_cache()

        def run_epoch(split):
            start_time = time.time()
            is_train = split == 'train'  # train/test 결정하는 곳
            if not is_train:
                correct_pred = {action:0 for action in range(0,9)}
                total_pred = {action:0 for action in range(0,9)}
            model.train(is_train)
            swin_model.train(is_train)
            data = self.train_dataset  # if is_train else self.test_dataset

            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            print("dpd:",len(loader))
            pbar = tqdm(enumerate(loader), total=len(loader)) # if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:  # x: state, y: action, r: rtg, t:timestep

                x = x.to(self.device)  # (16, 30, 4, 84, 84) --> (4, 1 256, 256*30 =7680)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                context_len = y.shape[1]
                with torch.set_grad_enabled(is_train):
                    with torch.cuda.amp.autocast():
                        features = []
                        #for i in range(context_len):
                        feature = swin_model(x)  # (16, 30, 4, 84(224), 84(224)) : 나중에 넣을때는 (16*30, 4, 84, 84) --> output 형태는 (480=16*30, 128=dim)
                        features.append(feature.unsqueeze(1))
                        # logits, loss = model(feature, y, y, r, t, context_len)  # logits, loss = model(x, y, y, r, t)
                        logits = torch.cat(features, dim=1)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                        if not is_train:
                            ss, predictions = torch.max(torch.softmax(logits, dim=2), 2)
                            for label, prediction in zip(y.reshape(-1), predictions.reshape(-1)):
                                # print(f"label:{label}, prediction:{prediction}")
                                if label == prediction:
                                    correct_pred[prediction.item()]+= 1
                                total_pred[prediction.item()]+= 1

                    loss = loss.mean()
                    losses.append(loss.item())

                del x, r, t

                if is_train:
                    model.zero_grad(set_to_none=True)  # 모든 모델의 parameter's gradient set to zero
                    swin_model.zero_grad(set_to_none=True)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(swin_model.parameters()),
                                                   config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # 기존 DT optimizer
                    # loss.backward()  # gradient 계산
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    # optimizer.step()  # w, b 업데이트

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            self.train_loss = float(sum(losses) / len(losses))
            self.elapsed_time = time.time() - start_time  # duration

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                print("테스트 결과:", test_loss)

                correct_count = 0
                total_count = 0
                for action_name, action_count in correct_pred.items():
                    accuracy = 100 * float(action_count) / total_pred[action_name] if total_pred[action_name] != 0 else 0
                    print(f'Accuracy for class: {str(action_name):5s} is {accuracy:.1f} %')
                    correct_count += action_count
                    total_count += total_pred[action_name]
                total_accuracy = 100 * float(correct_count) / total_count
                print(f'Accuracy for total: {total_accuracy:.1f} %')
                return test_loss

        best_loss = float('inf')
        # best_return = -float('inf')

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            if self.config.test_mode or self.config.early_stopping:
                test_loss = run_epoch('test')
                log_generator.make_log(episode=epoch, loss=self.train_loss, duration=self.elapsed_time)
            else:
                run_epoch('train')
                # 표 만드는 곳
                log_generator.make_log(episode=epoch, loss=self.train_loss, duration=self.elapsed_time)

            # 원래 주석
            # early stopping
            # if self.config.early_stopping:
            #     test_loss = run_epoch('test')
            #     # good_model = self.config.early_stopping and test_loss < best_loss
            #     if self.config.ckpt_path is not None:
            #         best_loss = test_loss
            #         self.save_checkpoint(epoch, optimizer)

            # 원래 주석 아님
            if self.config.ckpt_path is not None or not self.config.test_mode != True:
                self.save_checkpoint(epoch, optimizer)

            # -- pass in target returns (목표로 하는 값, 성능 제어에 도움=난이도 조정) => 훈련 도중 eval하는 용도  ((일단 보류))
            # if self.config.model_type == 'naive':
            #     eval_return = self.get_returns(0)
            # elif self.config.model_type == 'reward_conditioned':
            #     if self.config.game == 'MsPacman':
            #         eval_return = self.get_returns(1200)  # 총 150개의 pellet이 존재
            #     else:
            #         raise NotImplementedError()
            # else:
            #     raise NotImplementedError()

    def get_returns(self, ret):  # 실제 모델을 돌려서 나온 reward(rtg) 계산
        # self.model.train(False)
        self.swin_model.train(False)
        args = Args(self.config.game.lower(), self.config.seed)
        env = Env(args)

        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        test_iteration = 3
        for i in range(test_iteration):  # 10번을 돌려 -> 3으로 수정 (빨리 보려고)
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) #state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 4->1, 224, 224)
            rtgs = [ret]  # 목표로 하는 값 저장
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.swin_model.module, state, 1, temperature=1.0, sample=True,  # self.model.module,
                                    actions=None,
                                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0, -1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1
                if self.config.test_mode:
                    env.render()
                if done:
                    T_rewards.append(reward_sum)  # 현재 timestep까지의 reward 총 합
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 4->1, 224, 224)

                all_states = torch.cat([all_states, state], dim=0)  # (2, 1, 4, 224, 224)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                # 주어진 state 등등을 시퀀스 길이에 맞게 자르고 훈련 없이 그저 actions들만 추출(샘플링, 단 무작위 아니고 모델에 따라 실행한 값임)
                sampled_action = sample(self.model.module, self.swin_model.module, all_states.unsqueeze(0), 1,
                                        temperature=1.0, sample=True,
                                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                                        rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                                        timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1),dtype=torch.int64).to(self.device)))  # all_states.unsqueeze(0)
        env.close()

        eval_return = sum(T_rewards) / float(test_iteration)  # 원래는 10. 이였다  # epoch 동안의 reward의 평균치
        max_eval_return = max(T_rewards)
        print("target return: %d, eval return: %d, max eval return: %d" % (
        ret, eval_return, max_eval_return))  # 목표로 하는 reward(target return)와 실제 모델에서 돌려서 얻은 reward(eval return)의 비교
        self.model.train(True)
        self.swin_model.train(True)
        return eval_return


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path("ms_pacman"))
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = False  # Consistent with model training mode

    def _get_state(self):  # 이미지 스케일링
        state = cv2.resize(self.ale.getScreenGrayscale(), (252, 252),
                           interpolation=cv2.INTER_CUBIC)  # cv2.INTER_LINEAR, INTER_NEAREST
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(252, 252, device=self.device))

    def reset(self):
        if self.life_termination:  # 3 목숨 잃음
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # 한 판이 안끝난 상태의 reset
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)  # (1, 224, 224)

    def step(self, action):  # 스킵 프레임이 4개라 4번 액션 반복
        # max pool over last 2 frames
        frame_buffer = torch.zeros(2, 252, 252, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 1  # TODO
