import copy
import time
import torchvision.transforms as transforms
import math
import logging
# TODO 멀티 라벨용 추가한것 (230621)
import datetime as dt
import torch.nn as nn
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

from .utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
from optimizer import configure_optimizers
import torch.cuda.amp as amp
import log_generator
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

torch.backends.cudnn.enabled = False

# weight_decay: 오버피팅을 줄이는 방법 중 하나, 손실값에 패널티로 w^2을 더하고 가중치 업데이트시 가중치의 크기가 커지는 걸 방지 (regularization)
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    ckpt_path = None
    num_workers = 0,  # for DataLoader
    load_model = True  # 내가 추가
    early_stopping = False

    def __init__(self, **kwargs):  # 속성 값 지정
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, ac_model, swin_model, train_dataset, test_dataset, config):  # ac : action classifier
        self.swin_model = swin_model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.train_loss = None
        self.val_loss = []
        self.elapsed_time = None
        self.device = 'cpu'
        self.ac_model = ac_model
        # TODO 멀티 라벨용 추가한것 (230621)
        # self.criterion = nn.BCELoss()

        if torch.cuda.is_available():  # DataParallel wrappers keep raw model object in .module attribute
            self.device = torch.cuda.current_device()
            # self.swin_model = torch.nn.DataParallel(self.swin_model).to(self.device)

    def save_checkpoint(self, ep, opt):
        swin_model = self.swin_model.module if hasattr(self.swin_model, "module") else self.swin_model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save({'swin_model': swin_model.state_dict(), 'optimizer': opt.state_dict()},
                   f'swinT_weight_{self.config.game}_ep{ep}.pt')

    def train(self):
        swin_model, config = self.swin_model, self.config
        swin_model = swin_model.module if hasattr(self.swin_model, "module") else swin_model
        optimizer = configure_optimizers(swin_model, config.weight_decay, config.learning_rate, config.betas)
        scaler = torch.cuda.amp.GradScaler()  # 계산 비용을 줄이면서 더 큰 배치사이즈 사용 가능

        if self.config.load_model:
            checkpoint = torch.load("./swinT_weight_MsPacman_ep40.pt")
            swin_model.load_state_dict(checkpoint['swin_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            del checkpoint
            torch.cuda.empty_cache()

        def run_epoch(split):
            start_time = time.time()
            is_train = split == 'train'  # train/test 결정하는 곳
            correct_pred = {action:0 for action in range(0,4)}
            total_pred = {action:0 for action in range(0,4)}
            swin_model.train(is_train)
            data = self.train_dataset # if is_train else self.test_dataset # if is_train else self.test_dataset

            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) # if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:  # x: state, y: action, r: rtg, t:timestep

                x = x.to(self.device)  # (16, 3, 252, 234) 0~3
                y = y.to(self.device)  # (16, 1, 1)  2~5  -> (16, 2, 1)? multi label용으로?
                # r = r.to(self.device)
                # t = t.to(self.device)

                with torch.set_grad_enabled(is_train):
                    with torch.cuda.amp.autocast():
                        feature = swin_model(x)  # feature: (16, 4)
                        logits = feature.unsqueeze(1)  # (16, 1, 4)

                        # TODO 멀티 라벨용 주석한것 (230621)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                        # loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1, logits.size(-1)))

                        ss, predictions = torch.max(torch.softmax(logits, dim=2), 2)
                        for label, prediction in zip(y.reshape(-1), predictions.reshape(-1)):
                            # print(f"label:{label}, prediction:{prediction}")
                            if label == prediction:
                                correct_pred[prediction.item()] += 1
                            total_pred[prediction.item()] += 1

                    losses.append(loss.item())  # loss.item()

                if is_train:
                    # loss_tensor = torch.tensor(total_loss, requires_grad=True).to(self.device)
                    # 모든 모델의 parameter's gradient set to zero
                    swin_model.zero_grad(set_to_none=True)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(swin_model.parameters()), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

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
                # losses = torch.stack(losses)
                test_loss = float(sum(losses) / len(losses)) #float(np.mean(losses))
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
                print(f'Accuracy for total(test): {total_accuracy:.1f} %')
                return test_loss, total_accuracy
            else:
                correct_count = 0
                total_count = 0
                for action_name, action_count in correct_pred.items():
                    correct_count += action_count
                    total_count += total_pred[action_name]
                total_accuracy = 100 * float(correct_count) / total_count
                print(f'Accuracy for total(train): {total_accuracy:.1f} %')
                return total_accuracy  # train_accuracy

        best_loss = float('inf')
        # best_return = -float('inf')
        train_losses = []
        test_losses = []
        test_accs = []
        train_accs = []
        scores = []
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            if self.config.test_mode:
                test_loss, test_acc = run_epoch('test')
                # log_generator.make_log(episode=epoch, loss=test_loss)  # self.train_loss
            else:
                train_acc= run_epoch('train')
                train_accs.append(train_acc)

            if not self.config.test_mode:
                test_loss, test_acc = run_epoch('test')
                test_losses.append(test_loss)
                train_losses.append(self.train_loss)
                test_accs.append(test_acc)

            # 원래 주석 아님
            # if self.config.ckpt_path is not None or not self.config.test_mode != True:
            #     if epoch%10==0 and epoch!=0:
            #         self.save_checkpoint(epoch, optimizer)

            eval_return = self.get_returns()  # 총 150개의 pellet이 존재
            # print("evaluation score:", eval_return)
            scores.append(eval_return)
            if not self.config.test_mode:
                log_generator.make_log(episode=epoch, loss=self.train_loss, total_reward=eval_return)  # 표 만드는 곳

        # plt.figure(figsize=(7, 5))
        # plt.title("Training and Validation Loss")
        # plt.plot(test_losses, label="val")
        # plt.plot(train_losses, label="train")
        # plt.xlabel("iterations")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig("augmentation_Loss.png")
        # plt.show()
        #
        # # accuracy 용도
        # plt.title("Training and Validation Accuracy")
        # plt.plot(test_accs, label="val")
        # plt.plot(train_accs, label="train")
        # plt.xlabel("iterations")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.savefig("augmentation_Accuracy.png")
        # plt.show()

        now = dt.datetime.now()
        now = now.strftime('%Y-%m-%d')
        with open(f'rewLog_{now}.txt', 'w') as f:
            for s in scores:
                sen = str(s) + '\n'
                f.write(sen)
        f.close()


    def get_returns(self):  # 실제 모델을 돌려서 나온 reward(rtg) 계산
        # self.model.train(False)
        self.swin_model.train(False)
        args = Args(self.config.game.lower(), self.config.seed)
        env = Env(args)

        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        test_iteration = 1
        for i in range(test_iteration):  # 10번을 돌려 -> 3으로 수정 (빨리 보려고)
            state = env.reset()
            state = state.type(torch.float32).to(self.device) #state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 4->1, 224, 224)
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.swin_model.module, state, temperature=1.0, sample=True, top_k=1)

            j = 0
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0, -1]
                action = action + 1
                actions += [sampled_action]
                # TODO  액션 수 줄이면 제일 먼저 없애야하는 조건문!!
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1
                if self.config.test_mode:
                    pass
                    # env.render()
                if done:
                    T_rewards.append(reward_sum)  # 현재 timestep까지의 reward 총 합
                    break

                state = state.to(self.device)  # (1, 4->1, 224, 224) .unsqueeze(0)
                sampled_action = sample(self.swin_model.module, state,temperature=1.0, sample=True, top_k=1)  # all_states.unsqueeze(0)
        env.close()

        eval_return = sum(T_rewards) / float(test_iteration)
        max_eval_return = max(T_rewards)
        print("eval return: %d, max eval return: %d" % (eval_return, max_eval_return))
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
        self.transform = transforms.Compose([ transforms.ToTensor(),
                                              transforms.Lambda(self.crop172),
                                              transforms.Resize((252, 252), InterpolationMode.NEAREST),# FIXME 원래는 bicubic으로 늘린건데 resize로 늘려서 수정필요
                                              transforms.Lambda(self.norm255)])
        self.wall_img = None

    def _get_state(self):  # 이미지 스케일링
        state = (self.ale.getScreenRGB()).astype(np.float32)  # (210, 160, 3) --> 원래부터 rkqtdl 228로 나오는거였음;;
        # state = torch.tensor(state, dtype=torch.float16, device=self.device) #.div_(255)
        return state

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(3, 252, 234, device=self.device))

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
        observation = self._get_state()  # (252, 252, 3)
        if self.wall_img is None:
            temp = copy.deepcopy(observation) # np.array(copy.deepcopy(observation).cpu(), dtype=np.float16)
            self.wall_img = self.make_wall(temp)
            self.wall_img = self.transform(self.wall_img)
            del temp
        observation = self.transform(observation)
        observation = self.subtract_wall(observation)

        # plt.title("reset")
        # plt.imshow(observation.permute(1, 2, 0))
        # plt.show()

        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)  # (1, 224, 224)

    def step(self, action):  # 스킵 프레임이 4개라 4번 액션 반복
        # max pool over last 2 frames
        frame_buffer = torch.zeros(2, 3, 252, 252, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self.transform(self._get_state())
            elif t == 3:
                frame_buffer[1] = self.transform(self._get_state())
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        observation = self.subtract_wall(observation.cpu())
        self.state_buffer.append(observation)

        # plt.title("step image")
        # plt.imshow(observation.permute(1, 2, 0))
        # plt.show()

        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done
        # return self.state_buffer[-1], reward,done

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

    def crop172(self, image):
        image = image
        return image[:, 2:172, :]

    def norm255(self, image):
        return (image - 128) / 128

    def subtract_wall(self, image):
        cond = torch.ne(image, self.wall_img)
        final_state = torch.where(cond, image.float(), torch.zeros(3, 252, 252))
        final_state = final_state[:, :, 9:-9]
        return final_state

    def make_wall(self, _image):  # numpy, (210, 160, 3) --> (252, 252, 3)
        red_img = _image[:, :, 0]
        v = np.array([[228.]]).astype(np.float64)
        idx24 = []
        idx74 = []
        # (2, 4) or (7, 4) --> pellet 크기
        # TODO 그림 픽셀값이 다름...확인필요.......

        # (2, 4) 펠렛만 지우기
        for i in range(1, 170):
            for j in range(1, 156):  # (1<=j<=154)
                if np.equal(red_img[i:i + 2, j:j + 4], np.tile(v, (2, 4))).all() and (
                        red_img[i, j - 1] != 228. and red_img[i, j + 5] != 228. and red_img[i - 1, j] != 228 and
                        red_img[i + 3, j] != 228):
                    idx24.append((i, j))

        # (7, 4) 펠렛만 지우기
        for i in range(1, 170):
            for j in range(1, 156):  # (1<=j<=154)
                if np.equal(red_img[i:i + 7, j:j + 4], np.tile(v, (7, 4))).all() and (
                        red_img[i, j - 1] != 228. and red_img[i, j + 5] != 228. and red_img[i - 1, j] != 228 and
                        red_img[i + 8, j] != 228):
                    idx74.append((i, j))

        for k in idx24:
            _image[k[0]:k[0] + 2, k[1]:k[1] + 4, :] = [0, 24, 124]  # [0, 0, 0]

        for k in idx74:
            _image[k[0]:k[0] + 7, k[1]:k[1] + 4, :] = [0, 24, 124]  # [0, 0, 0]

        _image = np.where((_image != [228, 136, 136]) & (_image != [0, 24, 124]), [0, 24, 124], _image)  # [0,0,0]
        return _image


class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 1  # TODO
