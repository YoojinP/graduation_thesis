import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PreTrainConfig:
    batch_size = 1
    num_workers = 0
    learning_rate = 0.0006
    epochs = 5
    path = './action_classifier.pt'
    save_checkpoint = True
    load_checkpoint = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def get_config(cls):
        return {name: val for name, val in vars(cls).items() if not name.startswith('__')}

class Mlp(nn.Module):
    def __init__(self, input_size=(252, 234), in_channel=3, out_channel=None, act_layer=nn.GELU):
        super().__init__()
        self.input_size= input_size
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size-(kernel_size-1)-1) //stride +1
        convw = conv2d_size_out(conv2d_size_out(self.input_size[1]))
        convh = conv2d_size_out(conv2d_size_out(self.input_size[0]))
        linear_input_size = convh * convw * 32
        self.flatten = nn.Flatten() # flatten 말고 그냥 view로 shape 고쳐서 넣는 방법 사용하자 --> 수정 필요!!
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.act = act_layer()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, out_channel)

    # (16, 3, 252, 234)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # (4, 16, 125, 116)
        # x = self.pool1(x)
        x = (self.conv2(x))  # (16, 32, 123, 114)
        x = self.relu(self.bn2(x))
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def ac_train(train_dataset, pretrain_config, model):
    device = torch.device('cpu') # cuda
    loss_ = []
    loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=pretrain_config['batch_size'], num_workers=pretrain_config['num_workers'])
    total_batch = len(loader)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_config['learning_rate'])
    if pretrain_config['load_checkpoint']:
        model.load_state_dict(torch.load(pretrain_config['path'])['model'])
        optimizer.load_state_dict(torch.load(pretrain_config['path'])['optimizer'])

    pbar = tqdm(enumerate(loader), total=len(loader))  # if is_train else enumerate(loader)
    for epoch in range(pretrain_config['epochs']):
        avg_cost = 0
        running_loss = 0.0
        print(f"epoch:{epoch}")
        for it, (x, y) in pbar:  # x: state, y: action, r: rtg, t:timestep

            x = x.to(device)  # (B, 3, 252, 234)
            y = y.to(device)  # (B, 1, 1)

            output = model(x)
            logits = output.unsqueeze(1)

            model.zero_grad(set_to_none=True)
            optimizer.zero_grad()

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch
            running_loss += loss.item()

        loss_.append(running_loss/total_batch)

        if pretrain_config['save_checkpoint']:
            model = model.module if hasattr(model, "module") else model
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       f'action_classifier.pt')

        print('[Epoch: {:>4}] cost = {:>.9f}'.format(epoch + 1, avg_cost))
    show_plot(loss_)



def ac_test(test_dataset, pretrain_config, model):
    correct = 0
    total = 0
    device = torch.device('cuda')  # cuda

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_config['learning_rate'])
    model.load_state_dict(torch.load(pretrain_config['path'])['model'])
    optimizer.load_state_dict(torch.load(pretrain_config['path'])['optimizer'])

    test_loader = DataLoader(test_dataset, shuffle=True, pin_memory=True, batch_size=pretrain_config['batch_size'], num_workers=pretrain_config['num_workers'])
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():
        for it, (x, y) in pbar:
            images, labels = x.to(device), y.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 10개의 class중 가장 값이 높은 것을 예측 label로 추출.
            total += labels.size(0)  # test 개수
            correct += (predicted == labels).sum().item()

    print(f'accuracy of 10000 test images: {100 * correct / total}%')


def show_plot(loss):
    plt.plot(loss)
    plt.title(loss)
    plt.xlabel('epoch')
    plt.show()