# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import functional as F
import numpy as np
import sys
import matplotlib.pyplot as plt
torch.set_printoptions(profile="full", precision=4, sci_mode=False)
np.set_printoptions(threshold=sys.maxsize, precision=4)
try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):

    B, H, W, C = x.shape  # (1, 18, 18, 1)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  # (2, 6, 3, 6, 3, 32)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  # (batch * 윈도우 개수(h)* 윈도우 개수(w), 윈도우 사이즈, 윈도우 사이즈, dim) (72, 3, 3, 32)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))  # B: 2
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1) # x: (2, 6, 6, 3, 3, 32)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # relative position bias 파라미터 테이블 선언
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 창 내부의 각 토큰에 대한 상대 위치 인덱스를 얻는 코드
        # 1. 윈도우 사이즈를 기반으로 세로에서 변할 수 있는 범위를 모두 추출
        coords_h = torch.arange(self.window_size[0])  # 3 ([0, 1, 2])
        coords_w = torch.arange(self.window_size[1])  # 3 ([0, 1, 2])
        # 2. meshgrid: 1 dimensional vector인 N 개의 tensor를 받아서 N개의 N dimensional grids를 만든다
        # (2, Wh, Ww) (2,3,3) --> [[[0,0,0], [1,1,1], [2,2,2]], [[0,1,2] , [0,1,2] , [0,1,2]]]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # (2, Wh*Ww) (2,9) [[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]] 하나의 축을 기준으로, 윈도우 내에서 나올수 있는 모든 경우의 수
        coords_flatten = torch.flatten(coords, 1)
        # 3. (2, Wh*Ww, Wh*Ww) --> (2,9,9) broadcast, 각 패치마다 자기를 기준으로 다른 패치들이 하나의 축에서 어느정도의 거리에 위치하는지 표시
        # (후에 self-attention의 텐서와 더해짐)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # (Wh*Ww, Wh*Ww, 2) (9,9,2) 뒤의 2는 x축, y축 따로 이동값을 담은 matrix 두개를 가지므로 생김
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # 인덱스를 0부터 시작하게 만듬
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # (2M-1) 한변의 길이를 곱한다?? -> 원래의 최대 윈도우 크기를 반영하여 범위 값 조정
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # (Wh*Ww, Wh*Ww, X_axis_matrix + Y_axis_matrix)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 정규화
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pad_mask= None):

        B_, N, C = x.shape  # (72, 9, 32)
        # x를 벡터화한게 qkv
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale  # q = (72, 4, 9, 8)
        attn = (q @ k.transpose(-2, -1))  # attn = (72, 4, 9, 9)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # 수정
        nW = pad_mask.shape[0]
        # attn: (1, 36, 4, 9, 9), pad_mask.unsqueeze(1).unsqueeze(0): (1, 36, 4, 9, 9)
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + pad_mask.unsqueeze(1).unsqueeze(0)
        if mask is not None:
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N) # (1, 36, 4, 9, 9) -> (36, 4, 9, 9)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, layer_num=0, block_num =0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (14, 18), (7, 9)
        self.num_heads = num_heads  # 멀티 헤드의 개수
        self.window_size = window_size  
        self.shift_size = shift_size  
        self.mlp_ratio = mlp_ratio  # 중간에 늘렸다 줄일때 은닉계층 dim
        self.layer_num = layer_num  # 몇번째 basic layer 인지 표시
        self.block_num = block_num # 하나의 레이어에서 몇번째 블락인지 표시
        self.fused_window_process = fused_window_process
        if min(self.input_resolution) <= self.window_size:
            # 창 크기가 입력 해상도보다 크면 창을 분할하지 않는다
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # residual connection 에서 일어나는 dropout 같은 개념
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        flag= False
        if self.shift_size > 0:
            # SW-MSA를 위한 attention mask 생성 및 계산
            H, W = self.input_resolution
            # 패딩으로 달라진 입력 형태 반영하는 부분
            if self.layer_num == 0:
                H_ = H + 4
            else:
                H_ = H + 2
                flag = True
                
            img_mask = torch.zeros((1, H_, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))  # (-shift_size, -shift_size+?, .. None)
            f_slices = (slice(0, -self.window_size -1),
                        slice(-self.window_size -1, -self.shift_size -1),
                        slice(-self.shift_size -1, -1))  # (-shift_size, -shift_size+?, .. None)
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            if flag:
                h_slices = f_slices
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # (36, 9)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # (36, 9, 9)

            # if flag and self.block_num==1:
            #     num_ = (H_//self.window_size)*(W//self.window_size)
            #     plt.matshow(img_mask[0, :, :, 0].numpy())  # (36, 3, 3)
            #     plt.savefig('savefig/Overview.png', bbox_inches='tight')
            #     for i in range(num_):
            #         plt.matshow(attn_mask[i].numpy())
            #         plt.savefig(f'savefig/figure_{i}.png', bbox_inches = 'tight')
            #     # plt.show()
            #     print()
            #     plt.pause(1)
            #     plt.close()
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

        # padding용 마스크 추가 저장
        H, W = self.input_resolution
        if self.layer_num == 0:  # 입력 형태 달라지는 부분
            H_ = H + 4
            h_slices = (slice(0, 4),)
        else:
            H_ = H + 2
            h_slices = (slice(0, 1), slice(-1, None))

        pad_mask = torch.zeros((1, H_, W, 1))  # 1 H W 1
        for h in h_slices:
            pad_mask[:, h, :, :] = -10

        pad_windows = window_partition(pad_mask, self.window_size)  # nW, window_size, window_size, 1
        pad_windows = pad_windows.view(-1, self.window_size * self.window_size)  # (36=총 윈도우 개수, 9=하나의 윈도우 내 총 패치 개수)
        pad_attn_mask = pad_windows.unsqueeze(1) + pad_windows.unsqueeze(2)
        pad_attn_mask = pad_attn_mask.masked_fill(pad_attn_mask != 0, float(-900.0)).masked_fill(pad_attn_mask == 0, float(0.0))

        # if not flag:
            # num_ = (H_ // self.window_size) * (W // self.window_size)
            # for i in range(num_):
            #     plt.matshow(pad_attn_mask[i].numpy())
            #     # plt.savefig(f'savefig/figure_{i}.png', bbox_inches = 'tight')
            # plt.show()
            # print()
            # plt.pause(1)
            # plt.close()
        self.register_buffer("pad_mask", pad_attn_mask)  # (36, 9, 9)

    def forward(self, x):
        H, W = self.input_resolution  # layer1: (14, 18), layer2: (7, 9)
        B, L, C = x.shape  # layer1: (2, 18*14, 32), layer2: (2, 7*9, 128) (B, Ph*Pw, C=embed_dim)

        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x: norm 전 (2, 252, 32)

        x = x.view(B, H, W, C)  # (2, 14, 18, 32)

        # cyclic shift
        H_ = 0
        if self.shift_size > 0:
            if not self.fused_window_process:
                # x: (2, 14, 18, 32)
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # padding addition
                shifted_x, H_ = self.pad_add(shifted_x, H)
                # (72, 3, 3, 32) = (2x36, 3, 3, 32)
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            # x = (2, 18, 18, 32)
            shifted_x = x  # (batch_size, patchNum_h, patchNum_w, embed_dim)
            shifted_x, H_ = self.pad_add(shifted_x, H)
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            # 1차: (2*6*6=72, 3, 3, 32)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # (72, 9, 32)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask, pad_mask=self.pad_mask)  # nW*B, window_size*window_size, C (72, 9, 32)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # (72=batch*window_num_h*window_num_w, 3, 3, 32)

        # reverse cyclic shift
        if self.shift_size > 0 and self.block_num == 2:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H_, W)  # B H' W' C  (2, 18, 18, 32)
                # 패딩 제거
                shifted_x = self.pad_del(shifted_x, H)
                x = torch.roll(shifted_x, shifts=(self.shift_size*2, self.shift_size*2), dims=(1, 2))  # FIXME 리버스를 두번?
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H_, W)
            shifted_x = self.pad_del(shifted_x, H)
            x = shifted_x

        # residual
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        temp= (self.norm2(x))
        temp = self.mlp(temp)
        x = x + self.drop_path(temp)

        return x

    def pad_add(self, x, H):
        if self.layer_num == 0:  # 위 아래 36개씩
            x = F.pad(x, (0, 0, 0, 0, 4, 0), "constant", 0)
            H_ = H + 4
        else:  # 위 아래 18개씩
            x = F.pad(x, (0, 0, 0, 0, 1, 1), "constant", 0)
            H_ = H + 2
        return x, H_
    
    def pad_del(self, x, H):
        if self.layer_num == 0:
            x = x[:, 4:, :, :]
        else:
            x = x[:, 1:-1, :, :]
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 4 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape  # (1, 252, 32)

        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)  # (1, 14, 18, 32)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C (1, 7, 9, 32)
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C  (1, 7, 9, 128)
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C, (1, 63, 128)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):  # swin trnasformer 의 한 stage의 레이어 : depth 만큼 swin block 생성
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, layer_num=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process, layer_num=layer_num, block_num=i)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):  # x : (480, 3136, 32)
        for blk in self.blocks:  # 3
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:  # Patch Merging
            x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(84, 84), patch_size=(4,4), in_chans=4, embed_dim=32, norm_layer=None):
        super().__init__()
        img_size = img_size
        patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size  # (18, 14)
        self.patches_resolution = patches_resolution  # [14, 18]
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 패치 총 개수= 14*18

        self.in_chans = in_chans  # 1
        self.embed_dim = embed_dim  # 32

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(18, 13), stride=patch_size) # stride=patch_size # (Batch, H, W, in_chans) -> (H/4, W/4, embed_dim)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # (16, 3, 252, 234)
        B, C, H, W = x.shape

        # FIXME 가로로 늘린 결과
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # TODO 여기 값들 바뀌는 추이 확인 필요
        # 밑의 작업 결과 (B * L, Ph * Pw, C)
        # print('파티션 이후: {0}'.format(x))
        x = self.proj(x)  # (2, 32, 14, 18)
        # print('임베딩 이후 {0}'.format(x))
        x= x.flatten(2).transpose(1, 2)  # flatten -> (2, 32, 252), transpose -> (2, 252, 32)
        if self.norm is not None:
            x = self.norm(x)
            # print('norm 이후: {0}'.format(x))
        return x

class SwinTransformer(nn.Module):

    def __init__(self, img_size=(84,84), patch_size=(4,4), in_chans=1, num_classes=4,
                 embed_dim=32, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes  # 4
        self.num_layers = len(depths)  # 3? [2, 2, 2]
        self.embed_dim = embed_dim  # 32
        self.ape = ape  # False, 절대 위치
        self.patch_norm = patch_norm  # True
        self.num_features = int(embed_dim * 4 ** (self.num_layers - 1)) # int(embed_dim * 2 ** (self.num_layers - 1)) 
        self.mlp_ratio = mlp_ratio  # 4

        self.patch_embed = PatchEmbed(
            img_size= img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  # 이미지 내 패치 총 개수
        patches_resolution = self.patch_embed.patches_resolution  # 가로, 세로 별 패치 개수
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)  # pos drop_out 확률은 0%

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 2
            layer = BasicLayer(dim=int(embed_dim * 4 ** i_layer), # int(embed_dim * 2 ** i_layer)
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),  # 1차: (252/18= 14, (252-18)/13= 18)
                               depth=depths[i_layer],  # 3
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,  # patch merging은 한개의 layer 끝날때마다 진행한다
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,
                               layer_num = i_layer)  
            self.layers.append(layer)

        # 수정
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # TODO 멀티 라벨용 추가한것 (230621)
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, Ph*Pw, C) : (2, 252, 32)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        # (1, 63, 128), (1, 252, 32)
        x = self.norm(x)  # B L C -> (2, 63, 128)
        x = self.avgpool(x.transpose(1, 2))  # B C 1 -> (30, 128, 1)
        x = torch.flatten(x, 1)  # (2, 128)

        return x

    def forward(self, x):  # (B =2, C=1, 254, 234)
        x = self.forward_features(x)
        x = self.head(x)
        # TODO 멀티 라벨용 추가한것 (230621)
        # x = self.sigmoid(x)
        return x

