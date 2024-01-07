import tracemalloc
tracemalloc.start()
my_snapshot = None
import cv2
import random
import numpy as np


import torch

import matplotlib.pyplot as plt
import datetime as dt

scores = [1,23,141,45,1234]
x = dt.datetime.now()
x= x.strftime('%Y-%m-%d')
with open(f'eval-{x}.txt', 'w') as f:
    for s in scores:
        f.write(str(s)+'\n')
exit()
a= torch.Tensor([2.5124341])
print(a)
print(a.item())
exit()
# a = torch.randn(4)
# print(a)
# b = torch.randn(4)
# print(b)
# c = torch.add(a,b)
# print(c)
# a = torch.tensor([[ 239.1250, -117.7500,    4.8164,   15.9062,  -19.6719],
#                 [ 241.1250, -118.7500,    4.8359,   15.9531,  -19.8281],
#                 [ 240.5000, -118.4375,    4.8320,   15.9297,  -19.8438],
#                 [ 237.1250, -116.8125,    4.7227,   15.7109,  -19.4531]], dtype= torch.float32)
# b= torch.argmax(a, dim=1)
# print(b.shape)
# print(a.shape)
# exit()

val_losses = [2.3316738605499268, 2.309549570083618, 2.309549570083618, 1.8890142440795898]
            # [2.3316738605499268, 2.331678867340088, 2.3316636085510254, 2.331669569015503, 2.331662654876709, 2.3316595554351807, 2.331657886505127,
            #   2.331665515899658, 2.3316595554351807, 2.3316760063171387, 2.3316738605499268, 2.3316588401794434, 2.3316619396209717,
            #   2.3316564559936523, 2.331665277481079, 2.331662178039551, 2.3316614627838135, 2.3316617012023926, 2.3316681385040283,2.33166241645813]
train_losses = [1.788864016532898, 1.7013330459594727, 1.6647257804870605, 1.5988378524780273, 1.5354870557785034, 1.4903055429458618,
                1.4764004945755005, 1.4767065048217773, 1.512337327003479, 1.515470027923584, 1.505793809890747, 1.4725145101547241,
                1.4096561670303345, 1.3293640613555908, 1.2692431211471558, 1.2545619010925293, 1.2564738988876343, 1.3102331161499023,
                1.3662594556808472, 1.3890142440795898]
temp = []
for i in range(len(train_losses)):
    for j in range(4):
        temp.append(train_losses[i])

plt.figure(figsize=(7,5))
plt.title("Training and Validation Loss")
# plt.plot(val_losses,label="val")
plt.plot(temp,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

exit()

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


window_size = 3
shift_size = 1
H, W = 18, 18
total = (H//window_size)*(W//window_size)

img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1

# (4, 7, 7, 1)
mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
mask_windows = mask_windows.view(-1, window_size * window_size)  # (4, 49)

attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (4, 49, 49)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

plt.matshow(img_mask[0, :, :, 0].numpy())  # (49, 4, 4) vs  (4, 49, 49)
for i in range(total):
    plt.matshow(attn_mask[i].numpy())
# plt.matshow(attn_mask[12].numpy())
# plt.matshow(attn_mask[24].numpy())
# plt.matshow(attn_mask[48].numpy())

plt.show()


# v = np.array([224]) # (1,1)
# v= np.tile(v, (2, 4)) # (1,2)
# print(v)

exit()


# obss= []
# ar = np.zeros((1,2), int)
# a = np.ones((1,2), int)
# c= np.array([[1,2]], int)
# obss.append(ar)
# obss.append(a)
# obss.append(c)
# obss = np.concatenate(obss, axis=0)
# print(obss)

# cv2_img= cv2.imread('for_wall2.png', cv2.IMREAD_COLOR)
# for i in cv2_img.shape[0]:
#     for j in cv2_img.shape[1]:
#         if cv2_img[i,j]==(0, 24, 124):
#
# exit()

arr= {'a':None, 'b':None, 'c':None}
arr['a'], arr['b'], arr['c'] = 100, 2, 3
print(arr.items())

image = cv2.imread('D://0_pycharm_project//SwinDT//state_ (2).png')
image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY) # (500, 335)
cv2.imshow('raw',image)
cv2.imwrite('raw.png',image)
cv2.waitKey(0)
# 1번 contour 망함
# ret= cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
thr, ret = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)
cv2.imshow("ret", ret)
cv2.waitKey(0)
contours, _ = cv2.findContours(ret, 1, 2)
c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 랜덤 BGR값 생성

for cont in contours:
    x, y, w, h = cv2.boundingRect(cont)
    if 3<w<33 and 3<h<33:   # 3<w<20  and 3< h<20 모든 펠렛 탐지
        # cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)
        ssum= 0
        M =0
        for i in range(w):
            for j in range(h):
                M =M if M > image[y+j, x+i] else image[y+j, x+i]
                # ssum += image[y+j, x+i]
        for i in range(w):
            for j in range(h):
                M =M if M > image[y+j, x+i] else image[y+j, x+i]
                # image[y+j, x+i] =int(ssum/ (w+h)) if image[y+j, x+i]!=32 else 32
        image[y:y+h,x:x+w] = M # int(ssum/ (w+h))
        # image = cv2.drawContours(image, cont, 1, (255, 255, 255), 2)

cv2.imshow('draw_contour',image)
cv2.imwrite('ds.png', image)
cv2.waitKey()

# num = 0.000298471
# print(num)
# print(format(num,'.4f'))

# x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(2,4)
# print(x)

# x = torch.roll(x, shifts = 1) # dim이 정해져 있지 않으면 평평해진 후 이동한 다음 다시 원래 모양으로 복원
# x= torch.roll(x, 1, 0)
# x= torch.roll(x, -1, 0)
# x= torch.roll(x, shifts=(-1,1), dims=(1,0))
# dim =0 은 맨 왼쪽부터 시작
# shape에서 몇개가 들어있는 곳인지 확인하고 말그래도 shifts 수만큼 이동
# print(x)

# data = torch.randn((5, 5))
# print(data)
# print(torch.max(data, dim=0))

# x = torch.tensor([1, 2])
# y = torch.tensor([4, 5, 6])
# grid_x, grid_y = torch.meshgrid([x, y])
# print(grid_x)
# print(grid_x.shape)
#
# print(grid_y)
# print(grid_y.shape)

# t4d = torch.ones(3, 3, 4, 2) #쉽게 이미지를 생각하자 (batch, channel, height, width)
# p1d = (1, 1, 1, 1) # pad last dim by 1 on each side, width의 왼쪽 오른쪽에 pad 추가
# out = F.pad(t4d, p1d, "constant", 2) # effectively zero padding,
# print(out.size())
# print(out)


####
# tensor([[-1.1575e+00, -1.8878e-01,  4.2871e-01, -2.0162e+00, -3.4006e-01],
#         [ 6.3486e-01, -8.0135e-01,  2.4311e-01,  3.9699e-01, -1.3349e+00],
#         [ 2.6045e-01,  1.2943e+00,  6.0914e-01, -2.7016e-03, -1.0328e+00],
#         [-1.0054e+00,  3.4493e-01,  2.9346e+00, -3.3168e-01,  2.2873e-01],
#         [-9.0424e-02, -9.4272e-01, -4.7905e-01,  5.4701e-01, -2.0194e+00]])

####
# torch.return_types.max(
# values=tensor([0.4287, 0.6349, 1.2943, 2.9346, 0.5470]),
# indices=tensor([2, 0, 1, 2, 3]))
'''
a = torch.tensor([[[0, 1], [2, 3], [4, 5]], \
                 [[6, 7], [8, 9], [10, 11]], \
                 [[12, 13], [14, 15], [16, 17]], \
                 [[18, 19], [20, 21], [22, 23]]])  # (4, 3, 2)


b = a.view(4, 3, 2)
c = a.reshape(4, 3, 2)

print(b)
print(c)

a = torch.tensor([1,2,3,4,5,6]).view(3,2)
b = torch.tensor([9,8,7,6,5,4]).view(2,3)
ab = torch.matmul(a,b)
cd = a@b # @ 연산자를 이용하여 간단하게 행렬곱을 표현할 수 있음

print(ab==cd)


As = torch.randn(3,2,5)
Bs = torch.randn(3,5,2)
Cs = torch.einsum('btj,bjk->btk', As, Bs)

print(Cs.shape)

Ds = As @ Bs
print(Cs == Ds)
'''


























