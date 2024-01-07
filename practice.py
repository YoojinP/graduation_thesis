import os
import cv2
import shutil

p = -1
nums = {}
start = -1

# 1: 행동 번호-2, 2: 액션 번호당 딕셔너리 담기, 3: 0번 액션 프레임 구간 확인
# 3-> 2-> 1 순으로 진행
# case = 1
summ =0
cnt = 0
# line_num = 0
folders = [int(item[:-4]) for item in os.listdir('D:/0_pycharm_project/Atari_Challenge_Dataset/full/atari_v1/trajectories/')]
# folders = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19]
for file_name in (folders):
    count = 0
    total_frames = []
    trajectory_path = f'D:/0_pycharm_project/Atari_Challenge_Dataset/full/atari_v1/trajectories/{file_name}.txt' # f"./trajectory_folders/{file_name}.txt"
    f =open(trajectory_path,"r")
    strings = []
    flag = False
    screen_path = f'D:/0_pycharm_project/Atari_Challenge_Dataset/full/atari_v1/screens/{file_name}/'  # 폴더 경로
    screen_destination = './image_folders3/'
    # os.chdir(screen_path)  # 해당 폴더로 이동
    files = os.listdir(screen_path)  # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음

    while True:
        count += 1
        line = f.readline()
        if not line: break
        if count<=2:
            strings.append(line[:-1])
            continue
        frame = int(line.split(',')[0])
        n = int(line.split(',')[-1])

        # line_num += 1
        # if n!=0 : # 액션이 0이 아님
        total_frames.append(frame)
        string = line[:-1]

        if n not in [0,1,2,3,4,5]:
            flag = True
            # break
            if n== 6: #upright
                srting = line[:-2] + "0,1"
            elif n== 7: # upleft
                string = line[:-2] + "0,2"
            elif n== 8: # downright
                string = line[:-2] + "1,3"
            elif n== 9: # downleft
                string = line[:-2] + "1,2"
            else:
                continue
            string = line[:-1]
        else:
            n = n-2
            string = line[:-2] + str(n)
        strings.append(string)

    f.close()

    if flag:
        continue

    # if len(strings)<200:
    #     # print(f"len of string POOR:{len(strings)}")
    #     continue

    summ += len(strings)
    cnt += 1

    os.makedirs(f"{screen_destination}{file_name}", exist_ok=True)
    print("succeed: ", file_name)
    print(len(strings))
    for file in files:  # 이미지 복사 이동
        if '.png' in file and int(file.split('.')[0]) in total_frames:
            source_path = screen_path + file
            destination_path = screen_destination + f"{file_name}/" + file
            shutil.copyfile(source_path, destination_path)

    f = open(f'D:/0_pycharm_project/SwinDT/dec/dec_text3/{file_name}.txt', 'w')
    for s in strings:
        f.write(s + "\n")
    f.close()
print(f"총 개수:{cnt}, 총 길이 :{summ}")


# if case==1:
#     n = n-2
#     string = line[:-2] + str(n)
#     # print(string)
#     strings.append(string)
# elif case==2:  # 눈으로 확인용 --> 어차피 직접 액션 번호 바꿔줘야함
#     if n not in nums.keys():
#         nums[n] = []
#     nums[n].append(frame)
# elif case==3:
#     if n ==0 and start ==-1:
#         start = frame
#     elif n!=0 and start!=-1:
#         end = frame-1
#         # print(f"start:{start}, end:{end}")
#         for i in range(start, end+1):
#           total_frames.append(i)
#         start =-1
#
#     if n!=0 :
#         strings.append(line[:-1])


# if case==2:
#     for k in nums.keys():
#         print(k)
#         print(nums[k])
#
# elif case==3:
#     png_img = []
#     zeros = []
#     for file in files:
#         if '.png' in file and int(file.split('.')[0]) not in zeros:
#             source_path = screen_path + file
#             destination_path = screen_destination + file
#             shutil.copyfile(source_path, destination_path)
#
# elif case==4:
#     f= open(f'C:/Users/SIlab/PycharmProjects/atari_dqn/dec_txt/{file_name}.txt','w')
#     for s in strings:
#         f.write(s+"\n")
#     f.close()
#
#
# elif case==1:

