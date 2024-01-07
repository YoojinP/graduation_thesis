import os
import shutil

total_actions = {}
folders = [int(item[:-4]) for item in os.listdir('D:/0_pycharm_project/Atari_Challenge_Dataset/full/trajectories/')]
for file_name in (folders):
    count = 0
    nums = {}
    total_frames = []
    trajectory_path = f"D:/0_pycharm_project/Atari_Challenge_Dataset/full/trajectories/{file_name}.txt"
    f =open(trajectory_path,"r")
    strings = []

    while True:
        count += 1
        line = f.readline()
        if not line: break
        if count<=2:
            # strings.append(line[:-1])
            continue

        frame = int(line.split(',')[0])
        n = int(line.split(',')[-1])
        if n not in nums.keys():
            nums[n] = []
        nums[n].append(frame)

        if int(n) not in total_actions.keys():
            total_actions[int(n)] = 0

    # print(file_name)
    for key in sorted(nums.keys()):
        # print(f'{key} : {nums[key]}')
        total_actions[key] += len(nums[key])
    # print("-----------------------------------")

total_count = 0
for key in sorted(total_actions.keys()):
    # print(f'{key} : {total_actions[key]}')
    total_count += total_actions[key]

print(total_count)