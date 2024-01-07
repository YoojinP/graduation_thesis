import ctypes
import os
import sys
import matplotlib.pyplot as plt
import configparser
config = configparser.ConfigParser(interpolation=None)

path = os.path.dirname(os.path.realpath(sys.argv[0]))
config.read(os.path.join(path, 'graph_setting.ini'))

# x1 = config['loss_graph']['x']
# y1 = config['loss_graph']['y']
x1_min = int(config['loss_graph']['x_min'])
x1_max = int(config['loss_graph']['x_max'])
y1_min = int(config['loss_graph']['y_min'])
y1_max = int(config['loss_graph']['y_max'])
x_label1 = config['loss_graph']['x_label']
y_label1 = config['loss_graph']['y_label']
window_title1 = config['loss_graph']['window_title']
title1 = config['loss_graph']['title']
line_color1 = config['loss_graph']['line_color']

# x2 = config['heatmap']['x']
# y2 = config['heatmap']['y']
x2_min = int(config['heatmap']['x_min'])
x2_max = int(config['heatmap']['x_max'])
y2_min = float(config['heatmap']['y_min'])
y2_max = float(config['heatmap']['y_max'])
x_label2 = config['heatmap']['x_label']
y_label2 = config['heatmap']['y_label']
line_color2 = config['heatmap']['line_color']
alpha2 = float(config['heatmap']['alpha'])
window_title2 = config['heatmap']['window_title']
# title2 = config['heatmap']['title']

# x3 = config['random_action_graph']['x']
# y3 = config['random_action_graph']['y']
x3_min = int(config['random_action_graph']['x_min'])
x3_max = int(config['random_action_graph']['x_max'])
y3_min = int(config['random_action_graph']['y_min'])
y3_max = int(config['random_action_graph']['y_max'])
x_label3 = config['random_action_graph']['x_label']
y_label3 = config['random_action_graph']['y_label']
window_title3 = config['random_action_graph']['window_title']
title3 = config['random_action_graph']['title']
line_color3 = config['random_action_graph']['line_color']

# x4 = config['step_graph']['x']
# y4 = config['step_graph']['y']
x_label4 = config['step_graph']['x_label']
y_label4 = config['step_graph']['y_label']
x4_min = int(config['step_graph']['x_min'])
x4_max = int(config['step_graph']['x_max'])
y4_min = int(config['step_graph']['y_min'])
y4_max = int(config['step_graph']['y_max'])
window_title4 = config['step_graph']['window_title']
title4 = config['step_graph']['title']
line_color4 = config['step_graph']['line_color']

# x5 = config['total_reward_graph']['x']
# y5 = config['total_reward_graph']['y']
x_label5 = config['total_reward_graph']['x_label']
y_label5 = config['total_reward_graph']['y_label']
x5_min = int(config['total_reward_graph']['x_min'])
x5_max = int(config['total_reward_graph']['x_max'])
y5_min = int(config['total_reward_graph']['y_min'])
y5_max = int(config['total_reward_graph']['y_max'])
window_title5 = config['total_reward_graph']['window_title']
title5 = config['total_reward_graph']['title']
line_color5 = config['total_reward_graph']['line_color']

# x6 = config['progress_graph']['x']
# y6 = config['progress_graph']['y']
x_label6 = config['progress_graph']['x_label']
y_label6 = config['progress_graph']['y_label']
x6_min = int(config['progress_graph']['x_min'])
x6_max = int(config['progress_graph']['x_max'])
y6_min = int(config['progress_graph']['y_min'])
y6_max = int(config['progress_graph']['y_max'])
window_title6 = config['progress_graph']['window_title']
title6 = config['progress_graph']['title']
line_color6 = config['progress_graph']['line_color']

# x7 = config['duration_graph']['x']
# y7 = config['duration_graph']['y']
x_label7 = config['duration_graph']['x_label']
y_label7 = config['duration_graph']['y_label']
x7_min = int(config['duration_graph']['x_min'])
x7_max = int(config['duration_graph']['x_max'])
y7_min = int(config['duration_graph']['y_min'])
y7_max = int(config['duration_graph']['y_max'])
window_title7 = config['duration_graph']['window_title']
title7 = config['duration_graph']['title']
line_color7 = config['duration_graph']['line_color']

# x18 = config['train_time_graph']['x']
# y18 = config['train_time_graph']['y']
x_label18 = config['train_time_graph']['x_label']
y_label18 = config['train_time_graph']['y_label']
x18_min = int(config['train_time_graph']['x_min'])
x18_max = int(config['train_time_graph']['x_max'])
y18_min = int(config['train_time_graph']['y_min'])
y18_max = int(config['train_time_graph']['y_max'])
window_title18 = config['train_time_graph']['window_title']
title18 = config['train_time_graph']['title']
line_color18 = config['train_time_graph']['line_color']

# x8 = config['three_point_graph']['x']
# y8 = config['three_point_graph']['y']
x_label8 = config['three_point_graph']['x_label']
y_label8 = config['three_point_graph']['y_label']
x8_min = int(config['three_point_graph']['x_min'])
x8_max = int(config['three_point_graph']['x_max'])
y8_min = int(config['three_point_graph']['y_min'])
y8_max = int(config['three_point_graph']['y_max'])
window_title8 = config['three_point_graph']['window_title']
title8 = config['three_point_graph']['title']
line_color8 = config['three_point_graph']['line_color']

# x9 = config['three_point_graph']['x']
# y9 = config['three_point_graph']['y']
x_label9 = config['two_point_graph']['x_label']
y_label9 = config['two_point_graph']['y_label']
x9_min = int(config['two_point_graph']['x_min'])
x9_max = int(config['two_point_graph']['x_max'])
y9_min = int(config['two_point_graph']['y_min'])
y9_max = int(config['two_point_graph']['y_max'])
window_title9 = config['two_point_graph']['window_title']
title9 = config['two_point_graph']['title']
line_color9 = config['two_point_graph']['line_color']

# x10 = config['break_through_graph']['x']
# y10 = config['break_through_graph']['y']
x_label10 = config['break_through_graph']['x_label']
y_label10 = config['break_through_graph']['y_label']
x10_min = int(config['break_through_graph']['x_min'])
x10_max = int(config['break_through_graph']['x_max'])
y10_min = int(config['break_through_graph']['y_min'])
y10_max = int(config['break_through_graph']['y_max'])
window_title10 = config['break_through_graph']['window_title']
title10 = config['break_through_graph']['title']
line_color10 = config['break_through_graph']['line_color']

# x11 = config['steal_graph']['x']
# y11 = config['steal_graph']['y']
x_label11 = config['steal_graph']['x_label']
y_label11 = config['steal_graph']['y_label']
x11_min = int(config['steal_graph']['x_min'])
x11_max = int(config['steal_graph']['x_max'])
y11_min = int(config['steal_graph']['y_min'])
y11_max = int(config['steal_graph']['y_max'])
window_title11 = config['steal_graph']['window_title']
title11 = config['steal_graph']['title']
line_color11 = config['steal_graph']['line_color']

# x12 = config['block_graph']['x']
# y12 = config['block_graph']['y']
x_label12 = config['block_graph']['x_label']
y_label12 = config['block_graph']['y_label']
x12_min = int(config['block_graph']['x_min'])
x12_max = int(config['block_graph']['x_max'])
y12_min = int(config['block_graph']['y_min'])
y12_max = int(config['block_graph']['y_max'])
window_title12 = config['block_graph']['window_title']
title12 = config['block_graph']['title']
line_color12 = config['block_graph']['line_color']

# x13 = config['rebound_graph']['x']
# y13 = config['rebound_graph']['y']
x_label13 = config['rebound_graph']['x_label']
y_label13 = config['rebound_graph']['y_label']
x13_min = int(config['rebound_graph']['x_min'])
x13_max = int(config['rebound_graph']['x_max'])
y13_min = int(config['rebound_graph']['y_min'])
y13_max = int(config['rebound_graph']['y_max'])
window_title13 = config['rebound_graph']['window_title']
title13 = config['rebound_graph']['title']
line_color13 = config['rebound_graph']['line_color']

# x14 = config['chipout_graph']['x']
# y14 = config['chipout_graph']['y']
x_label14 = config['chipout_graph']['x_label']
y_label14 = config['chipout_graph']['y_label']
x14_min = int(config['chipout_graph']['x_min'])
x14_max = int(config['chipout_graph']['x_max'])
y14_min = int(config['chipout_graph']['y_min'])
y14_max = int(config['chipout_graph']['y_max'])
window_title14 = config['chipout_graph']['window_title']
title14 = config['chipout_graph']['title']
line_color14 = config['chipout_graph']['line_color']

# x15 = config['opponent_graph']['x']
# y15 = config['opponent_graph']['y']
x_label15 = config['opponent_graph']['x_label']
y_label15 = config['opponent_graph']['y_label']
x15_min = int(config['opponent_graph']['x_min'])
x15_max = int(config['opponent_graph']['x_max'])
y15_min = int(config['opponent_graph']['y_min'])
y15_max = int(config['opponent_graph']['y_max'])
window_title15 = config['opponent_graph']['window_title']
title15 = config['opponent_graph']['title']
line_color15 = config['opponent_graph']['line_color']

# x16 = config['stage_clear_graph']['x']
# y16 = config['stage_clear_graph']['y']
x_label16 = config['stage_clear_graph']['x_label']
y_label16 = config['stage_clear_graph']['y_label']
x16_min = int(config['stage_clear_graph']['x_min'])
x16_max = int(config['stage_clear_graph']['x_max'])
y16_min = int(config['stage_clear_graph']['y_min'])
y16_max = int(config['stage_clear_graph']['y_max'])
window_title16 = config['stage_clear_graph']['window_title']
title16 = config['stage_clear_graph']['title']
line_color16 = config['stage_clear_graph']['line_color']

# x17 = config['stage_clear_step_graph']['x']
# y17 = config['stage_clear_step_graph']['y']
x_label17 = config['stage_clear_step_graph']['x_label']
y_label17 = config['stage_clear_step_graph']['y_label']
x17_min = int(config['stage_clear_step_graph']['x_min'])
x17_max = int(config['stage_clear_step_graph']['x_max'])
y17_min = int(config['stage_clear_step_graph']['y_min'])
y17_max = int(config['stage_clear_step_graph']['y_max'])
window_title17 = config['stage_clear_step_graph']['window_title']
title17 = config['stage_clear_step_graph']['title']
line_color17 = config['stage_clear_step_graph']['line_color']


# window = Tk()
# monitor_h = window.winfo_screenheight()
# monitor_w = window.winfo_screenwidth()
# print("width x height = %d x %d (pixels)" %(monitor_w/4, monitor_h/4))

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
screen_w = screensize[0]
screen_h = screensize[1] - 100
# print("width x height = %d x %d (pixels)" %(screen_w, screen_h))


def loss_graph(episode_count, loss, fix):
    x = episode_count
    y = loss
    f = plt.figure(window_title1, figsize=(5, 4))
    move_figure(f, 0, 150)
    plt.plot(x, y, color=line_color1)
    plt.title(title1)
    if fix:
        plt.xlim(x1_min, x1_max)
        plt.ylim(y1_min, y1_max)
    plt.xlabel(x_label1)
    plt.ylabel(y_label1)
    plt.tight_layout()


def heatmap(episode_count, coordinate):
    coord_x = []
    coord_y = []

    for i in coordinate:
        for j in i:
            coord_x.append(j[0])
            coord_y.append(j[1])

    f = plt.figure(window_title2, figsize=(5, 4))
    move_figure(f, screen_w/4, 150)
    plt.grid(True)
    plt.scatter(coord_x, coord_y, alpha=alpha2, color=line_color2)
    plt.xlabel(x_label2)
    plt.ylabel(y_label2)
    plt.xlim(x2_min, x2_max)
    plt.ylim(y2_min, y2_max * -1)
    plt.title('Agent\'s Visited Locations over {} Episodes'.format(episode_count[-1]), fontsize=12)
    plt.tight_layout()


def random_action_graph(episode_count, random_action_prob, fix):
    x = episode_count
    y = random_action_prob

    f = plt.figure(window_title3, figsize=(5, 4))
    move_figure(f, 2 * (screen_w/4), 150)
    plt.plot(x, y, color=line_color3)
    plt.title(title3)
    plt.xlabel(x_label3)
    plt.ylabel(y_label3)
    if fix:
        plt.xlim(x3_min, x3_max)
        plt.ylim(y3_min, y3_max)
    plt.tight_layout()


def step_graph(episode_count, episode_steps, fix):
    x = episode_count
    y = episode_steps

    f = plt.figure(window_title4, figsize=(5, 4))
    move_figure(f, 3 * (screen_w/4), 150)
    plt.plot(x, y, color=line_color4)
    plt.title(title4)
    if fix:
        plt.xlim(x4_min, x4_max)
        plt.ylim(y4_min, y4_max)
    plt.xlabel(x_label4)
    plt.ylabel(y_label4)
    plt.tight_layout()


def total_reward_graph(episode_count, episode_reward, fix):
    x = episode_count
    y = episode_reward

    f = plt.figure(window_title5, figsize=(5, 4))
    move_figure(f, 0, 150+((screen_h-150)/2))
    plt.plot(x, y, color=line_color5)
    plt.title(title5)
    if fix:
        plt.xlim(x5_min, x5_max)
        plt.ylim(y5_min, y5_max)
    plt.xlabel(x_label5)
    plt.ylabel(y_label5)
    plt.tight_layout()


def progress_graph(episode_count, progress, fix):
    x = episode_count
    y = progress

    f = plt.figure(window_title6, figsize=(5, 4))
    move_figure(f, screen_w/4, 150+((screen_h-150)/2))
    plt.plot(x, y, color=line_color6)
    plt.title(title6)
    if fix:
        plt.xlim(x6_min, x6_max)
        plt.ylim(y6_min, y6_max)
    plt.xlabel(x_label6)
    plt.ylabel(y_label6)
    plt.tight_layout()


def duration_graph(episode_count, duration, fix):
    x = episode_count
    y = duration

    f = plt.figure(window_title7, figsize=(5, 4))
    move_figure(f, 2 * (screen_w/4), 150+((screen_h-150)/2))
    plt.plot(x, y, color=line_color7)
    plt.title(title7)
    if fix:
        plt.xlim(x7_min, x7_max)
        plt.ylim(y7_min, y7_max)
    plt.xlabel(x_label7)
    plt.ylabel(y_label7)
    plt.tight_layout()


def train_time_graph(episode_count, train_time, fix):
    x = episode_count
    y = train_time

    f = plt.figure(window_title18, figsize=(5, 4))
    move_figure(f, screen_w/4, 150)
    plt.plot(x, y, color=line_color18)
    plt.title(title18)
    if fix:
        plt.xlim(x18_min, x18_max)
        plt.ylim(y18_min, y18_max)
    plt.xlabel(x_label18)
    plt.ylabel(y_label18)
    plt.tight_layout()


def three_point_graph(episode_count, three_point_reward, fix):
    x = episode_count
    y = three_point_reward

    f = plt.figure(window_title8, figsize=(5, 4))
    move_figure(f, 3 * (screen_w/4), 150+((screen_h-150)/2))
    plt.plot(x, y, color=line_color8)
    plt.title(title8)
    plt.xlabel(x_label8)
    plt.ylabel(y_label8)
    if fix:
        plt.xlim(x8_min, x8_max)
        plt.ylim(y8_min, y8_max)
    plt.tight_layout()


def two_point_graph(episode_count, two_point_reward, fix):
    x = episode_count
    y = two_point_reward

    f = plt.figure(window_title9, figsize=(5, 4))
    move_figure(f, 0, 150)
    plt.plot(x, y, color=line_color9)
    plt.title(title9)
    plt.xlabel(x_label9)
    plt.ylabel(y_label9)
    if fix:
        plt.xlim(x9_min, x9_max)
        plt.ylim(y9_min, y9_max)
    plt.tight_layout()


def break_through_graph(episode_count, break_through_reward, fix):
    x = episode_count
    y = break_through_reward

    f = plt.figure(window_title10, figsize=(5, 4))
    move_figure(f, screen_w/4, 150)
    plt.plot(x, y, color=line_color10)
    plt.title(title10)
    plt.xlabel(x_label10)
    plt.ylabel(y_label10)
    if fix:
        plt.xlim(x10_min, x10_max)
        plt.ylim(y10_min, y10_max)
    plt.tight_layout()


def steal_graph(episode_count, steal_reward, fix):
    x = episode_count
    y = steal_reward

    f = plt.figure(window_title11, figsize=(5, 4))
    move_figure(f, 2 * (screen_w/4), 150)
    plt.plot(x, y, color=line_color11)
    plt.title(title3)
    plt.xlabel(x_label11)
    plt.ylabel(y_label11)
    if fix:
        plt.xlim(x11_min, x11_max)
        plt.ylim(y11_min, y11_max)
    plt.tight_layout()


def block_graph(episode_count, block_reward, fix):
    x = episode_count
    y = block_reward

    f = plt.figure(window_title12, figsize=(5, 4))
    move_figure(f, 3 * (screen_w/4), 150)
    plt.plot(x, y, color=line_color12)
    plt.title(title12)
    if fix:
        plt.xlim(x12_min, x12_max)
        plt.ylim(y12_min, y12_max)
    plt.xlabel(x_label12)
    plt.ylabel(y_label12)
    plt.tight_layout()


def rebound_graph(episode_count, rebound_reward, fix):
    x = episode_count
    y = rebound_reward

    f = plt.figure(window_title13, figsize=(5, 4))
    move_figure(f, 0, 150+((screen_h-150)/2))
    plt.plot(x, y, color=line_color13)
    plt.title(title13)
    if fix:
        plt.xlim(x13_min, x13_max)
        plt.ylim(y13_min, y13_max)
    plt.xlabel(x_label13)
    plt.ylabel(y_label13)
    plt.tight_layout()


def chipout_graph(episode_count, chipout_reward, fix):
    x = episode_count
    y = chipout_reward

    f = plt.figure(window_title14, figsize=(5, 4))
    move_figure(f, screen_w/4, 150+((screen_h-150)/2))
    plt.plot(x, y, color=line_color14)
    plt.title(title14)
    if fix:
        plt.xlim(x14_min, x14_max)
        plt.ylim(y14_min, y14_max)
    plt.xlabel(x_label14)
    plt.ylabel(y_label14)
    plt.tight_layout()


def opponent_graph(episode_count, opponent_reward, fix):
    x = episode_count
    y = opponent_reward

    f = plt.figure(window_title15, figsize=(5, 4))
    move_figure(f, 2 * (screen_w/4), 150+((screen_h-150)/2))
    plt.plot(x, y, color=line_color15)
    plt.title(title15)
    if fix:
        plt.xlim(x15_min, x15_max)
        plt.ylim(y15_min, y15_max)
    plt.xlabel(x_label15)
    plt.ylabel(y_label15)
    plt.tight_layout()


def stage_clear_graph(episode_count, done_stage, fix):
    clear_x = []
    clear_y = []

    for i in range(len(episode_count)):
        if done_stage[i]:
            clear_x.append(episode_count[i])
            clear_y.append(done_stage[i])

    f = plt.figure(window_title16, figsize=(5, 4))
    move_figure(f, 3 * (screen_w/4), 150+((screen_h-150)/2))
    plt.scatter(clear_x, clear_y, s=5, c=line_color16)
    plt.title(title16)
    plt.xlabel(x_label16)
    plt.ylabel(y_label16)
    if fix:
        plt.xlim(x16_min, x16_max)
        plt.ylim(y16_min, y16_max)
    plt.yticks([0, 1], labels=['Fail', 'Clear'])
    plt.tight_layout()


def stage_clear_step_graph(episode_count, clear_steps, fix):
    clear_step_x = episode_count
    clear_step_y = clear_steps

    f = plt.figure(window_title17, figsize=(5, 4))
    move_figure(f, 0, 150)
    plt.plot(clear_step_x, clear_step_y, color=line_color17)
    plt.title(title17)
    if fix:
        plt.xlim(x17_min, x17_max)
        plt.ylim(y17_min, y17_max)
    plt.xlabel(x_label17)
    plt.ylabel(y_label17)
    plt.tight_layout()


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = plt.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
