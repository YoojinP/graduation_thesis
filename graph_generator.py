import tkinter.ttk
from graph_function import *


from tkinter import *
from tkinter.font import Font
from tkinter import filedialog
import tkinter.messagebox as mbox


window = Tk()
window.title("Graph Generator from SI Lab")
# monitor_height = int(window.winfo_screenheight() / 2)
monitor_width = window.winfo_screenwidth()/3
window.geometry('650x225+%d+0' % monitor_width)
window.resizable(False, False)
window.minsize(650, 250)
window.maxsize(650, 250)


def file_find():
    file = filedialog.askopenfilename(title='Select Log File', filetypes=(('log', '*.log'), ('all files', '*.*')))
    en_filepath.delete(0, END)
    en_filepath.insert(END, file)
    bt_read_checker.set(False)


# 파일 읽기 체크
bt_read_checker = BooleanVar()

Entry_frame = Frame(window)
Entry_frame.pack(side='top', fill='x')

# 빈 칸 생성
en_filepath = Entry(Entry_frame, width=87)
en_filepath.pack(side='left', padx=5, pady=5)

# 파일 찾기
u = Font(weight='bold', size=9)
bt_upload = Button(Entry_frame, text="...", width=2, height=1, command=file_find)
bt_upload['font'] = u
bt_upload.pack(side='left', padx=3)
#
# # # 파일 읽기(확인용)
# # progress_checker = IntVar()
# # cur_bar = tkinter.ttk.Progressbar(New_window, width=50, variable=progress_checker)  # TODO: New_window 생성
# # cur_bar.grid(row=0, column=0, columnspan=50)
#
# Group A
# Blank1 = Frame(window)
# Blank1.pack(side='top', fill='x')
# Blank_label1 = Label(Blank1)
# Blank_label1.pack()

Blank1 = Frame(window)
Blank1.pack(side='top', fill='x')
Blank_label1 = Label(Blank1, text='Group A: ', font=Font(size=10))
Blank_label1.pack(side='left', fill='x', padx=5)

# Check_A = Frame(window)
# Check_A.pack(side='top', fill='x')

# 체크 박스 생성 (loss)
loss_checker = BooleanVar()
loss_checker.set(True)
loss_cb = Checkbutton(Blank1, text="Loss", variable=loss_checker)
loss_cb.pack(side='left', fill='x',  padx=5)

# 체크 박스 생성 (location)
location_checker = BooleanVar()
location_checker.set(True)
location_cb = Checkbutton(Blank1, text="Heatmap", variable=location_checker)
location_cb.pack(side='left', padx=5)

# 체크 박스 생성 (random_action_prob)
random_action_prob_checker = BooleanVar()
random_action_prob_checker.set(True)
action_perc_cb = Checkbutton(Blank1, text="Random Action(%)", variable=random_action_prob_checker)
action_perc_cb.pack(side='left', padx=5)


# Group B
Blank3 = Frame(window)
Blank3.pack(side='top', fill='x')
Blank_label3 = Label(Blank3, text='Group B: ', font=Font(size=10))
Blank_label3.pack(side='left', fill='x', padx=5)
#
# Check_B = Frame(window)
# Check_B.pack(side='top', fill='x')

# 체크 박스 생성 (steps)
steps_checker = BooleanVar()
steps_checker.set(True)
steps_cb = Checkbutton(Blank3, text="Steps", variable=steps_checker)
steps_cb.pack(side='left', padx=5)

# 체크 박스 생성 (total reward)
total_reward_checker = BooleanVar()
total_reward_checker.set(True)
total_reward_cb = Checkbutton(Blank3, text="Total Rewards", variable=total_reward_checker)
total_reward_cb.pack(side='left', padx=5)

# 체크 박스 생성 (progress)
progress_checker = BooleanVar()
progress_checker.set(True)
action_progress = Checkbutton(Blank3, text="Number of Passes", variable=progress_checker)
action_progress.pack(side='left', padx=5)

# Group C
# Blank4 = Frame(window)
# Blank4.pack(side='top', fill='x')
# Blank_label4 = Label(Blank4)
# Blank_label4.pack()

Blank5 = Frame(window)
Blank5.pack(side='top', fill='x')
Blank_label5 = Label(Blank5, text='Group C: ', font=Font(size=10))
Blank_label5.pack(side='left', fill='x', padx=5)

Check_C = Frame(window)
Check_C.pack(side='top', fill='x')

# 체크 박스 생성 (duration)
duration_checker = BooleanVar()
duration_checker.set(True)
duration_cb = Checkbutton(Blank5, text="Durations(ms)", variable=duration_checker)
duration_cb.pack(side='left', padx=5)

# 체크 박스 생성 (train time)
TrainTime_checker = BooleanVar()
TrainTime_checker.set(True)
TrainTime_cb = Checkbutton(Blank5, text="Train Time(minutes)", variable=TrainTime_checker)
TrainTime_cb.pack(side='left', padx=5)

# 체크 박스 생성 (Three Point Reward)
ThreePoint_checker = BooleanVar()
ThreePoint_checker.set(False)
ThreePoint_cb = Checkbutton(Blank5, text="Three Point Rewards", variable=ThreePoint_checker)
ThreePoint_cb.pack(side='left', padx=5)

Blank5_2 = Frame(window)
Blank5_2.pack(side='top', fill='x')
Blank_label5_2 = Label(Blank5_2, text='               ')
Blank_label5_2.pack(side='left', fill='x', padx=5)

# 체크 박스 생성 (Two Point Reward)
TwoPoint_checker = BooleanVar()
TwoPoint_checker.set(False)
TwoPoint_cb = Checkbutton(Blank5_2, text="Two Point Rewards", variable=TwoPoint_checker)
TwoPoint_cb.pack(side='left', padx=5)

# 체크 박스 생성 (Break Through Reward)
break_through_checker = BooleanVar()
break_through_checker.set(False)
break_through_cb = Checkbutton(Blank5_2, text="Break Through Rewards", variable=break_through_checker)
break_through_cb.pack(side='left', padx=5)

# 체크 박스 생성 (Steal Reward)
steal_checker = BooleanVar()
steal_checker.set(False)
steal_cb = Checkbutton(Blank5_2, text="Steal Rewards", variable=steal_checker)
steal_cb.pack(side='left', padx=5)

Blank5_3 = Frame(window)
Blank5_3.pack(side='top', fill='x')
Blank_label5_3 = Label(Blank5_3, text='               ')
Blank_label5_3.pack(side='left', fill='x', padx=5)

# 체크 박스 생성 (Block Reward)
block_checker = BooleanVar()
block_checker.set(False)
block_cb = Checkbutton(Blank5_3, text="Block Rewards", variable=block_checker)
block_cb.pack(side='left', padx=5)

# 체크 박스 생성 (Steal Reward)
rebound_checker = BooleanVar()
rebound_checker.set(False)
rebound_cb = Checkbutton(Blank5_3, text="Rebound Rewards", variable=rebound_checker)
rebound_cb.pack(side='left', padx=5)

# 체크 박스 생성 (Block Reward)
chipout_checker = BooleanVar()
chipout_checker.set(False)
chipout_cb = Checkbutton(Blank5_3, text="Chip Out Rewards", variable=chipout_checker)
chipout_cb.pack(side='left', padx=5)

Blank5_4 = Frame(window)
Blank5_4.pack(side='top', fill='x')
Blank_label5_4 = Label(Blank5_4, text='               ')
Blank_label5_4.pack(side='left', fill='x', padx=5)

# 체크 박스 생성 (Block Reward)
opponent_checker = BooleanVar()
opponent_checker.set(False)
opponent_cb = Checkbutton(Blank5_4, text="Opponent Rewards", variable=opponent_checker)
opponent_cb.pack(side='left', padx=5)

# Group D
# Blank6 = Frame(window)
# Blank6.pack(side='top', fill='x')
# Blank_label6 = Label(Blank6)
# Blank_label6.pack()

Blank7 = Frame(window)
Blank7.pack(side='top', fill='x')
Blank_label7 = Label(Blank7, text='Group D: ', font=Font(size=10))
Blank_label7.pack(side='left', fill='x', padx=5)

# Check_D = Frame(window)
# Check_D.pack(side='top', fill='x')

# 체크 박스 생성 (stage_clear)
stage_clear_checker = BooleanVar()
stage_clear_checker.set(True)
clear_cb = Checkbutton(Blank7, text="Stage Clears", variable=stage_clear_checker)
clear_cb.pack(side='left', padx=5)

# 체크 박스 생성 (clear)
stage_clear_steps_checker = BooleanVar()
stage_clear_steps_checker.set(True)
clear_cb = Checkbutton(Blank7, text="Stage Clear Steps", variable=stage_clear_steps_checker)
clear_cb.pack(side='left', padx=5)

Blank8 = Frame(window)
Blank8.pack(side='top', fill='x')
Blank_label8 = Label(Blank8)
Blank_label8.pack()

# Option
Blank9 = Frame(window)
Blank9.pack(side='top', fill='x')
Blank_label9 = Label(Blank9, text='[Option]:', font=('Arial Bold', 10))
Blank_label9.pack(side='left', fill='x', padx=5)

# Fix axis range
fix_axis_checker = BooleanVar()
fix_axis_checker.set(False)
fix_axis_cb = Checkbutton(Blank9, text="Fix the Axis Ranges ", variable=fix_axis_checker)
fix_axis_cb.pack(side='left', padx=5)


# 전체 그래프 생성
def make_generator():
    File = {
        'episode': [],
        'loss': [],
        'coordinate': [],
        'random_action_prob': [],
        'steps': [],
        'total_reward': [],
        'number_of_passes': [],
        'duration': [],
        'train_time': [],
        'three_point_reward': [],
        'two_point_reward': [],
        'break_through_reward': [],
        'steal_reward': [],
        'block_reward': [],
        'rebound_reward': [],
        'chipout_reward': [],
        'opponent_reward': [],
        'stage_clear': [],
        'stage_clear_steps': []
    }

    if len(en_filepath.get()) == 0:
        mbox.showinfo("warning", "Select a file.")
        return
    else:
        file_loc = en_filepath.get()
        try:
            with open(file_loc, 'r', encoding='UTF-8') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line[:-1]
                    dict_log = eval(line)
                    for key, value in dict_log.items():
                        if value is not None:
                            File[key].append(value)
        except:
            mbox.showinfo("warning", "The path is invalid or the file format is invalid.")

        plt.close('all')

        if fix_axis_checker.get():
            fix = True
            if ThreePoint_checker.get() and File['three_point_reward']:
                three_point_graph(File["episode"], File["three_point_reward"], fix)
            if TwoPoint_checker.get() and File['two_point_reward']:
                two_point_graph(File["episode"], File["two_point_reward"], fix)
            if break_through_checker.get() and File['break_through_reward']:
                break_through_graph(File["episode"], File["break_through_reward"], fix)
            if steal_checker.get() and File['steal_reward']:
                steal_graph(File["episode"], File["steal_reward"], fix)
            if block_checker.get() and File['block_reward']:
                block_graph(File["episode"], File["block_reward"], fix)
            if rebound_checker.get() and File['rebound_reward']:
                rebound_graph(File["episode"], File["rebound_reward"], fix)
            if chipout_checker.get() and File['chipout_reward']:
                chipout_graph(File["episode"], File["chipout_reward"], fix)
            if opponent_checker.get() and File['opponent_reward']:
                opponent_graph(File["episode"], File["opponent_reward"], fix)
            if stage_clear_checker.get() and File['stage_clear']:
                stage_clear_graph(File["episode"], File["stage_clear"], fix)
            if stage_clear_steps_checker.get() and File["stage_clear_steps"]:
                stage_clear_step_graph(File["episode"], File["stage_clear_steps"], fix)
            if loss_checker.get() and File['loss']:
                loss_graph(File["episode"], File["loss"], fix)
            if location_checker.get() and File['coordinate']:
                heatmap(File["episode"], File["coordinate"])
            if random_action_prob_checker.get() and File['random_action_prob']:
                random_action_graph(File["episode"], File["random_action_prob"], fix)
            if steps_checker.get() and File['steps']:
                step_graph(File["episode"], File["steps"], fix)
            if total_reward_checker.get() and File['total_reward']:
                total_reward_graph(File["episode"], File["total_reward"], fix)
            if progress_checker.get() and File["number_of_passes"]:
                progress_graph(File["episode"], File["number_of_passes"], fix)
            if duration_checker.get() and File['duration']:
                duration_graph(File['episode'], File['duration'], fix)
            if TrainTime_checker.get() and File['train_time']:
                train_time_graph(File["episode"], File["train_time"], fix)

            plt.show()

        else:
            fix = False
            if ThreePoint_checker.get() and File['three_point_reward']:
                three_point_graph(File["episode"], File["three_point_reward"], fix)
            if TwoPoint_checker.get() and File['two_point_reward']:
                two_point_graph(File["episode"], File["two_point_reward"], fix)
            if break_through_checker.get() and File['break_through_reward']:
                break_through_graph(File["episode"], File["break_through_reward"], fix)
            if steal_checker.get() and File['steal_reward']:
                steal_graph(File["episode"], File["steal_reward"], fix)
            if block_checker.get() and File['block_reward']:
                block_graph(File["episode"], File["block_reward"], fix)
            if rebound_checker.get() and File['rebound_reward']:
                rebound_graph(File["episode"], File["rebound_reward"], fix)
            if chipout_checker.get() and File['chipout_reward']:
                chipout_graph(File["episode"], File["chipout_reward"], fix)
            if opponent_checker.get() and File['opponent_reward']:
                opponent_graph(File["episode"], File["opponent_reward"], fix)
            if stage_clear_checker.get() and File['stage_clear']:
                stage_clear_graph(File["episode"], File["stage_clear"], fix)
            if stage_clear_steps_checker.get() and File["stage_clear_steps"]:
                stage_clear_step_graph(File["episode"], File["stage_clear_steps"], fix)
            if loss_checker.get() and File['loss']:
                loss_graph(File["episode"], File["loss"], fix)
            if location_checker.get() and File['coordinate']:
                heatmap(File["episode"], File["coordinate"])
            if random_action_prob_checker.get() and File['random_action_prob']:
                random_action_graph(File["episode"], File["random_action_prob"], fix)
            if steps_checker.get() and File['steps']:
                step_graph(File["episode"], File["steps"], fix)
            if total_reward_checker.get() and File['total_reward']:
                total_reward_graph(File["episode"], File["total_reward"], fix)
            if progress_checker.get() and File["number_of_passes"]:
                progress_graph(File["episode"], File["number_of_passes"], fix)
            if duration_checker.get() and File['duration']:
                duration_graph(File['episode'], File['duration'], fix)
            if TrainTime_checker.get() and File['train_time']:
                train_time_graph(File["episode"], File["train_time"], fix)

            plt.show()


# 그래프 생성 버튼
f = Font(weight='bold', size=12)
bt_graph = Button(window, text="Generate Graphs", width=15, height=7, command=make_generator, bg='navajowhite')
bt_graph['font'] = f
bt_graph.place(x=486, y=30)

# if plt.show(block=False) and bt_graph.bind('<button-1>'):
#     plt.close('all')

# Saver = Frame(window)
# Saver.pack(side='top', fill='x')
# bt_save_graph = Button(Saver, text="Make Graph", width=30, height=1, command=make_graph, bg='navajowhite')
# bt_save_graph.pack(side='top', fill='x', padx=5)

window.mainloop()