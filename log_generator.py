import os
import time

now = time.localtime()
now_time = time.strftime('%Y%m%d_%H%M%S', now)


def make_log(episode=None, loss=None, coordinate=None, random_action_prob=None, steps=None, total_reward=None,
             number_of_passes=None, duration=None, three_point_reward=None, two_point_reward=None,
             break_through_reward=None, steal_reward=None, block_reward=None, rebound_reward=None, chipout_reward=None,
             opponent_reward=None, stage_clear=False, stage_clear_steps=None, train_time=None):
    line = "{" + "\"episode\":{}, \"loss\":{}, \"coordinate\":{}, \"random_action_prob\":{}, \"steps\":{}, "\
                 "\"total_reward\":{}, \"number_of_passes\":{}, \"duration\":{}, \"three_point_reward\":{}, " \
                 "\"two_point_reward\":{}, \"break_through_reward\":{}, \"steal_reward\":{}, \"block_reward\":{}, " \
                 "\"rebound_reward\":{}, \"chipout_reward\":{}, \"opponent_reward\":{}, \"stage_clear\":{}, " \
                 "\"stage_clear_steps\":{}, \"train_time\":{} "\
        .format(episode, loss, coordinate, random_action_prob, steps, total_reward,
                number_of_passes, duration, three_point_reward, two_point_reward,
                break_through_reward, steal_reward, block_reward, rebound_reward, chipout_reward, opponent_reward,
                stage_clear, stage_clear_steps, train_time) + "}\n"

    path = os.path.dirname(os.path.realpath(__file__)) + '/GraphLog_dir'
    if not os.path.isdir(path):
        os.mkdir(path)

    file = open(path + "/GraphLog{}_name.log".format(now_time), 'a', encoding='UTF-8')
    file.write(line)

    file.close()