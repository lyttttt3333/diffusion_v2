import numpy as np


def extract_base_func_init(info):
    reward_index = list(range(len(info)))
    return reward_index

def green_extract(info):
    tgt_name = "battery_3"
    reward_index = []
    for idx, item in enumerate(info):
        name = item["name"]
        if name == tgt_name:
            reward_index.append(idx)
    if len(reward_index) != 0:
        return reward_index
    else:
        return None

def purple_extract(info):
    tgt_name = "battery_4"
    reward_index = []
    for idx, item in enumerate(info):
        name = item["name"]
        if name == tgt_name:
            reward_index.append(idx)
    if len(reward_index) != 0:
        return reward_index
    else:
        return None

def blue_extract(info):
    tgt_name = "battery_5"
    reward_index = []
    for idx, item in enumerate(info):
        name = item["name"]
        if name == tgt_name:
            reward_index.append(idx)
    if len(reward_index) != 0:
        return reward_index
    else:
        return None

def front_extract(info):
    reward_index = None
    max_y = -1e3
    for idx, item in enumerate(info):
        y = item["position"][1]
        if y > max_y:
            max_y = y
            reward_index = idx
    return [reward_index]

def left_extract(info):
    reward_index = None
    min_x = 1e3
    for idx, item in enumerate(info):
        x = item["position"][0]
        if x < min_x:
            min_x = x
            reward_index = idx
    return [reward_index]

def right_extract(info):
    reward_index = None
    max_x = -1e3
    for idx, item in enumerate(info):
        x = item["position"][0]
        if x > max_x:
            max_x = x
            reward_index = idx
    return [reward_index]

def back_extract(info):
    reward_index = None
    min_y = 1e3
    for idx, item in enumerate(info):
        y = item["position"][1]
        if y < min_y:
            min_y = y
            reward_index = idx
    return [reward_index] 







def extract_base_func_tgt(info):
    reward_init_index = info["wait_reward_index"]
    reward_index = info["free_list"]
    return [[reward_init_index, reward_index]]

def furthest_extract(info):
    tgt_positions = info["tgt_positions"]
    free_list = info["free_list"]
    free_position = tgt_positions[free_list]
    reward_init_index = info["wait_reward_index"]
    result_list = []
    for idx in reward_init_index:
        init_position = info["obj_wait_list"][idx]["position"]
        dist = np.linalg.norm(free_position - init_position, axis = -1)
        index = np.argmax(dist)
        tgt_index = free_list[index]
        result_list.append([[idx],[tgt_index]])
    return result_list

def nearest_extract(info):
    tgt_positions = info["tgt_positions"]
    free_list = info["free_list"]
    free_position = tgt_positions[free_list]
    reward_init_index = info["wait_reward_index"]
    result_list = []
    for idx in reward_init_index:
        init_position = info["obj_wait_list"][idx]["position"]
        dist = np.linalg.norm(free_position - init_position, axis = -1)
        index = np.argmin(dist)
        tgt_index = free_list[index]
        result_list.append([[idx],[tgt_index]])
    return result_list


def front_row_extract(info):
    free_list = info["free_list"]
    tgt_list = [0,1,2,3]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def middle_row_extract(info):
    free_list = info["free_list"]
    tgt_list = [4,5,6,7]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def back_row_extract(info):
    free_list = info["free_list"]
    tgt_list = [8,9,10,11]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def left_column_extract(info):
    free_list = info["free_list"]
    tgt_list = [3, 7, 11]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def middle_column_extract(info):
    free_list = info["free_list"]
    tgt_list = [1,2,5,6,9,10]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def right_column_extract(info):
    free_list = info["free_list"]
    tgt_list = [0, 4, 8]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def left_front_extract(info):
    free_list = info["free_list"]
    tgt_list = [3]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def right_front_extract(info):
    free_list = info["free_list"]
    tgt_list = [0]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def left_back_extract(info):
    free_list = info["free_list"]
    tgt_list = [11]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]

def right_back_extract(info):
    free_list = info["free_list"]
    tgt_list = [8]
    reward_index = list(set(free_list).intersection(tgt_list))
    reward_init_index = info["wait_reward_index"]
    return [[reward_init_index, reward_index]]