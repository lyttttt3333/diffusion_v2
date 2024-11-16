
# init check

def check_base_func_init(info):
    return True

def green_check(info):
    name_list = []
    for idx, item in enumerate(info):
        name_list.append(item["name"])
    if "battery_3" in name_list:
        return True
    else:
        return False 

def blue_check(info):
    name_list = []
    for idx, item in enumerate(info):
        name_list.append(item["name"])
    if "battery_4" in name_list:
        return True
    else:
        return False 

def purple_check(info):
    name_list = []
    for idx, item in enumerate(info):
        name_list.append(item["name"])
    if "battery_5" in name_list:
        return True
    else:
        return False 

# tgt check

def check_base_func_tgt(info):
    if len(info["free_list"]) != 0:
        return True
    else:
        return False

def front_row_check(info):
    for i in [0,1,2,3]:
        if i in info["free_list"]:
            return True
    return False

def middle_row_check(info):
    first_row = False
    middle_row = False
    last_row = False
    for i in [0,1,2,3]:
        if i in info["free_list"]:
            first_row = True
            break
    for i in [4,5,6,7]:
        if i in info["free_list"]:
            middle_row = True
            break
    for i in [8,9,10,11]:
        if i in info["free_list"]:
            last_row = True
            break
    if first_row and middle_row and last_row:
        return True
    else:
        return False

def back_row_check(info):
    for i in [8,9,10,11]:
        if i in info["free_list"]:
            return True
    return False

def left_column_check(info):
    for i in [3,7,11]:
        if i in info["free_list"]:
            return True
    return False

def middle_column_check(info):
    first_row = False
    middle_row = False
    last_row = False
    for i in [0,4,8]:
        if i in info["free_list"]:
            first_row = True
            break
    for i in [1,2,5,6,9,10]:
        if i in info["free_list"]:
            middle_row = True
            break
    for i in [3,7,11]:
        if i in info["free_list"]:
            last_row = True
            break
    if first_row and middle_row and last_row:
        return True
    else:
        return False


def right_column_check(info):
    for i in [0, 4 ,8]:
        if i in info["free_list"]:
            return True
    return False

def left_front_check(info):
    for i in [3]:
        if i in info["free_list"]:
            return True
    return False

def right_front_check(info):
    for i in [0]:
        if i in info["free_list"]:
            return True
    return False

def left_back_check(info):
    for i in [11]:
        if i in info["free_list"]:
            return True
    return False

def right_back_check(info):
    for i in [7]:
        if i in info["free_list"]:
            return True
    return False