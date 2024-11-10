import os

# 文件夹路径
folder_path = "/home/yitong/diffusion/data_train/battery_4"

# 获取文件列表
files = os.listdir(folder_path)

# 遍历文件列表并批量重命名
for filename in files:
    # 检查文件名是否符合指定格式
    if filename.startswith("episode_") and filename.endswith(".hdf5"):
        # 提取数字部分并加上 100
        number_str = filename[len("episode_"):-len(".hdf5")]
        
        # 检查数字部分是否为数字
        if number_str.isdigit():
            new_number = int(number_str) + 480
            new_filename = f"episode_{new_number}.hdf5"

            # 拼接完整路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f'Renamed "{filename}" to "{new_filename}"')

print("批量重命名完成！")
