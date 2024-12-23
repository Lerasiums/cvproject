import os
import shutil
import random

# 訓練資料的路徑
path = "C:\\Users\\user\\Desktop\\data\\blood_cell\\train\\raw"

# 遍歷每個類別資料夾
for folder in os.listdir(path):
    filelist = []
    l = os.walk(path + "\\" + folder)
    for subfolder, _, file in l:
        for sfile in file:
            filelist.append(os.path.join(subfolder, sfile))

    # 隨機選取 20% 的檔案
    num = int(0.2 * len(filelist))
    mfilelist = random.sample(filelist, k=num)

    # 目標測試資料夾路徑
    test_folder = "C:\\Users\\user\\Desktop\\data\\blood_cell\\test\\raw" + "\\" + folder

    # 如果目標資料夾不存在，則創建
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # 移動選取的檔案到目標資料夾
    for file in mfilelist:
        shutil.move(file, test_folder)
