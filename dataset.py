import numpy as np
import cv2 as cv
import random
import torchvision.transforms as T
from torch_geometric.transforms import ToSLIC, RadiusGraph

import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

# 將影像轉換為圖形數據的函式
def regular_grid_transform(img, class_index: int, num_classes: int):
    img = cv.resize(img, (32, 32))  # 確保圖片大小一致
    row, col = img.shape[0], img.shape[1]

    b, g, r = cv.split(img)  # 分解 RGB 通道
    num_edges = 2 * (2 * row * col - (row + col))  # 計算需要的邊數

    x = np.zeros((row * col, 3))  # 節點特徵矩陣
    edge_index = np.zeros((2, num_edges))  # 邊索引矩陣
    y = np.zeros((1, num_classes))  # 標籤

    # 生成節點特徵
    v_index = 0
    for v_row in range(row):
        for v_col in range(col):
            x[v_index][0] = b[v_row][v_col] / 255
            x[v_index][1] = g[v_row][v_col] / 255
            x[v_index][2] = r[v_row][v_col] / 255
            v_index += 1

    # 生成邊索引
    e_index = 0
    for v_row in range(row):
        for v_col in range(col - 1):
            v_s = v_row * col + v_col
            v_e = v_s + 1
            edge_index[0][e_index] = v_s
            edge_index[1][e_index] = v_e
            e_index += 1
            edge_index[1][e_index] = v_s
            edge_index[0][e_index] = v_e
            e_index += 1
    for v_row in range(row - 1):
        for v_col in range(col):
            v_s = v_row * col + v_col
            v_e = v_s + col
            edge_index[0][e_index] = v_s
            edge_index[1][e_index] = v_e
            e_index += 1
            edge_index[1][e_index] = v_s
            edge_index[0][e_index] = v_e
            e_index += 1

    # 標記類別
    y[0][class_index] = 1

    # 轉換為 PyTorch 張量
    x = torch.from_numpy(x).float()
    edge_index = torch.from_numpy(edge_index).long()
    y = torch.from_numpy(y).float()

    return Data(x=x, edge_index=edge_index, y=y)

# 定義自訂數據集類別
class MyDataset(InMemoryDataset):
    Trainset = True

    def __init__(
        self,
        root,
        Trainset,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.Trainset = Trainset
        super(MyDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ""

    @property
    def processed_file_names(self):
        return ["datas.pt"]

    def download(self):
        pass

    # 處理數據的主函式
    def process(self):
        data_list = []  # 存放處理後的數據
        class_mark = 0  # 當前類別索引
        class_num = len(os.listdir(self.raw_dir))  # 總類別數

        # 數據轉換
        transform = T.Compose([
            T.ToTensor(),
            ToSLIC(n_segments=300),
            RadiusGraph(r=256, max_num_neighbors=16),
        ])

        random.seed(8161)  # 設定隨機種子

        for filename in os.listdir(self.raw_dir):
            filelist = []  # 當前類別的所有圖片路徑
            rawfilelist = os.walk(os.path.join(self.raw_dir, filename))

            # 初始化 Ground Truth 張量
            gt = torch.zeros((1, class_num))
            gt[0][class_mark] = 1
            patch_data = Data(y=gt)

            # 收集圖片路徑
            for folder, subfolder, file in rawfilelist:
                for imgfile in file:
                    if imgfile.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filelist.append(os.path.join(folder, imgfile))

            # 將每張圖片轉換為圖數據
            for file in filelist:
                try:
                    img = cv.imread(file)
                    if img is None:
                        raise ValueError("Image is None")  # 如果圖片為空，跳過
                    data = transform(img=img)
                    data.update(patch_data)  # 添加 Ground Truth
                    data_list.append(data)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")  # 打印錯誤
                    continue

            print(f"class {class_mark} complete")
            class_mark += 1

        # 保存處理後的數據
        data, slices = self.collate(data_list=data_list)
        torch.save((data, slices), self.processed_paths[0])

# 打印數據集資訊的函式
def dataset_info(dataset):
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
