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
    # 為了確保 GPU 記憶體空間足夠，將影像尺寸從 (100, 100) 縮放到 (32, 32)
    img = cv.resize(img, (32, 32))
    # 獲取影像的行列數
    row, col = img.shape[0], img.shape[1]

    # 將影像分解為 RGB 三個通道
    b, g, r = cv.split(img)

    # 計算需要的邊數 (2 * 節點數 - 行列的節點)
    num_edges = 2 * (2 * row * col - (row + col))

    # 初始化節點特徵矩陣 (x) 和邊索引矩陣 (edge_index)
    x = np.zeros((row * col, 3))
    edge_index = np.zeros((2, num_edges))
    # 初始化標籤 (y) 的張量，形狀根據類別數量進行設置
    y = np.zeros((1, num_classes))

    # 生成節點特徵，將像素值作為節點特徵
    v_index = 0
    for v_row in range(row):
        for v_col in range(col):
            # 正規化像素值到 [0, 1]
            x[v_index][0] = b[v_row][v_col] / 255
            x[v_index][1] = g[v_row][v_col] / 255
            x[v_index][2] = r[v_row][v_col] / 255
            v_index += 1

    # 生成邊的索引
    e_index = 0
    # 水平連接的邊
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
    # 垂直連接的邊
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

    # 根據類別索引設置標籤，對應類別標記為 1
    y[0][class_index] = 1

    # 將 numpy 陣列轉換為 PyTorch 張量
    x = torch.from_numpy(x).float()
    edge_index = torch.from_numpy(edge_index).long()
    y = torch.from_numpy(y).float()  # CrossEntropyLoss 要求標籤必須為 float 類型

    # 生成並返回一個 Data 物件
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


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
        # 繼承父類別的初始化方法
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
        data_list = []  # 用於存放處理後的數據
        class_mark = 0  # 當前類別的索引
        class_num = len(os.listdir(self.raw_dir))  # 總類別數量

        # 數據轉換：使用 SLIC 超像素和 RadiusGraph 生成圖結構
        transform = T.Compose(
            [
                T.ToTensor(),  # 轉換為張量
                ToSLIC(n_segments=300),  # 使用 SLIC 分割
                RadiusGraph(r=256, max_num_neighbors=16),  # 根據半徑生成邊
            ]
        )

        random.seed(8161)  # 設定隨機種子

        # 遍歷原始資料夾中的每個類別
        for filename in os.listdir(self.raw_dir):
            filelist = []  # 用於存放當前類別的所有影像路徑
            rawfilelist = os.walk(self.raw_dir + "\\" + filename)

            # 初始化當前類別的 Ground Truth 張量
            gt = torch.zeros((1, class_num))
            gt[0][class_mark] = 1  # 標記對應類別為 1

            # 將標籤封裝為 Data 物件
            patch_data = Data(y=gt)

            # 搜集影像檔案
            for folder, subfolder, file in rawfilelist:
                for imgfile in file:
                    filelist.append(os.path.join(folder, imgfile))

            # 將每張影像轉換為圖數據並加入 data_list
            for file in filelist:
                img = cv.imread(file)
                data = transform(img=img)  # 轉換影像為圖數據
                data.update(patch_data)  # 將 Ground Truth 添加到圖數據中
                data_list.append(data)

            print("class " + str(class_mark) + " complete")
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
