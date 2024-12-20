import os
import torch
from tqdm import tqdm
import math
import torch.nn as nn

from torch_geometric.transforms import ToSLIC
from torch_geometric.datasets import MNISTSuperpixels
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from dataset import MyDataset, dataset_info
from gcn_model import MyModel

from argparse import ArgumentParser

# 定義訓練函式
def train(model, train_loader, valid_loader, device, n_epoch):
    optimizer = torch.optim.Adam(model.parameters())  # 使用 Adam 優化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵損失函數

    # 初始化變數
    best_loss = float('inf')  # 最佳驗證損失
    stop_count = 0  # 停止計數器

    for epoch in range(n_epoch):  # 設定訓練迴圈
        model.train()  # 設置模型為訓練模式

        loss_record = []
        # tqdm 用於顯示訓練進度條
        train_bar = tqdm(train_loader, position=0, leave=True)

        for data in train_bar:  # 迭代所有訓練數據
            optimizer.zero_grad()  # 清除梯度
            data = data.to(device)  # 將數據移至指定設備
            pred = model(data)  # 模型前向傳播
            loss = criterion(pred, data.y)  # 計算損失
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新權重

            loss_record.append(loss.detach().item())  # 記錄每次損失

            # 更新進度條描述
            train_bar.set_description(f"Epoch [{epoch+1} / {n_epoch}]")
            train_bar.set_postfix({"loss": loss.detach().item()})

        # 計算該輪次的平均訓練損失
        mean_train_loss = sum(loss_record) / len(loss_record)
        print(f"Mean train loss in Epoch {epoch+1} : {mean_train_loss:.4f}")

        # 模型驗證
        model.eval()  # 設置模型為驗證模式
        loss_record = []
        for data in valid_loader:
            data = data.to(device)
            with torch.no_grad():  # 不進行梯度計算
                pred = model(data)
                loss = criterion(pred, data.y)

            loss_record.append(loss.item())

        # 計算平均驗證損失
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f"Mean valid loss in Epoch {epoch+1} : {mean_valid_loss:.4f}")

        # 保存最佳模型
        model_dir = "C:\\Users\\user\\Desktop\\cvproject\\model"
        os.makedirs(model_dir, exist_ok=True)  # 確保目錄存在
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            print("Saving best model...")
            torch.save(model.state_dict(), os.path.join(model_dir, "model.ckpt"))
        else:
            stop_count += 1

        # 若模型多輪未改進，提早終止訓練
        if stop_count >= 20:
            print("Model is not improving. Training session is over.")
            break

# 定義測試函式
def test(model, test_loader, device):
    # 載入訓練過程中保存的最佳模型
    model_dir = "C:\\Users\\user\\Desktop\\cvproject\\model"
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.ckpt")))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_bar = tqdm(test_loader, position=0, leave=True)

    for data in test_bar:  # 迭代測試數據
        data = data.to(device)
        out = model(data)  # 模型前向傳播
        loss = criterion(out, data.y)  # 計算損失
        pred = out.argmax(dim=1)  # 預測結果

        test_bar.set_description("Test progress ")
        test_bar.set_postfix({"loss": loss.detach().item()})

        # 計算正確率
        correct += (pred == data.y.argmax(dim=1)).sum().item()
        total += data.y.size(0)

    print(f"Test accuracy : {correct/total:.4f}")

# 根據資料集名稱獲取路徑
def get_dataset_path(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        dataset_name = "mnist"
    elif dataset_name == "cifar":
        dataset_name = "CIFAR-10"
    elif dataset_name == "animal":
        dataset_name = "animal"
    elif dataset_name == "fruit":
        dataset_name = "fruit"
    elif dataset_name == "blood_cell":
        dataset_name = "blood_cell"

    # 返回訓練和測試數據的路徑
    train_path = "C:\\Users\\user\\Desktop\\data\\" + dataset_name + "\\train"
    test_path = "C:\\Users\\user\\Desktop\\data\\" + dataset_name + "\\test"

    return train_path, test_path

# 設定設備 (GPU 或 CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 配置不同資料集的輸入與輸出維度
in_dim_config = {"mnist": 3, "cifar": 3, "animal": 3, "fruit": 3, "blood_cell": 3}
out_dim_config = {"mnist": 10, "cifar": 10, "animal": 9, "fruit": 9, "blood_cell": 4}

# 解析命令列參數
parser = ArgumentParser()
parser.add_argument("dataset_name", type=str)  # 資料集名稱
args = parser.parse_args()
dataset_name = args.dataset_name

# 獲取資料集路徑
train_path, test_path = get_dataset_path(dataset_name)

# 初始化自定義資料集
Train_Dataset = MyDataset(root=train_path, Trainset=True)

# 將訓練資料隨機分割為訓練集與驗證集 (3:1)
valid_set_size = int(len(Train_Dataset) * 0.25)
train_set_size = len(Train_Dataset) - valid_set_size
Train_Dataset, Valid_Dataset = random_split(
    Train_Dataset,
    [train_set_size, valid_set_size],
    generator=torch.Generator().manual_seed(8161),
)

# 測試資料集
Test_Dataset = MyDataset(root=test_path, Trainset=False)

# 創建 DataLoader
Train_Loader = DataLoader(Train_Dataset, batch_size=32, shuffle=True)
Valid_Loader = DataLoader(Valid_Dataset, batch_size=32, shuffle=False)
Test_Loader = DataLoader(Test_Dataset, batch_size=1, shuffle=False)

# 初始化模型
model = MyModel(in_dim=in_dim_config[dataset_name], out_dim=out_dim_config[dataset_name])
model = model.to(device)  # 將模型移至設備
n_epoch = 100  # 訓練輪次

# 開始訓練
train(model=model, train_loader=Train_Loader, valid_loader=Valid_Loader, device=device, n_epoch=n_epoch)

# 開始測試
test(model=model, test_loader=Test_Loader, device=device)
