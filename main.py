### main.py
import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
from gcn_model import get_model
from dataset import MyDataset
import matplotlib.pyplot as plt

# 定義訓練函式
def train(model, train_loader, valid_loader, device, n_epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    train_losses = []  # 用於記錄每個 epoch 的訓練損失
    valid_losses = []  # 用於記錄每個 epoch 的驗證損失

    for epoch in range(n_epoch):
        model.train()
        train_loss = []
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data = data.to(device)
            optimizer.zero_grad()

            xc_logis, xo_logis = model(data)
            
            # 確保目標是圖標籤
            if data.y.ndim > 1:
                data.y = data.y.argmax(dim=1)

            loss = criterion(xc_logis, data.y) + criterion(xo_logis, data.y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # 記錄平均訓練損失
        epoch_train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(epoch_train_loss)

        # 驗證模型
        valid_loss = validate(model, valid_loader, device, criterion)
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "best_model.pth")

    return train_losses, valid_losses


def validate(model, valid_loader, device, criterion):
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)

            xc_logis, xo_logis = model(data)
            
            # 確保目標是圖標籤
            if data.y.ndim > 1:
                data.y = data.y.argmax(dim=1)

            loss = criterion(xc_logis, data.y) + criterion(xo_logis, data.y)
            valid_loss.append(loss.item())
    return sum(valid_loss) / len(valid_loss)

# 測試函式
def test(model, test_loader, device):
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            xc_logis, _ = model(data)
            pred = xc_logis.argmax(dim=1)  # 預測類別索引
            
            # 確保目標是整數標籤
            if data.y.ndim > 1:
                data.y = data.y.argmax(dim=1)
            
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")


# 主程式
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_name", type=str, default="MyModel", help="Model name: MyModel, CausalGCN")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset_name

    # 加載資料集
    train_dataset = MyDataset(root=f"./data/{dataset_name}/train", Trainset=True)
    test_dataset = MyDataset(root=f"./data/{dataset_name}/test", Trainset=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    in_dim = 3  # 節點特徵維度
    out_dim = 8  # 類別數
    args.hidden = 64
    args.layers = 3
    model = get_model(args.model_name, in_dim, out_dim, args).to(device)

    # 訓練與記錄損失
    train_losses, valid_losses = train(model, train_loader, train_loader, device, args.epochs)

    # 繪製損失曲線
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), valid_losses, label="Valid Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png")  # 保存圖像
    plt.show()

    # 測試模型
    test(model, test_loader, device)
