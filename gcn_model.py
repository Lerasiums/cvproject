# 這個模型基於 JK-Net（Jumping Knowledge Network）的設計，靈感來自以下論文：
# Towards Sparse Hierarchical Graph Classifiers
# 作者：Catalina Cangea, Petar Velickovic, Nikola Jovanovic, Thomas Kipf, Pietro Liò
# 論文連結：https://arxiv.org/abs/1811.01287

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, BatchNorm, TopKPooling, MLP, global_mean_pool, global_max_pool
from torch.nn import ReLU

# 定義自訂的圖神經網路模型類別
class MyModel(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(MyModel, self).__init__()
        torch.manual_seed(8161)  # 設定隨機種子以確保結果可重現
        
        # 第一層 GCN 卷積 + 批次正規化 + 激活函數 + TopK Pooling
        self.conv1 = GCNConv(in_dim, 64)  # 圖卷積層，輸入維度為 in_dim，輸出維度為 64
        self.bn1 = BatchNorm(64)  # 批次正規化層，規範化 64 維的輸出
        self.relu1 = ReLU()  # 激活函數
        self.pool1 = TopKPooling(64, ratio=0.8)  # TopK Pooling，保留 80% 的節點
        
        # 第二層 GCN 卷積 + 批次正規化 + 激活函數 + TopK Pooling
        self.conv2 = GCNConv(64, 64)  # 圖卷積層，輸入和輸出維度均為 64
        self.bn2 = BatchNorm(64)
        self.relu2 = ReLU()
        self.pool2 = TopKPooling(64, ratio=0.8)

        # 第三層 GCN 卷積 + 批次正規化 + 激活函數 + TopK Pooling
        self.conv3 = GCNConv(64, 64)
        self.bn3 = BatchNorm(64)
        self.relu3 = ReLU()
        self.pool3 = TopKPooling(64, ratio=0.8)

        # 多層感知機（MLP）作為最後的全連接層
        self.mlp = MLP([(64 * 2), out_dim])  # 輸入為 64 * 2（全局池化的輸出），輸出為類別數

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # 解包數據，包括節點特徵、邊索引、批次索引

        # 第一層卷積 + Pooling
        x = self.conv1(x, edge_index)  # 圖卷積運算
        x = self.bn1(x)  # 批次正規化
        x = self.relu1(x)  # 激活函數
        x, edge_index, _, batch, _, score1 = self.pool1(x, edge_index, None, batch, None)  # TopK Pooling

        # 進行全局池化（mean 和 max），並將結果串接
        readout1 = torch.cat(
            [global_mean_pool(x=x, batch=batch),  # 全局平均池化
             global_max_pool(x=x, batch=batch)],  # 全局最大池化
            dim=1,  # 沿特徵維度串接
        )

        # 第二層卷積 + Pooling
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu2(x)
        x, edge_index, _, batch, _, score2 = self.pool2(x, edge_index, None, batch, None)

        readout2 = torch.cat(
            [global_mean_pool(x=x, batch=batch),
             global_max_pool(x=x, batch=batch)],
            dim=1,
        )

        # 第三層卷積 + Pooling
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu3(x)
        x, edge_index, _, batch, _, score3 = self.pool3(x, edge_index, None, batch, None)

        readout3 = torch.cat(
            [global_mean_pool(x=x, batch=batch),
             global_max_pool(x=x, batch=batch)],
            dim=1,
        )

        # 將三層的讀取結果加總
        readout = readout1 + readout2 + readout3

        # 通過 MLP 進行最終分類
        readout = self.mlp(readout)

        return readout
