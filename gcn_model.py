### gcn_model.py
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_add_pool, global_mean_pool, GATConv
from gcn_conv import GCNConv

# 原有的模型類 MyModel 保留
class MyModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MyModel, self).__init__()
        torch.manual_seed(8161)
        self.conv1 = GCNConv(in_dim, 64)
        self.bn1 = BatchNorm1d(64)
        self.relu1 = ReLU()
        self.pool1 = global_add_pool
        self.fc = Linear(64, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

# 匯入 CausalGCN 類
class CausalGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, args, edge_norm=True):
        super(CausalGCN, self).__init__()
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm)

        self.bn_feat = BatchNorm1d(num_features)
        self.conv_feat = GConv(num_features, hidden)
        self.convs = torch.nn.ModuleList([GConv(hidden, hidden) for _ in range(args.layers)])
        
        self.edge_att_mlp = Linear(hidden * 2, 2)
        self.node_att_mlp = Linear(hidden, 2)

        self.fc_context = Sequential(BatchNorm1d(hidden), Linear(hidden, num_classes))
        self.fc_objects = Sequential(BatchNorm1d(hidden), Linear(hidden, num_classes))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        row, col = edge_index

        x = F.relu(self.conv_feat(self.bn_feat(x), edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        node_att = F.softmax(self.node_att_mlp(x), dim=-1)

        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x

        xc = global_mean_pool(xc, batch)  # 使用全局池化
        xo = global_mean_pool(xo, batch)

        xc_logis = self.fc_context(xc)
        xo_logis = self.fc_objects(xo)

        return xc_logis, xo_logis

# 新增 get_model 方法
def get_model(model_name, in_dim, out_dim, args):
    if model_name == "MyModel":
        return MyModel(in_dim, out_dim)
    elif model_name == "CausalGCN":
        return CausalGCN(num_features=in_dim, num_classes=out_dim, args=args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")