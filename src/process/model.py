from http.client import UnimplementedFileMode
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

class Readout(nn.Module):
    def __init__(self, in_channels):
        super(Readout, self).__init__()
        self.in_channels = in_channels
        # 使用全局池化 + MLP
        self.fc1 = nn.Linear(in_channels * 2, 128)  # mean + max pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, h, batch):
        # 使用全局平均池化和最大池化
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        
        # 拼接两种池化结果
        h = torch.cat([h_mean, h_max], dim=1)
        
        # MLP
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        h = torch.sigmoid(self.fc3(h))
        
        return h.squeeze(1)  # 从 [batch_size, 1] 变为 [batch_size]    

class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, emb_size, max_nodes, device):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device) 
        self.emb_size = emb_size
        self.readout = Readout(gated_graph_conv_args['out_channels'])       

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.ggc(x, edge_index)
        x = self.readout(x, batch)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
