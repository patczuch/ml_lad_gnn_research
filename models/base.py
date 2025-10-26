import torch
from torch_geometric.nn import global_mean_pool

from .gcn import GCN
from .gat import GAT
from .gin import GIN


class PureNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, gnn, nlayers=2, gat_heads=4, dropout=0.5, with_bn=True, with_bias=True):
        super().__init__()

        if gnn == "GCN":
            self.gnn_model = GCN(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
        elif gnn == "GAT":
            self.gnn_model = GAT(nfeat, nhid, nhid, nlayers, gat_heads, 1, dropout, with_bn)
        elif gnn == "GIN":
            self.gnn_model = GIN(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
        else:
            raise Exception("gnn mode error!")
        self.cls = torch.nn.Linear(nhid, nclass)

    def forward(self, x, edge_index, batch):
        h = self.gnn_model(x, edge_index)
        g = global_mean_pool(h, batch)
        out = torch.log_softmax(self.cls(g), dim=1)
        return out, g