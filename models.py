import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution
from abc import ABC


class GCN(nn.Module, ABC):
    def __init__(self, n_feat, n_hid, n_class, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, neigh_tab):
        x = F.relu(self.gc1(x, neigh_tab))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, neigh_tab)
        return F.log_softmax(x, dim=1)
