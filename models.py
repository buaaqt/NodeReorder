import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from layers import GraphConvolution, GraphSAGELayer
from abc import ABC
from utils import accuracy


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


class GraphSAGE(nn.Module, ABC):
    def __init__(self, n_class, batch_size, sage: GraphSAGELayer, dropout):
        super(GraphSAGE, self).__init__()
        self.sage = sage
        self.batch_size = batch_size
        self.n_class = n_class
        self.weight = Parameter(torch.FloatTensor(sage.out_features, n_class))
        self.dropout = dropout
        init.xavier_uniform(self.weight)
        self.x_ent = nn.CrossEntropyLoss()

    def forward(self, in_features, neigh_tab, _range):
        embeds = self.sage(in_features, neigh_tab, _range)
        embeds = F.dropout(embeds, self.dropout, training=self.training)
        res = torch.mm(embeds, self.weight)
        return res

    def loss(self, in_features, neigh_tab, _range, labels):
        _range = random.sample(_range, self.batch_size)
        labels = Variable(torch.LongTensor(labels[np.array(_range)]))
        res = F.log_softmax(self.forward(in_features, neigh_tab, _range), dim=1)
        acc_train = accuracy(res, labels)
        return self.x_ent(res, labels), acc_train

    def __repr__(self):
        return '\n' \
               + self.__class__.__name__

