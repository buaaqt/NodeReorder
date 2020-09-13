import math
import random
import torch
import torch.nn.functional as F

from abc import ABC
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module, ABC):
    """Native GCN layer"""

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def setparams(self, w):
        self.weight = Parameter(torch.from_numpy(w).float())
        w_shape = self.weight.size()
        self.in_features = w_shape[0]
        self.out_features = w_shape[1]
        self.reset_parameters()

    def reset_parameters(self):
        std_dev = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std_dev, std_dev)
        if self.bias is not None:
            self.bias.data.uniform_(-std_dev, std_dev)

    def forward(self, _input, neigh_tab):
        support = torch.mm(_input, self.weight)
        aggr_res = torch.zeros(support.shape)
        for node in neigh_tab:
            neigh = list(neigh_tab[node])
            neigh.append(node)
            aggr_res[node] = torch.mean(support[neigh], 0)
        aggr_res = torch.FloatTensor(aggr_res)
        if self.bias is not None:
            return aggr_res + self.bias
        else:
            return aggr_res

    def __repr__(self):
        return '\n' \
               + self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')\n' \
               + 'weight:' + str(self.weight) + ',\n' \
               + 'bias:' + str(self.bias) + ',\n'


class GraphSAGELayer(Module, ABC):

    def __init__(self, in_features, out_features, bias=False, agg_func='MEAN'):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(2 * in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.agg_func = agg_func
        self.reset_parameters()

    def setparams(self, w):
        self.weight = Parameter(torch.from_numpy(w).float())
        w_shape = self.weight.size()
        self.in_features = w_shape[0]
        self.out_features = w_shape[1]
        self.reset_parameters()

    def reset_parameters(self):
        std_dev = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std_dev, std_dev)
        if self.bias is not None:
            self.bias.data.uniform_(-std_dev, std_dev)

    def forward(self, _input, neigh_tab, batch_nodes):
        aggregated = self.aggregate(_input, neigh_tab, batch_nodes)
        combined = torch.cat([_input[batch_nodes], aggregated], dim=1)
        return F.relu(torch.mm(combined, self.weight))

    def aggregate(self, in_features, neigh_tab, batch_nodes, sample_size=10):
        sub_neigh_tab = {i: neigh_tab[i] for i in batch_nodes}
        sampled_neighs = {i: set(random.sample(list(sub_neigh_tab[i]), sample_size)) if len(sub_neigh_tab[i]) >= sample_size
                          else sub_neigh_tab[i]
                          for i in sub_neigh_tab}
        aggr_res = torch.zeros((len(batch_nodes), in_features.shape[1]))
        if self.agg_func is 'MEAN':
            for i, node in enumerate(sampled_neighs):
                neigh = list(neigh_tab[node])
                aggr_res[i] = torch.mean(in_features[neigh], 0)
        else:
            raise RuntimeError('Undefined aggregate function!')
        return aggr_res

    def __repr__(self):
        return '\n' \
               + self.__class__.__name__
