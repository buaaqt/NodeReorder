import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    temp_array = list(classes)
    temp_array.sort()
    classes_dict = {c: np.identity(len(temp_array))[i, :] for i, c in
                    enumerate(temp_array)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_cora(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    neigh_tab = gen_neigh_tab(edges)
    node_cluster = node_clustering(neigh_tab, 0.05)
    node_order = np.array(reorder(node_cluster))
    order_map = {j: i for i, j in enumerate(node_order)}
    edges = np.array(list(map(order_map.get, edges.flatten())),
                     dtype=np.int32).reshape(edges.shape)

    neigh_tab = gen_neigh_tab(edges)
    idx_features_labels = idx_features_labels[node_order]
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return neigh_tab, features, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def gen_neigh_tab(edges: np.ndarray):
    [u, v] = np.split(edges.flatten('F'), 2)
    edges_pd = pd.DataFrame({
        'src': u,
        'dst': v
    }, dtype=int)

    src_group = edges_pd.groupby('src').apply(lambda x: v[x['dst']]).to_dict()
    dst_group = edges_pd.groupby('dst').apply(lambda x: u[x['src']]).to_dict()
    res = {}
    keys = set(dst_group.keys()) | set(src_group.keys())
    for key in keys:
        if key not in src_group:
            res[key] = set(dst_group[key].tolist())
        elif key in dst_group:
            res[key] = set(src_group[key].tolist()) | set(dst_group[key].tolist())
        else:
            res[key] = set(src_group[key].tolist())
    return res


def gen_similar_tab(neigh_tab: dict):
    nodes_num = len(neigh_tab)
    temp_tab = []
    for u in range(nodes_num):
        for v in range(u + 1, nodes_num):
            s = (len(neigh_tab[u] & neigh_tab[v]) + 0.0) / len(neigh_tab[u] | neigh_tab[v])
            if s == 0:
                continue
            temp_tab.append([u, v, s])

    res_tab = np.array(temp_tab)
    res_tab = res_tab[np.lexsort(-res_tab.T)]
    _i = (res_tab[:, 0]).astype(int)
    _j = (res_tab[:, 1]).astype(int)
    _val = res_tab[:, 2]

    similar_tab = sp.coo_matrix((_val, (_i, _j)), shape=(nodes_num, nodes_num))
    return similar_tab


class NodeCluster:
    def __init__(self, v_center, v_center_set):
        self.v_center = v_center
        self.v = [{v_center: v_center_set}]

    def node_append(self, u, u_set):
        u_deg = len(u_set)
        if u_deg > len(self.v[0][self.v_center]):
            self.v.insert(0, {u: u_set})
            self.v_center = u
        else:
            self.v.append({u: u_set})
        return

    def get_distance(self, u_set):
        return (len(self.v[0][self.v_center] & u_set) + 0.0) / \
                len(self.v[0][self.v_center] | u_set)


def node_clustering(neigh_tab: dict, s0):
    nodes_num = len(neigh_tab)
    c = []
    for v in neigh_tab.keys():
        if len(c) == 0:
            c.append(NodeCluster(v, neigh_tab[v]))
        else:
            dist = list(map(lambda c_k: c_k.get_distance(neigh_tab[v]), c))
            max_dist_idx = max(range(len(dist)), key=dist.__getitem__)
            if dist[max_dist_idx] < s0:
                c.append(NodeCluster(v, neigh_tab[v]))
            else:
                c[max_dist_idx].node_append(v, neigh_tab[v])
    c.sort(key=lambda x: len(x.v[0][x.v_center]))
    return c


def reorder(c: list):
    res = []
    for item in c:
        nodes = list(map(lambda x: [*(x.keys())][0], item.v))
        res.extend(nodes)
    return res
