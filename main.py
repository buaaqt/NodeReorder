import argparse
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from utils import load_cora, accuracy
from layers import GraphSAGELayer
from models import GCN, GraphSAGE


def train(model_type, epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if model_type is 'gcn':
        output = model(features, neigh_tab)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, neigh_tab)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    else:
        _range = idx_train.tolist()
        loss_train, acc_train = model.loss(features, neigh_tab, _range, labels)
        loss_train.backward()
        optimizer.step()

        _range = idx_val.tolist()
        loss_val, acc_val = model.loss(features, neigh_tab, _range, labels)

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model_type):
    model.eval()
    if model_type is 'gcn':
        output = model(features, neigh_tab)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

    else:
        _range = idx_test.tolist()
        loss_test, acc_test = model.loss(features, neigh_tab, _range, labels)

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sage',
                        help='Select a graph neural network model.')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Select a graph dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        print('Using Cuda with', torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(args.seed)

    # Load data
    neigh_tab, features, labels, idx_train, idx_val, idx_test = load_cora()

    if args.model is 'gcn':
        model = GCN(n_feat=features.shape[1],
                    n_hid=args.hidden,
                    n_class=labels.max().item() + 1,
                    dropout=args.dropout)

    elif args.model is 'sage':

        sage = GraphSAGELayer(in_features=features.shape[1],
                              out_features=args.hidden)
        model = GraphSAGE(n_class=labels.max().item() + 1,
                          batch_size=128,
                          sage=sage,
                          dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(args.model, epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    test(args.model)
