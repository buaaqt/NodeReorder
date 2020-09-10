import argparse
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from utils import load_cora, accuracy
from models import GCN


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, neigh_tab)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, neigh_tab)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Select a graph dataset.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
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

    model = GCN(n_feat=features.shape[1],
                n_hid=args.hidden,
                n_class=labels.max().item() + 1,
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        neigh_tab = neigh_tab.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    test()
