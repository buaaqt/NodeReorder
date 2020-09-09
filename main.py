import argparse

from utils import load_cora

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Select a graph dataset.')
    args = parser.parse_args()

    # Load data
    features, labels, idx_train, idx_val, idx_test = load_cora()