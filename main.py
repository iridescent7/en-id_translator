import os
import argparse
import torch

from core.dataset import Dataset
from core.solver import Solver

def main(args):
    try:
        if args.download_data:
            Dataset.download_all()

        if args.build_dataset:
            Dataset.build_dataset(test_size=0.2)
            
        if args.mode == 'train':
            solver = Solver(args)

    except RuntimeError as err:
        print('Error: {}'.format(err))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--download_data', action='store_true')
    parser.add_argument('--build_dataset', action='store_true')

    parser.add_argument('--mode', type=str,
                        choices=['train', 'test'])

    args = parser.parse_args()
    main(args)