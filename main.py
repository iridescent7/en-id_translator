import os
import argparse
import torch

from core.dataset import Dataset

def main(args):
    try:
        if args.download_data:
            print('Downloading datasets...')
            Dataset.download_all()
            print()

        if args.build_data:
            print('Building datasets...')
            Dataset.build_dataset()
            print()
            
    except RuntimeError as err:
        print('Error: {}'.format(err))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--download_data', action='store_true')
    parser.add_argument('--build_data', action='store_true')

    parser.add_argument('--mode', type=str,
                        choices=['train', 'test'])

    args = parser.parse_args()
    main(args)