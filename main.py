import os
import argparse
import torch

from core.dataset import Dataset

def main(args):
    try:
        if args.download_data:
            print('Downloading dataset...')
            Dataset.download_all()
            print()

        # TODO: kita main nya di ipynb aja kli ya
        # di ipynb juga bisa import .py
        if args.build_data:
            print('Building dataset...')
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