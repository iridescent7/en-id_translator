import os
import argparse

from dataset import Dataset

def main(args):
    print(args)

    if args.mode == 'download':
        Dataset.download_all()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, required=True,
                        choices=['download'])

    args = parser.parse_args()
    main(args)

def train_model()