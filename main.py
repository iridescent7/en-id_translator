import os
import argparse
import torch

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

def get_batch(source, i):
    sequence_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+sequence_len]
    target = source[i+1:i+1+sequence_len].reshape(-1)

    return data, target

def train_model(model):
    loss = nn.CrossEntropyLoss()
    learning_rate = 5.0
    optimizer = torch.optim.Adam(model.parameters(), learning_rate=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        1.0, 
        gamma=0.95
    )

    model.train()
    total_loss = 0
    log_interval = 200
    start_time = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_