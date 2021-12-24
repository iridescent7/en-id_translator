
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer

from core.dataset import Dataset
from core.model import Seq2SeqTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 1

def collate_fn(batch):
    src_batch, tgt_batch = [], []

    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(src_sample))
        tgt_batch.append(torch.tensor(tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    
    return src_batch, tgt_batch

def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones((size, size), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_pad_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_pad_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_pad_mask, tgt_pad_mask

class SequenceDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class Solver():
    def __init__(self,
                 args,
                 src_lang: str='en',
                 tgt_lang: str='id',
                 train_batch_size: int=5000,
                 val_batch_size: int=5000,
                 num_workers: int=4):
        self.data = Dataset.load_dataset()

        if args.mode == 'train':
            def get_pair_set(data, type):
                return [(x, y) for x, y in zip(data[type][src_lang].token_ids, data[type][tgt_lang].token_ids)]

            self.train_dataloader = data.DataLoader(dataset=SequenceDataset(get_pair_set(self.data, 'train')),
                                    batch_size=train_batch_size,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,
                                    pin_memory=True)

            self.val_dataloader = data.DataLoader(dataset=SequenceDataset(get_pair_set(self.data, 'test')),
                                    batch_size=val_batch_size,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,
                                    pin_memory=True)

            self.seq2seq = Seq2SeqTransformer(3, 3, 512, 8,
                            len(self.data['train'][src_lang].vocab),
                            len(self.data['train'][tgt_lang].vocab),
                            512, 0.1)

            for p in self.seq2seq.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            self.seq2seq = self.seq2seq.to(DEVICE)
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
            self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

            self.losses = []

            NUM_EPOCHS = 18

            for epoch in range(1, NUM_EPOCHS+1):
                start_time = timer()
                train_loss = self._train_epoch()

                end_time = timer()
                val_loss = self._evaluate()

                print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    def _train_epoch(self):
        self.seq2seq.train()
        losses = 0

        for src, tgt in self.train_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input)

            logits = self.seq2seq(src, tgt_input, src_mask, tgt_mask,src_pad_mask, tgt_pad_mask, src_pad_mask)

            self.optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()
            losses += loss.item()

        return losses / len(self.train_dataloader)

    def _evaluate(self):
        self.seq2seq.eval()
        losses = 0

        for src, tgt in self.val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = self.seq2seq(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(self.val_dataloader)