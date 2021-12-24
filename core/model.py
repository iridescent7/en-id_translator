import torch
import torch.nn as nn
import torch.functional as F
import math

from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_length: int=1000):
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(0, max_length).reshape(max_length, 1)
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)

        pos_emb = torch.zeros((max_length, emb_size))
        pos_emb[:, 0::2] = torch.sin(pos * den)
        pos_emb[:, 1::2] = torch.cos(pos * den)
        pos_emb = pos_emb.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, token_emb):
        return self.dropout(token_emb + self.pos_emb[:token_emb.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.emb(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                encoder_layers: int,
                decode_layers: int,
                emb_size: int,
                nhead: int,
                source_vocab_size: int,
                target_vocab_size: int,
                dim_feedforward: int,
                dropout: float):
                
        super(Seq2SeqTransformer, self).__init__()

        self.pos_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.source_token_emb = TokenEmbedding(source_vocab_size, emb_size)
        self.target_token_emb = TokenEmbedding(target_vocab_size, emb_size)
        
        self.transformer = nn.Transformer(d_model=emb_size,
                                    nhead=nhead,
                                    num_encoder_layers=encoder_layers,
                                    num_decoder_layers=decode_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout)

        self.generator = nn.Linear(emb_size, target_vocab_size)
    
    def forward(self,
                source: Tensor,
                target: Tensor,
                source_mask: Tensor,
                target_mask: Tensor,
                source_padding_mask: Tensor,
                target_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):

        source_emb = self.pos_encoding(self.source_token_emb(source))
        target_emb = self.pos_encoding(self.target_token_emb(target))

        out_seq = self.transformer(source_emb, target_emb, source_mask, target_mask,
                                None, source_padding_mask, target_padding_mask, memory_key_padding_mask)

        return self.generator(out_seq)

    def encode(self, source: Tensor, source_mask: Tensor):
        return self.transformer.encoder(self.pos_encoding(self.source_token_emb(source)), source_mask)
    
    def decode(self, target: Tensor, memory: Tensor, target_mask: Tensor):
        return self.transformer.decoder(self.pos_encoding(self.target_token_emb(target)), memory, target_mask)
  