import math
import torch
from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from typing import Iterable, List

## Model Sequence 2 Sequence dengan Transformer
class Seq2SeqTransformer(nn.Module):
  def __init__(self,
               num_encoder_layers,
               num_decode_layers,
               embedding_size,
               n_head,
               source_vocab_size,
               target_vocab_size,
               dimension_feedforward,
               dropout):
    super(Seq2SeqTransformer, self).__init__()
    self.transformer = Transformer(d_model=embedding_size,
                                   nhead=n_head,
                                   num_encoder_layers=num_encoder_layers,
                                   num_decoder_layers=num_decode_layers,
                                   dim_feedforward=dimension_feedforward,
                                   droput=dropout)
    # Mungkin layer ini bisa kita ganti2, sesuai dengan kebutuhan kita.
    self.generator = nn.Linear(embedding_size, target_vocab_size)
    self.source_token_embedding = TokenEmbedding(source_vocab_size, embedding_size)
    self.target_token_embedding = TokenEmbedding(target_vocab_size, embedding_size)
    self.positional_embedding = PositionalEncoding(embedding_size, dropout=dropout)
  
  def forward(self,
              source,
              target,
              source_mask,
              target_mask,
              source_padding_mask,
              target_padding_mask,
              memory_key_padding_mask):
    source_embedding = self.positional_encoding(self.source_token_embedding(source))
    target_embedding = self.positional_encoding(self.target_token_embedding(target))
    output = self.transformer(source_embedding, target_embedding, source_mask, target_mask,
                              None, source_padding_mask, target_padding_mask, memory_key_padding_mask)

  def encode(self, source, source_mask):
    return self.transformer.encoder(self.positional_encoding(self.positional_encoding(source)), source_mask)
  
  def decode(self, target, memory, target_mask):
    return self.transformer.decoder(self.positional_encoding(
        self.target_token_embedding(target)),
        memory,
        target_mask)
  