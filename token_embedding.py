## Fungsi ini adalah untuk mengkonversi tensor input indices menjadi token embedding dalam bentuk tensor.
class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, embedding_size):
    super(TokenEmbedding, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.embedding_size = embedding_size

  def forward(self, tokens):
    return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)