# Fungsi layer ini adalah menambahkan notion word order 

class PositionalEncoding():
  def __init(self, embedding_size, dropout, max_length):
    super(PositionalEncoding, self).__init__()

    # Bisa ditambahkan berbagai layer tambahan di atasnya, seperti dropout, dll.
    density = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
    position = torch.arrange(0, max_length).reshape(max_length, 1)
    position_embedding = torch.zeros((max_length, embedding_size))
    position_embedding[:, 0::2] = torch.sin(position * density)
    position_embedding[:, 1::2] = torch.cos(position * density)
    position_embedding = position_embedding.unsquezze(-2)

  def forward(self, token_embedding):
    return self.droput(token_embedding + self.pos_embedding:token_embedding.size(0), :])
