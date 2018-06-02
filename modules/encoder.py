"""Networks that can be used as encoders in a variational autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.rnn_utils import pack_padded_sequence


def RNNTextEncoder(nn.Module):
    """Recurrent neural network for encoding text.

    Args:
        dim: Dimension of the latent variable.
        vocab_size: Number of words in the vocabulary.
        embedding_size: Size of word embedding vectors.
        hidden_size: Number of hidden units in each recurrent layer.
        bidirectional: If `True` becomes a bidirectional RNN.
        dropout_rate: Probability of a word being droped out.
    """
    # TODO: Allow for multiple layers.
    def __init__(self,
                 dim,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 bidirectional=False,
                 dropout_rate=0.0):
        super(RNNTextEncoder, self).__init__()

        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = (2 if bidirectional else 1)
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout = nn.Dropout(dropout_rate)
        self.rnn_encoder = nn.GRU(embedding_size, hidden_size,
                                  num_layers=1,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        self.hidden2mean = nn.Linear(hidden_size * num_directions, dim)
        self.hidden2logv = nn.Linear(hidden_size * num_directions, dim)

    def forward(self, x, lengths):
        """Computes forward pass of the text encoder.

        Args:
            x: torch.Tensor(batch_size, seq_len). Input data.
            lengths: List of sequence lengths of each batch element.

        Returns:
            mu: torch.Tensor(batch_size, dim).
            sigma: torch.Tensor(batch_size, dim).
        """
        # Stupid torch sequence sorting -_-
        batch_size = x.shape[0]
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        _, unsorted_idx = torch.sort(sorted_idx)
        x = x[sorted_idx]

        # Embed
        embeddings = self.embedding(x)

        # Feed through RNN
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        _, hidden = self.rnn_encoder(packed)

        # Flatten hidden
        if self.num_directions > 1 or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.num_directions)
        else:
            hidden.squeeze_()

        # FC layer
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        # Stupid torch sequence unsorting -_-
        mean = mean[unsorted_idx]
        std = std[unsorted_idx]

        return mean, std

