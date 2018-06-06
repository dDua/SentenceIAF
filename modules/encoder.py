"""Networks that can be used as encoders in a variational autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class RNNTextEncoder(nn.Module):
    """Recurrent neural network for encoding text.

    Args:
        dim: Dimension of the latent variable.
        vocab_size: Number of words in the vocabulary.
        embedding_size: Size of word embedding vectors.
        hidden_size: Number of hidden units in each recurrent layer.
        bidirectional: If `True` becomes a bidirectional RNN.
        h_dim: Dimension of the 'additional vector' h emitted by the encoder.
            If `None` then no h vector is emitted.
    """
    # TODO: Allow for multiple layers.
    def __init__(self,
                 dim,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 bidirectional=False,
                 h_dim=None):
        super(RNNTextEncoder, self).__init__()

        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = (2 if bidirectional else 1)
        self.h_dim = h_dim

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn_encoder = nn.GRU(embedding_size, hidden_size,
                                  num_layers=1,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        self.hidden2mean = nn.Linear(hidden_size * self.num_directions, dim)
        self.hidden2logv = nn.Linear(hidden_size * self.num_directions, dim)
        if self.h_dim is not None:
            self.hidden2h = nn.Linear(hidden_size * self.num_directions, h_dim)

    def forward(self, x, lengths):
        """Computes forward pass of the text encoder.

        Args:
            x: torch.Tensor(batch_size, seq_len). Input data.
            lengths: torch.Tensor(batch_size). List of sequence lengths of each
                batch element.

        Returns:
            mean: torch.Tensor(batch_size, dim). Mean used in reparameterization
                trick.
            std: torch.Tensor(batch_size, dim). Std. deviation used in
                reparameterization trick.
        """
        batch_size = x.shape[0]

        # Stupid torch sequence sorting -_-
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        _, unsorted_idx = torch.sort(sorted_idx)
        x = x[sorted_idx]

        # Embed
        embeddings = self.embedding(x)

        # Feed through RNN
        packed = pack_padded_sequence(embeddings, sorted_lengths, batch_first=True)
        _, hidden = self.rnn_encoder(packed)

        # Flatten hidden
        if self.num_directions > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.num_directions)
        else:
            hidden.squeeze_()

        # FC layer
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)

        # Stupid torch sequence unsorting -_-
        mean = mean[unsorted_idx]
        logv = logv[unsorted_idx]

        # Compute h vector
        if self.h_dim is not None:
            h = self.hidden2h(hidden)
            h = h[unsorted_idx]
        else:
            h = None

        return mean, logv, h

