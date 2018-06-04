# Copyright 2018 Dua, Logan and Matsubara
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementations of the inference network and generative model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules import RNNTextEncoder, NormalizingFlow


class RNNTextInferenceNetwork(nn.Module):
    """Infers latent variable z from observations x."""
    def __init__(self,
                 dim,
                 vocab_size,
                 encoder_kwargs,
                 normalizing_flow_kwargs):
        super(RNNTextInferenceNetwork, self).__init__()

        self.dim = dim
        self.vocab_size = vocab_size

        self.encoder = RNNTextEncoder(dim, vocab_size, **encoder_kwargs)
        self.normalizing_flow = NormalizingFlow(dim, **normalizing_flow_kwargs)

    def forward(self, x, lengths):
        """Computes foward pass of the inference network.

        Args:
            x: torch.Tensor(batch_size, seq_len). Input data.
            lengths: torch.Tensor(batch_size). List of sequence lengths of each
                batch element.

        Returns:
            z: torch.Tensor(batch_size, dim). Latent variable sampled from
                p(z_k | x).
            KL: scalar. The KL divergence between q(z_k | x) and p(z_k).
        """
        # Compute z_k
        batch_size = x.shape[0]
        mean, logv = self.encoder(x, lengths)
        epsilon = torch.randn(batch_size, self.dim)
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        std = torch.exp(0.5 * logv)
        z_0 = mean + std * epsilon
        z_k, sum_logdet = self.normalizing_flow(z_0)

        # Compute KL divergence (discarding constants)
        # Note: Assuming p(z) ~ N(0,I) and q(z) ~ N(mean, sigma**2) (diagonal).
        log_p_zk = torch.sum(-0.5 * z_k**2, dim=1)
        log_q_z0 = torch.sum(-0.5 * (logv + (z_0 - mean)**2) / torch.exp(logv), dim=1)
        kl = log_q_z0 - sum_logdet - log_p_zk
        kl = torch.mean(kl)

        return z_k, kl


class RNNTextGenerativeModel(nn.Module):
    """Recurrent generative model for text data."""
    def __init__(self,
                 dim,
                 vocab_size,
                 hidden_size,
                 embedding_size,
                 dropout_rate,
                 sos_idx,
                 max_length):
        super(RNNTextGenerativeModel, self).__init__()

        self.dim = dim
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout = nn.Dropout(dropout_rate)
        self.latent2hidden = nn.Linear(dim, hidden_size)
        self.decoder_rnn = nn.GRU(embedding_size, hidden_size,
                                  num_layers=1,
                                  bidirectional=False,
                                  batch_first=True)
        self.hidden2logp = nn.Linear(hidden_size, vocab_size)

    def forward(self, z, x=None, lengths=None):
        """Computes a forward pass of the generative model.

        Args:
            z: torch.Tensor(batch_size, dim). The latent variable.
            x: torch.Tensor(batch_size, seq_length). Optional.
                If not given, the generator uses greedy sampling to generate the
                    output.
                If given, the generator uses teacher forcing.

        Returns:
            logp: torch.Tensor(batch_size, seq_length). Log-probabilities of
                the output.
        """

        # Greedy sampling
        if x is None:
            batch_size = z.shape[0]
            x = self.sos_idx * torch.ones(batch_size, 1, dtype=torch.int64)
            if torch.cuda.is_available():
                x = x.cuda()
            logp = []

            # Initial hidden state
            hidden = self.latent2hidden(z)
            hidden.unsqueeze_(0)

            logp = torch.zeros(batch_size, self.max_length, self.vocab_size)
            for i in range(self.max_length):
                # Embed
                embeddings = self.word_dropout(self.embedding(x))

                # Feed through RNN
                rnn_out, hidden = self.decoder_rnn(embeddings, hidden)

                # Compute outputs
                logits = self.hidden2logp(rnn_out)
                logp[:,i,:] = F.log_softmax(logits, dim=-1)

                # Sample greedily from outputs
                x = torch.argmax(logits, dim=1)

            # TODO: Check the shape of this...
            logp = torch.tensor(logp, dtype=torch.float32)

        # Teacher forcing
        else:
            # Stupid torch sequence sorting -_-
            sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
            _, unsorted_idx = torch.sort(sorted_idx)
            x = x[sorted_idx]
            z = z[sorted_idx]

            # Initial hidden state
            hidden = self.latent2hidden(z)
            hidden.unsqueeze_(0)

            # Embed
            embeddings = self.word_dropout(self.embedding(x))

            # Feed through RNN
            packed = pack_padded_sequence(embeddings, sorted_lengths, batch_first=True)
            rnn_out, _ = self.decoder_rnn(packed, hidden)
            unpacked, _ = pad_packed_sequence(rnn_out, batch_first=True)

            # Compute outputs
            logits = self.hidden2logp(unpacked)
            logp = F.log_softmax(logits, dim=-1)

            # Stupid torch sequence unsorting -_-
            logp = logp[unsorted_idx]

        return logp

