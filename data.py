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
"""Simple object for working with text data."""

import os
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """A Dataset object for text files.

    NOTE: Assumes entire dataset can be stored in memory.

    Args:
        fname: File containing the text data.
        vocab: Vocab object for mapping words to ids.
        max_length: Maximum length of sentences.
    """
    def __init__(self, fname, vocab, max_length=50):
        super(TextDataset, self).__init__()
        self.fname = fname
        self.vocab = vocab
        self.max_length = max_length
        with open(fname, 'r') as f:
            self._load(f)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def _load(self, f):
        self._data = []
        for line in f:
            tokens = [self.vocab.sos_token, *line.split(), self.vocab.eos_token]
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            ids = [self.vocab.word2id(x) for x in tokens]
            length = len(ids)
            # Pad - NOTE: +1 for offset.
            ids += [self.vocab.pad_idx] * (1 + self.max_length - length)
            ids = torch.LongTensor(ids)
            self._data.append({'input': ids[:-1],
                               'target': ids[1:],
                               'lengths': length})

