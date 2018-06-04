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
"""Defines Vocab object for translating words to/from integer ids."""

from collections import Counter


class Vocab(object):
    """Vocabulary object used to map between words and ids.

    Args:
        counter: Counter storing vocab counts.
        pad_token: Token used to denote padding.
        unk_token: Token used to replace unknown values.
        sos_token: Token used to denote the start of sentences
    """
    def __init__(self,
                 counter,
                 pad_token='<pad>',
                 sos_token='<sos>',
                 eos_token='<eos>',
                 unk_token='<unk>'):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self._counter = counter
        self._id2word = [pad_token, sos_token, eos_token, unk_token]
        self._id2word += [x for x in sorted(counter, key=counter.get, reverse=True)
                          if x not in self._id2word]
        self._word2id = {word: i for i, word in enumerate(self._id2word)}

    def __len__(self):
        return len(self._id2word)

    @classmethod
    def load(cls, f):
        """Loads serialized vocabulary."""
        counter = Counter()
        for line in f:
            word, count = line.strip().split('\t')
            counter[word] = int(count)
        vocab = cls(counter)
        return vocab

    def write(self, f):
        """Serializes vocabulary."""
        for word, count in sorted(self._counter.items(), key=lambda x: -x[1]):
            line = '%s\t%i\n' % (word, count)
            f.write(line)

    def word2id(self, word):
        """Looks up the id of a word in vocabulary."""
        if word in self._word2id:
            return self._word2id[word]
        else:
            return self.unk_idx

    def id2word(self, id):
        """Looks up the word with a given id from the vocabulary."""
        return self._id2word[id]

    @property
    def pad_idx(self):
        return self.word2id(self.pad_token)

    @property
    def sos_idx(self):
        return self.word2id(self.sos_token)

    @property
    def eos_idx(self):
        return self.word2id(self.eos_token)

    @property
    def unk_idx(self):
        return self.word2id(self.unk_token)

