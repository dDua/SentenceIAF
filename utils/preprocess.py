#! /usr/bin/env python3
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
"""Helper script for extracting a vocabulary from the PTB dataset.

Usage:
    ./preprocess.py [PATH TO TRAIN DATA] [PATH TO SAVE VOCAB TO]
"""
import argparse
from collections import Counter

from utils import Vocab


FLAGS = None


def main(_):
    counter = Counter()
    with open(FLAGS.input, 'r') as f:
        for line in f:
            counter.update(line.split())
    vocab = Vocab(counter)
    with open(FLAGS.output, 'w') as f:
        vocab.write(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input text file.')
    parser.add_argument('output', type=str, help='Path to save vocab to.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

