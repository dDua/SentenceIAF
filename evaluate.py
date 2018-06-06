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
"""Evaluation script."""

import argparse
import os
import logging
import shutil
import sys
import torch
import torch.nn.functional as F
import yaml
from math import exp, log
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data import TextDataset
from model import RNNTextInferenceNetwork, RNNTextGenerativeModel
from utils import Vocab, configure_logging, word_dropout


def suml2p(logp, target, pad_idx):
    """Computes the sum of log2 probs."""
    suml2p = 0.0
    n = 0
    for pred, idx in zip(logp, target):
        if idx == pad_idx:
            continue
        else:
            n += 1
            # Convert base of log
            lp = pred[idx]
            l2p = lp / log(2)
            suml2p += l2p
    return suml2p, n


def main(_):
    # Set up logging
    configure_logging(FLAGS.debug_log)

    # Load configuration
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    # Get the checkpoint path
    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])

    # Load vocab and datasets
    logging.info('Loading the vocabulary.')
    with open(config['data']['vocab'], 'r') as f:
        vocab = Vocab.load(f)
    logging.info('Loading test data.')
    test_data = TextDataset(config['data']['test'],
                             vocab=vocab,
                             max_length=config['training']['max_length'])
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available())

    # Initialize models
    logging.info('Initializing the inference network and generative model.')
    inference_network = RNNTextInferenceNetwork(
        dim=config['model']['dim'],
        vocab_size=len(vocab),
        encoder_kwargs=config['model']['encoder'],
        normalizing_flow_kwargs=config['model']['normalizing_flow'])
    generative_model = RNNTextGenerativeModel(
        dim=config['model']['dim'],
        vocab_size=len(vocab),
        max_length=config['training']['max_length'],
        sos_idx=vocab.sos_idx,
        **config['model']['generator'])
    if torch.cuda.is_available():
        inference_network = inference_network.cuda()
        generative_model = generative_model.cuda()

    # Restore
    ckpt = os.path.join(ckpt_dir, 'model.pt.best')
    if os.path.exists(ckpt):
        logging.info('Model checkpoint detected at: `%s`. Restoring.' % ckpt)
        checkpoint = torch.load(ckpt)
        inference_network.load_state_dict(checkpoint['state_dict_in'])
        generative_model.load_state_dict(checkpoint['state_dict_gm'])
    else:
        logging.error('No model checkpoint found. Terminating.')
        sys.exit(1)

    # Init test summaries
    test_nll = 0.0
    test_kl = 0.0
    test_loss = 0.0
    test_suml2p = 0.0
    test_n = 0.0

    # Evaluate
    inference_network.eval()
    generative_model.eval()

    for batch in test_loader:

        x = batch['input']
        target = batch['target']
        lengths = batch['lengths']
        if torch.cuda.is_available():
            x = x.cuda()
            target = target.cuda()
            lengths = lengths.cuda()

        # Forward pass of inference network
        z, kl = inference_network(x, lengths)

        # Teacher forcing
        logp, _ = generative_model(z, x, lengths)

        # Compute loss
        length = logp.shape[1]
        logp = logp.view(-1, len(vocab))
        target = target[:,:length].contiguous().view(-1)
        nll = F.nll_loss(logp, target, ignore_index=vocab.pad_idx,
                         size_average=False)
        loss = nll + kl
        l2p, n = suml2p(logp, target, vocab.pad_idx)

        # Update summaries
        test_nll += nll.data
        test_kl += kl.data
        test_loss += loss.data
        test_suml2p += l2p.data
        test_n += n

    # Normalize losses
    test_nll /= len(test_data)
    test_kl /= len(test_data)
    test_loss /= len(test_data)
    H = -test_suml2p / test_n
    test_perplexity = 2**H

    # Log output
    logging.info('NLL: %0.4f' % test_nll)
    logging.info('KL: %0.4f' % test_kl)
    logging.info('ELBO: %0.4f' % test_loss)
    logging.info('Perplexity: %0.4f' % test_perplexity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file.')
    parser.add_argument('--debug_log', type=str, default=None,
                        help='If given, write DEBUG level logging events to '
                             'this file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

