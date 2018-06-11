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
"""Generate sentences from model and inspect interpolants."""

import argparse
import logging
import os
import sys
import yaml

import torch

from model import RNNTextInferenceNetwork, RNNTextGenerativeModel
from utils import Vocab, configure_logging


class Beam(object):
    """Ordered beam of candidate outputs.
    Code borrowed from OpenNMT PyTorch implementation:
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
    """
    def __init__(self, vocab, width, generative_model):
        self.pad_idx = vocab.pad_idx
        self.sos_idx = vocab.sos_idx
        self.eos_idx = vocab.eos_idx
        self.width = width
        self.gen_model = generative_model
        self.max_length = generative_model.max_length

    def search(self, z):
        batch_size = z.shape[0]
        sos_x = self.sos_idx * torch.ones(batch_size, 1, dtype=torch.int64)
        sample = torch.zeros(batch_size, self.max_length)
        if torch.cuda.is_available():
            sos_x = sos_x.cuda()
            sample = sample.cuda()

        # Initial hidden state
        hidden = self.gen_model.latent2hidden(z)
        hidden.unsqueeze_(0)

        sample_list = list()
        x_list = list()
        hidden_list = list()
        sample_list.append(sample)
        x_list.append(sos_x)
        hidden_list.append(hidden)

        for i in range(self.max_length):
            tmp_sample_list = list()
            tmp_x_list = list()
            tmp_hidden_list = list()
            all_top_logit_idx_tuple_list = list()
            for j in range(len(x_list)):
                x = x_list[j]
                hidden = hidden_list[j]
                # Embed
                embeddings = self.gen_model.embedding(x)
                # Feed through RNN
                rnn_out, hidden = self.gen_model.decoder_rnn(embeddings, hidden)

                # Compute outputs
                logits = self.gen_model.hidden2logp(rnn_out).squeeze(1)

                # Sample from outputs
                top_k_logits, top_k_xs = logits.topk(self.width)
                for k in range(self.width):
                    sample = sample_list[j].detach()
                    sample[:, i] = x.detach()
                    tmp_sample_list.append(sample)
                    tmp_x_list.append(top_k_xs[:, k].unsqueeze(0).detach())
                    tmp_hidden_list.append(hidden.detach())
                    all_top_logit_idx_tuple_list.append((j, k, top_k_logits[0][k]))

            top_k_tuple_list = sorted(all_top_logit_idx_tuple_list, key=lambda tup: tup[2], reverse=True)
            sample_list = list()
            x_list = list()
            hidden_list = list()
            for j in range(self.width):
                tmp_tuple = top_k_tuple_list[j]
                target_idx = self.width * tmp_tuple[0] + tmp_tuple[1]
                sample_list.append(tmp_sample_list[target_idx])
                x_list.append(tmp_x_list[target_idx])
                hidden_list.append(tmp_hidden_list[target_idx])
        return sample_list[0]


def generate_example(inference_network, generative_model, vocab, beam_width=1):
        # Infer two greedy samples
        z_0 = torch.randn(1, inference_network.dim)
        if torch.cuda.is_available:
            z_0 = z_0.cuda()
        #  TODO: Figure out wtf to do w/ `h`...
        h = None
        z_k, _ = inference_network.normalizing_flow(z_0, h)
        if beam_width == 1:
            _, sample = generative_model(z_k)
        else:
            beam = Beam(vocab, beam_width, generative_model)
            sample = beam.search(z_k)

        example = [vocab.id2word(int(x)) for x in sample[0]]
        try:
            T = example.index(vocab.eos_token)
            example = example[:T]
        except ValueError:
            pass
        example = ' '.join(example)
        return example, z_k


def generate_interpolants(z_k_0, z_k_1, generative_model, vocab, steps=5):
    intermediate_examples = []
    for k in range(1, steps):
        alpha = k / steps
        z_k = alpha * z_k_0 + (1 - alpha) * z_k_1
        _, sample = generative_model(z_k)
        example = [vocab.id2word(int(x)) for x in sample[0]]
        try:
            T = example.index(vocab.eos_token)
            example = example[:T]
        except ValueError:
            pass
        example = ' '.join(example)
        intermediate_examples.append(example)
    return intermediate_examples


def main(_):
    # Set up logging
    configure_logging(FLAGS.debug_log)

    # Load configuration
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    # Get the checkpoint path
    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])

    # Load model vocab
    logging.info('Loading the vocabulary.')
    with open(config['data']['vocab'], 'r') as f:
        vocab = Vocab.load(f)

    # Initialize models
    logging.info('Initializing the generative model.')
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

    inference_network.eval()
    generative_model.eval()

    for _ in range(FLAGS.n_samples):
        logging.info('=== Example ===')
        example_0, z_k_0 = generate_example(inference_network,
                                            generative_model,
                                            vocab,
                                            config['model']['beam_width'])
        example_1, z_k_1 = generate_example(inference_network,
                                            generative_model,
                                            vocab,
                                            config['model']['beam_width'])
        intermediate_examples = generate_interpolants(z_k_0, z_k_1,
                                                      generative_model,
                                                      vocab)
        logging.info('Start: %s' % example_0)
        for example in intermediate_examples:
            logging.info(example)
        logging.info('End: %s' % example_1)

        # TODO: Formatted LaTeX table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file.')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples.')
    parser.add_argument('--debug_log', type=str, default=None,
                        help='If given, write DEBUG level logging events to '
                             'this file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

