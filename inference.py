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
import torch.nn.functional as F

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
        init_log_softmax = torch.zeros(1)
        if torch.cuda.is_available():
            sos_x = sos_x.cuda()
            sample = sample.cuda()
            init_log_softmax = init_log_softmax.cuda()

        # Initial hidden state
        hidden = self.gen_model.latent2hidden(z)
        hidden.unsqueeze_(0)

        log_softmax_list = list()
        sample_list = list()
        x_list = list()
        hidden_list = list()
        log_softmax_list.append(init_log_softmax)
        sample_list.append(sample)
        x_list.append(sos_x)
        hidden_list.append(hidden)

        for i in range(self.max_length):
            tmp_sample_list = list()
            tmp_x_list = list()
            tmp_hidden_list = list()
            all_top_logit_tuple_list = list()

            for j in range(len(x_list)):
                x = x_list[j]
                input_hidden = hidden_list[j]
                # Embed
                embeddings = self.gen_model.embedding(x)
                # Feed through RNN
                rnn_out, output_hidden = self.gen_model.decoder_rnn(embeddings, input_hidden)

                # Compute outputs
                logits = self.gen_model.hidden2logp(rnn_out).squeeze(1)
                log_softmax = F.log_softmax(logits, dim=-1)

                # Sample from outputs
                top_k_log_softmax, top_k_xs = log_softmax.topk(self.width)
                top_k_log_softmax.detach()
                top_k_xs.detach()

                for k in range(self.width):
                    score = log_softmax_list[j] + top_k_log_softmax[:, k]
                    sample = sample_list[j].clone()
                    top_x = top_k_xs[:, k].unsqueeze(0)
                    sample[:, i] = top_x
                    tmp_sample_list.append(sample)
                    tmp_x_list.append(top_x)
                    tmp_hidden_list.append(output_hidden.detach())
                    all_top_logit_tuple_list.append((j, k, score))

            sorted_top_logit_tuple_list = sorted(all_top_logit_tuple_list, key=lambda tup: tup[2], reverse=True)
            log_softmax_list = list()
            sample_list = list()
            x_list = list()
            hidden_list = list()

            for j in range(self.width):
                tmp_tuple = sorted_top_logit_tuple_list[j]
                target_idx = self.width * tmp_tuple[0] + tmp_tuple[1]
                log_softmax_list.append(tmp_tuple[2])
                sample_list.append(tmp_sample_list[target_idx])
                x_list.append(tmp_x_list[target_idx])
                hidden_list.append(tmp_hidden_list[target_idx])
        return sample_list[0]


def generate_example(inference_network,
                     generative_model,
                     vocab,
                     beam_width):
        # Infer two greedy samples
        z_0 = torch.randn(1, inference_network.dim)
        if torch.cuda.is_available:
            z_0 = z_0.cuda()
        #  TODO: Figure out wtf to do w/ `h`...
        z_k, _ = inference_network.normalizing_flow(z_0, None)
        # Debug option to use beam_width < 1, for comparing greedy one to beam_width = 1
        if beam_width < 1:
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
        if FLAGS.early_interp:
            return example, z_0
        else:
            return example, z_k


def generate_interpolants(z_0,
                          z_1,
                          h,
                          inference_network,
                          generative_model,
                          vocab,
                          beam_width,
                          steps=5):
    intermediate_examples = []
    for k in range(1, steps):
        alpha = k / steps
        z = (1 - alpha) * z_0 + alpha * z_1
        if FLAGS.early_interp:
            z_k, _ = inference_network.normalizing_flow(z, h)
        else:
            z_k = z
        if beam_width < 1:
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
        intermediate_examples.append(example)
    return intermediate_examples


def interpolate(inference_network,
                generative_model,
                vocab):
    for _ in range(FLAGS.n_samples):
        logging.info('=== Example ===')
        example_0, z_k_0 = generate_example(inference_network,
                                            generative_model,
                                            vocab,
                                            FLAGS.beam_width)
        example_1, z_k_1 = generate_example(inference_network,
                                            generative_model,
                                            vocab,
                                            FLAGS.beam_width)
        intermediate_examples = generate_interpolants(z_k_0, z_k_1, None,
                                                      inference_network,
                                                      generative_model,
                                                      vocab,
                                                      FLAGS.beam_width)
        logging.info('Start: %s' % example_0)
        for example in intermediate_examples:
            logging.info(example)
        logging.info('End: %s' % example_1)


def sample(inference_network,
           generative_model,
           vocab):
    for _ in range(FLAGS.n_samples):
        logging.info('=== Example ===')
        z_0 = torch.randn(1, inference_network.dim)
        if torch.cuda.is_available:
            z_0 = z_0.cuda()
        z_k, _ = inference_network.normalizing_flow(z_0, None)
        for beam_width in FLAGS.beam_widths:
            beam = Beam(vocab, beam_width, generative_model)
            sample = beam.search(z_k)
            example = [vocab.id2word(int(x)) for x in sample[0]]
            try:
                T = example.index(vocab.eos_token)
                example = example[:T]
            except ValueError:
                pass
            example = ' '.join(example)
            logging.info('Beam width %i: %s' % (beam_width, example))


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

    if FLAGS.which == 'interpolate':
        interpolate(inference_network, generative_model, vocab)
    elif FLAGS.which == 'sample':
        sample(inference_network, generative_model, vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file.')
    parser.add_argument('--debug_log', type=str, default=None,
                        help='If given, write DEBUG level logging events to '
                             'this file.')
    parser.add_argument('-n', '--n_samples', type=int, default=10,
                        help='Number of samples.')

    subparsers = parser.add_subparsers()

    interpolate_parser = subparsers.add_parser('interpolate')
    interpolate_parser.add_argument('-e', '--early_interp', action='store_true',
                        help='If specified interpolation is done in z0 space '
                             'instead of zk space.')
    interpolate_parser.add_argument('-b', '--beam_width', type=int, default=1,
                                    help='Width used in beam search.')
    interpolate_parser.set_defaults(which='interpolate')

    sample_parser = subparsers.add_parser('sample')
    sample_parser.add_argument('-b', '--beam_widths', type=int, nargs='+',
                               default=[1],
                               help='Widths used in beam search.')
    sample_parser.set_defaults(which='sample')

    FLAGS, _ = parser.parse_known_args()

    main(_)

