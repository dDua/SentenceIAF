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
import os
import logging
import torch
import yaml

from model import RNNTextInferenceNetwork, RNNTextGenerativeModel
from utils import Vocab, interpolate, configure_logging


def generate_example(inference_network, generative_model, vocab):
        # Infer two greedy samples
        z_0 = torch.randn(1, inference_network.dim)
        if torch.cuda.is_available:
            z_0 = z_0.cuda()
        #  TODO: Figure out wtf to do w/ `h`...
        h = None
        z_k, _ = inference_network.normalizing_flow(z_0, h)
        _, sample = generative_model(z_k)
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
                                            vocab)
        example_1, z_k_1 = generate_example(inference_network,
                                            generative_model,
                                            vocab)
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

