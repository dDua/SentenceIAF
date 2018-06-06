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
"""Training script."""

import argparse
import os
import json
import logging
import numpy as np
import shutil
import sys
import time
import torch
import yaml
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import defaultdict

from data import TextDataset
from model import RNNTextInferenceNetwork, RNNTextGenerativeModel
from losses import annealed_free_energy_loss
from utils import Vocab, configure_logging, word_dropout


FLAGS = None


def safe_copy_config(config, force_overwrite=False):
    """
    This function attempts to save a copy of the current config to the
    checkpoint directory.

    If an existing configuration is detected and there is a conflict then a
    ValueError will be raised unless `force_overwrite` is enabled.

    Args:
        config: dict. The model configuration.
        force_overwrite: bool. Whether or not to overwrite the existing config
            in case of conflict.
    """
    # Check for existing configuration and handle conflicts
    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])
    config_path = os.path.join(ckpt_dir, 'config.yaml')
    if os.path.exists(config_path):
        logging.info('Existing configuration file detected.')
        with open(config_path, 'r') as f:
            existing_config = yaml.load(f)
        if config != existing_config:
            if force_overwrite:
                logging.warn('Specified configuration does not match '
                            'existing configuration in checkpoint directory. '
                            'Forcing overwrite of existing configuration.')
                shutil.copyfile(FLAGS.config, config_path)
            else:
                raise ValueError('Specified configuration does not match '
                                 'existing configuration in checkpoint '
                                 'directory.')
        else:
            logging.info('Current configuration matches existing '
                         'configuration.')
    else:
        logging.info('No existing configuration found. Copying config file '
                     'to `%s`.' % config_path)
        shutil.copyfile(FLAGS.config, config_path)


def get_beta(config, t):
    beta_0 = config['training']['beta_0']
    beta_growth_rate = config['training']['beta_growth_rate']
    return min(1.0, beta_0 + t * beta_growth_rate)


def save_checkpoint(state, is_best, filename):
    logging.info('Saving checkpoint to: `%s`.' % filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')


def main(_):
    # Set up logging
    configure_logging(FLAGS.debug_log)

    # Load configuration
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    # Get the directory paths
    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])
    summary_dir = os.path.join(config['training']['summary_dir'],
                               config['experiment_name'])

    # Create the directories if they do not already exist
    if not os.path.exists(ckpt_dir):
        logging.info('Creating checkpoint directory: `%s`.' % ckpt_dir)
        os.makedirs(ckpt_dir)
    if not os.path.exists(summary_dir):
        logging.info('Creating summary directory: `%s`.' % summary_dir)
        os.makedirs(summary_dir)

    # Check for conflicting configurations
    safe_copy_config(config, FLAGS.force_overwrite)

    # Load vocab and datasets
    logging.info('Loading the vocabulary.')
    with open(config['data']['vocab'], 'r') as f:
        vocab = Vocab.load(f)
    logging.info('Loading training and validation data.')
    train_data = TextDataset(config['data']['train'],
                             vocab=vocab,
                             max_length=config['training']['max_length'])
    val_data = TextDataset(config['data']['val'],
                           vocab=vocab,
                           max_length=config['training']['max_length'])

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

    # Setup model optimizers
    optimizer_in = torch.optim.Adam(inference_network.parameters(),
                                    lr=config['training']['learning_rate'])
    optimizer_gm = torch.optim.Adam(generative_model.parameters(),
                                    lr=config['training']['learning_rate'])

    # Restore
    ckpt = os.path.join(ckpt_dir, 'model.pt')
    if os.path.exists(ckpt):
        logging.info('Model checkpoint detected at: `%s`. Restoring.' % ckpt)
        checkpoint = torch.load(ckpt)
        epoch = checkpoint['epoch']
        t = checkpoint['t']
        best_loss = checkpoint['best_loss']
        inference_network.load_state_dict(checkpoint['state_dict_in'])
        generative_model.load_state_dict(checkpoint['state_dict_gm'])
        optimizer_in.load_state_dict(checkpoint['optimizer_in'])
        optimizer_gm.load_state_dict(checkpoint['optimizer_gm'])
    else:
        logging.info('No existing checkpoint found.')
        epoch = 0
        t = 0
        best_loss = float('inf')

    # Start training
    while epoch < config['training']['epochs']:
        logging.info('Starting epoch - %i.' % epoch)

        inference_network.train()
        generative_model.train()

        # Training step
        training_loader = DataLoader(
            dataset=train_data,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available())

        for batch in training_loader:

            optimizer_in.zero_grad()
            optimizer_gm.zero_grad()

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
            x_hat = word_dropout(x, config['training']['word_dropout_rate'],
                                 vocab.unk_idx)
            logp, _ = generative_model(z, x_hat, lengths)

            # Obtain current value of the annealing constant
            beta = get_beta(config, t)

            # Compute loss
            length = logp.shape[1]
            logp = logp.view(-1, len(vocab))
            target = target[:,:length].contiguous().view(-1)
            loss = annealed_free_energy_loss(logp, target, kl, beta,
                                             ignore_index=vocab.pad_idx)

            # Backpropagate gradients
            loss.backward()
            optimizer_in.step()
            optimizer_gm.step()

            # Log
            if not t % config['training']['log_frequency']:
                logging.info('Iteration: %i - Loss: %0.4f.' % (t, loss))
                # Print a greedy sample
                z = torch.randn(1, config['model']['dim'])
                if torch.cuda.is_available:
                    z = z.cuda()
                logp, sample = generative_model(z)
                example = [vocab.id2word(int(x)) for x in sample[0]]
                try:
                    T = example.index(vocab.eos_token)
                    example = example[:T]
                except ValueError:
                    pass
                example = ' '.join(example)
                logging.info('Example - `%s`' % example)

            t += 1

        # TODO: Evaluate validation loss.

        # TODO: Log to tensorboard

        # Save checkpoint.
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            't': t,
            'best_loss': best_loss,
            'state_dict_in': inference_network.state_dict(),
            'state_dict_gm': generative_model.state_dict(),
            'optimizer_in': optimizer_in.state_dict(),
            'optimizer_gm': optimizer_gm.state_dict()
        }, is_best, ckpt)

        epoch += 1

            # if args.tensorboard_logging:
            #     writer.add_scalar("%s/ELBO"%split.upper(), loss.data[0], epoch*len(data_loader) + iteration)
            #     writer.add_scalar("%s/NLL Loss"%split.upper(), NLL_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
            #     writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
            #     writer.add_scalar("%s/KL Weight"%split.upper(), KL_weight, epoch*len(data_loader) + iteration)

            #     if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
            #         print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
            #             %(split.upper(), iteration, len(data_loader)-1, loss.data[0], NLL_loss.data[0]/batch_size, KL_loss.data[0]/batch_size, KL_weight))

            #     if split == 'valid':
            #         if 'target_sents' not in tracker:
            #             tracker['target_sents'] = list()
            #         tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
            #         tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            # print("%s Epoch %02d/%i, Mean ELBO %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file.')
    parser.add_argument('-f', '--force_overwrite', action='store_true',
                        help='Force config overwrite.')
    parser.add_argument('--debug_log', type=str, default=None,
                        help='If given, write DEBUG level logging events to '
                             'this file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

