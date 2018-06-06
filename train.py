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
import logging
import shutil
import torch
import torch.nn.functional as F
import yaml
from math import exp
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data import TextDataset
from model import RNNTextInferenceNetwork, RNNTextGenerativeModel
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
    linear = beta_0 + t * beta_growth_rate
    beta = (1 + exp(-1 * linear))**-1
    return beta


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
    summary_writer =  SummaryWriter(summary_dir)

    # Check for conflicting configurations
    safe_copy_config(config, FLAGS.force_overwrite)

    # Load vocab and datasets
    logging.info('Loading the vocabulary.')
    with open(config['data']['vocab'], 'r') as f:
        vocab = Vocab.load(f)
    logging.info('Loading train and valid data.')
    train_data = TextDataset(config['data']['train'],
                             vocab=vocab,
                             max_length=config['training']['max_length'])
    valid_data = TextDataset(config['data']['valid'],
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

    # Start train
    while epoch < config['training']['epochs']:
        logging.info('Starting epoch - %i.' % epoch)

        inference_network.train()
        generative_model.train()

        # Training step
        logging.info('Start train step.')
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available())

        # Init train summaries
        train_nll = 0.0
        train_kl = 0.0
        train_loss = 0.0

        for batch in train_loader:

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

            # Compute annealed loss
            length = logp.shape[1]
            logp = logp.view(-1, len(vocab))
            target = target[:,:length].contiguous().view(-1)
            nll = F.nll_loss(logp, target, ignore_index=vocab.pad_idx,
                             size_average=False)
            loss = nll + beta * kl

            # Update summaries
            train_nll += nll.data
            train_kl += kl.data
            train_loss += loss.data

            # Backpropagate gradients
            loss /= config['training']['batch_size']
            loss.backward()
            optimizer_in.step()
            optimizer_gm.step()

            # Log
            if not t % config['training']['log_frequency']:
                # Note: logged train loss only for a single batch - see
                # tensorboard for summary over epochs
                logging.info('Iteration: %i - Training Loss: %0.4f.' % (t, loss))

                # Print a greedy sample
                z_0 = torch.randn(1, config['model']['dim'])
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
                logging.info('Example - `%s`' % example)

            t += 1

        # Validation step
        logging.info('Start valid step.')
        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available())

        # Init valid summaries
        valid_nll = 0.0
        valid_kl = 0.0
        valid_loss = 0.0

        for batch in valid_loader:

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

            # Compute annealed loss
            length = logp.shape[1]
            logp = logp.view(-1, len(vocab))
            target = target[:,:length].contiguous().view(-1)
            nll = F.nll_loss(logp, target, ignore_index=vocab.pad_idx,
                             size_average=False)
            loss = nll + kl

            # Update summaries
            valid_nll += nll.data
            valid_kl += kl.data
            valid_loss += loss.data

        # Normalize losses
        train_nll /= len(train_data)
        train_kl /= len(train_data)
        train_loss /= len(train_data)
        valid_nll /= len(valid_data)
        valid_kl /= len(valid_data)
        valid_loss /= len(valid_data)

        # Tensorboard logging
        summary_writer.add_scalar("elbo/train", train_loss.data, epoch)
        summary_writer.add_scalar("kl/train", train_kl.data, epoch)
        summary_writer.add_scalar("nll/train", train_nll.data, epoch)
        summary_writer.add_scalar("elbo/val", valid_loss.data, epoch)
        summary_writer.add_scalar("kl/val", valid_kl.data, epoch)
        summary_writer.add_scalar("nll/val", valid_nll.data, epoch)

        # Save checkpoint
        is_best = valid_loss < best_loss
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

