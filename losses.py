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
"""Custom loss functions."""

import torch
import torch.nn.functional as F


def annealed_free_energy_loss(logp, target, kl, beta, ignore_index=None):
    """Computes a variant of the annealed free energy loss function given in
        Equation 20 of:

        Variational Inference with Normalizing Flows
        Rezende and Mohamed, ICML 2015

    Args:
        logp: torch.Tensor(batch_size, vocab_size). Log probabilities output by the
            generative model.
        target: torch.Tensor(batch_size, vocab_size). Observed data.
        kl: scalar. KL divergence term output by the inference network.
        ignore_index: int. Specifies a target value that is ignored and does
            not contribute to the input gradient.

    Returns:
        loss: scalar. The total loss.
    """
    nll = F.nll_loss(logp, target, ignore_index=ignore_index)
    return kl + beta * nll

