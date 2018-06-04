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
"""Utilities for the SentenceIAF model."""

import logging
import os
import torch


def configure_logging(debug_log=None):
    """Configures the root logger.

    Args:
        debug_log: str. Path to DEBUG log file.
    """
    # Define format
    format = '%(asctime)-15s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format)
    # Get root level logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Stream INFO level logging events to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    # If enabled write DEBUG level logging events to log file
    if debug_log is not None:
        fh = logging.FileHandler(debug_log, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
        logging.info('DEBUG level logging enabled. '
                     'Log written to: `%s`.' % debug_log)


def interpolate(start, end, steps):
    """Linearly interpolates between start and end vectors.

    Args:
        start: torch.Tensor(TO BE DETERMINED). Initial vector.
        end: torch.Tensor(TO BE DETERMINED). Final vector.
        steps: int. Number of interpolation steps.

    Returns:
        interpolation: torch.Tensor(steps + 2, TO BE DETERMINED). Tensor
            whose i'th element is the interpolated vector at step i.
    """
    interpolation = torch.zeros((start.shape[0], steps + 2))
    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = torch.linspace(s,e,steps+2)
    interpolation.t_()
    return interpolation

