"""Implementation of normalizing flows, initially described in:

    'Variational Inference with Normalizing Flows'
    Rezende and Mohamed, ICML 2015

"""
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizingFlow(nn.Module):
    """General implementation of a normalizing flow.

    Normalizing flows apply a series of maps to a latent variable.

    NOTE: Keyword arguments are passed to the maps.

    Args:
        dim: Dimension of the latent variable.
        map: Type of map applied at each step of the flow. Should be a subclass
            of the `Map` object.
        K: Number of maps to apply.
    """
    def __init__(self, dim, map, K, **kwargs):
        super(NormalizingFlow, self).__init__()

        self.dim = dim
        self.K = K

        self.maps = nn.ModuleList([map(dim, **kwargs) for _ in range(K)])

    def forward(self, z):
        """Computes forward pass of the planar flow.

        Args:
            z: torch.Tensor(batch_size, dim).
                The latent variable.

        Returns:
            f_z: torch.Tensor(batch_size, dim).
                The output of the final map
            sum_logdet: scalar.
                The sum of the log-determinants of the transformations.
        """
        f_z = z
        sum_logdet = 0.0
        for planar_map in self.planar_maps:
            f_z, logdet = planar_map(f_z)
            sum_logdet += logdet
        return f_z, sum_logdet


class Map(nn.Module):
    """Generic parent class for maps used in a normalizing flow.

    Args:
        dim: Dimensionality of the latent variable.
    """
    def __init__(self, dim):
        super(Map, self).__init__()

    def forward(self, z):
        """Computes the forward pass of the map.

        This method should be implemented for each subclass of Map. This
        function should always have the following signature:

        Args:
            z: torch.Tensor(batch_size, dim).
                The latent variable.

        Returns:
            f_z: torch.Tensor(batch_size, dim).
                The transformed latent variable.
            logdet: scalar.
                The log-determinant of the transformation.
        """
        raise NotImplementedError


class PlanarMap(Map):
    """The map used in planar flows, as described in:

        'Variational Inference with Normalizing Flows'
            Rezende and Mohamed, ICML 2015

    Args:
        dim: Dimensionality of the latent variable.
    """
    def __init__(self, dim):
        super(PlanarMap, self).__init__(dim)

        self.u = torch.nn.Parameter(torch.Tensor(1, dim))
        self.w = torch.nn.Parameter(torch.Tensor(1, dim))
        self.b = torch.nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        """Resets planar map parameters."""
        a = sqrt(6 / self.dim)

        self.u.data.uniform_(-a, a)
        self.w.data.uniform_(-a, a)
        self.b.data.uniform_(-a, a)

    def forward(self, z):
        """Computes forward pass of the planar map.

        Args:
            z: torch.Tensor(batch_size, dim).
                The latent variable.

        Returns:
            f_z: torch.Tensor(batch_size, dim).
                The transformed latent variable.
            logdet: scalar.
                The log-determinant of the transformation.
        """
        # Ensure invertibility using approach in appendix A.1
        wu = torch.matmul(self.u, self.w.t()).squeeze()
        mwu = F.softplus(wu) - 1
        u_hat = self.u + (mwu - wu) * self.w / torch.sum(self.w**2)

        # Compute f_z using Equation 10.
        wz = torch.matmul(self.w, z.unsqueeze(2))
        wz.squeeze_(2)
        f_z = z + self.u * F.tanh(wz + self.b)

        # Compute psi using Equation 11.
        psi = self.w * (1 - F.tanh(wz + self.b)**2)

        # Compute logdet using Equation 12.
        logdet = torch.log(torch.abs(1 + torch.matmul(self.u, psi.t())))

        return f_z, logdet

