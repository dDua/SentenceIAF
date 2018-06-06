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
        map_type: Type of map applied at each step of the flow. One of
            'planar', 'linear', 'resnet'.
        K: Number of maps to apply.
    """
    def __init__(self, dim, map_type, K, **kwargs):
        super(NormalizingFlow, self).__init__()

        self.dim = dim
        self.map_type = map_type
        self.K = K

        if map_type == 'planar':
            map = PlanarMap
        elif map_type == 'radial':
            map = RadialMap
        elif map_type == 'linear':
            map = LinearMap
        elif map_type == 'resnet':
            raise NotImplementedError('ResnetMap not currently implemented.')
        else:
            raise ValueError('Unknown `map_type`: %s' % map_type)

        self.maps = nn.ModuleList([map(dim, **kwargs) for _ in range(K)])

    def forward(self, z, h):
        """Computes forward pass of the planar flow.

        Args:
            z: torch.Tensor(batch_size, dim). The latent variable.

        Returns:
            f_z: torch.Tensor(batch_size, dim). The output of the final map
            sum_logdet: scalar. The sum of the log-determinants of the
                transformations.
        """
        f_z = z
        sum_logdet = 0.0
        for map in self.maps:
            f_z, logdet = map(f_z, h)
            sum_logdet += logdet
        return f_z, sum_logdet


class Map(nn.Module):
    """Generic parent class for maps used in a normalizing flow.

    Args:
        dim: Dimensionality of the latent variable.
    """
    def __init__(self, dim):
        super(Map, self).__init__()
        self.dim = dim

    def forward(self, z, h):
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
        super(PlanarMap, self).__init__(dim=dim)

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

    def forward(self, z, h):
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
        logdet = torch.log(torch.abs(1 + torch.matmul(psi, self.u.t())))

        return f_z, logdet


class RadialMap(Map):
    """The map used in radial flows, as described in:

        'Variational Inference with Normalizing Flows'
            Rezende and Mohamed, ICML 2015

    Args:
        dim: Dimensionality of the latent variable.
    """
    def __init__(self, dim):
        super(RadialMap, self).__init__(dim=dim)

        self.z0 = torch.nn.Parameter(torch.Tensor(1, dim))
        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        self.beta = torch.nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        """Resets radial map parameters."""
        a = sqrt(6 / self.dim)

        self.z0.data.uniform_(-a, a)
        self.alpha.data.uniform_(-a, a)
        self.beta.data.uniform_(-a, a)

    def forward(self, z, h):
        """Computes forward pass of the radial map.

        Args:
            z: torch.Tensor(batch_size, dim).
                The latent variable.

        Returns:
            f_z: torch.Tensor(batch_size, dim).
                The transformed latent variable.
            logdet: scalar.
                The log-determinant of the transformation.
        """
        # Ensure invertibility using approach in appendix A.2
        beta_prime = -self.alpha + F.softplus(self.beta)

        # Compute f_z and logdet using Equation 14.
        diff = z - self.z0
        r = torch.abs(diff)
        h = 1 / (self.alpha + r)
        dh = - (h ** 2)

        f_z = z + beta_prime * h * diff
        logdet = (1 + beta_prime * h)**(self.dim - 1) * (1 + beta_prime*h + beta_prime * dh * r)

        return f_z, logdet


class LinearMap(Map):
    """The map used in linear IAF step, as described in:

        'Improved Variational Inference with Inverse Autoregressiev Flow'
            Kingma, Salimans, Jozefowicz, Chen, Sutskever, and Welling, NIPS 2016

    Args:
        dim: Dimensionality of the latent variable.
    """
    def __init__(self, dim):
        super(LinearMap, self).__init__(dim=dim)

    def forward(self, z, h):
        """Computes forward pass of the linear map.

        Args:
            z: torch.Tensor(batch_size, dim).
                The latent variable.

        Returns:
            f_z: torch.Tensor(batch_size, dim).
                The transformed latent variable.
            logdet: scalar.
                The log-determinant of the transformation.
        """
        # Compute f_z using Equation in Appendix A.
        l_mat = torch.zeros(self.dim, self.dim)
        for i in range(self.dim):
            for j in range(0, i + 1):
                l_mat[:,i,j] = h[:, i*self.dim + j] if i < j else 1
        f_z = torch.matmul(l_mat, z.t())
        return f_z, 0

