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
"""Implementation of normalizing flows, initially described in:

    'Variational Inference with Normalizing Flows'
    Rezende and Mohamed, ICML 2015

"""
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
            map = ResNetMap
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
        r = diff.norm(p=2, dim=1, keepdim=True)
        h = 1 / (self.alpha + r)
        dh = - (h ** 2)

        f_z = z + beta_prime * h * diff
        det = (1 + beta_prime * h)**(self.dim - 1) * (1 + beta_prime * h + beta_prime * dh * r)
        logdet = torch.log(det)

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
        self.cuda_available = torch.cuda.is_available()

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
        batch_size = z.shape[0]

        # Compute f_z using Equation in Appendix A.
        l_mat = torch.zeros(batch_size, self.dim, self.dim)
        if torch.cuda.is_available():
            l_mat = l_mat.cuda()
        k = 0
        for i in range(self.dim):
            for j in range(0, i + 1):
                l_mat[:, i, j] = h[:, k] if j < i else 1
                if i != j:
                    k += 1
        f_z = torch.matmul(l_mat, z.unsqueeze(2)).squeeze()
        return f_z, torch.zeros(1).to('cuda' if self.cuda_available else 'cpu')


class IAF(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, latent_size, downsample=False):
        super(IAF,self).__init__()
        self.hidden_width_up = hidden_size2+2*latent_size+hidden_size2
        self.hidden_width_dn_pos = hidden_size2+2*latent_size
        self.hidden_width_dn_prior = hidden_size2+2*latent_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.latent_size = latent_size
        self.downsample = downsample
        
        self.auto_reg_upconv1 = nn.Sequential()
        self.auto_reg_upconv1.add_module('relu1_up',nn.ELU())
        if downsample:
            self.auto_reg_upconv1.add_module('width1_up',nn.Conv2d(hidden_size1,\
                                self.hidden_width_up,(4,1),stride=(2,1),padding=(1,0)))
        else:
            self.auto_reg_upconv1.add_module('width1_up',nn.Conv2d(hidden_size1,\
                                self.hidden_width_up,(3,1), padding=(1,0)))
        
        self.auto_reg_upconv2 = nn.Sequential()
        self.auto_reg_upconv2.add_module('relu2_up',nn.ELU())
        self.auto_reg_upconv2.add_module('width2_up',nn.Conv2d(hidden_size2,\
                                hidden_size1,(3,1), padding=(1,0)))
        
        self.auto_reg_dnconv1 = nn.Sequential()
        self.auto_reg_dnconv1.add_module('relu1_dn',nn.ELU())
        self.auto_reg_dnconv1.add_module('width1_dn',nn.Conv2d(hidden_size1,\
                        self.hidden_width_dn_pos+self.hidden_width_dn_prior,(3,1), padding=(1,0)))
        
        self.auto_reg_dnconv2 = nn.Sequential()
        self.auto_reg_dnconv2.add_module('relu2_dn',nn.ELU())
        if downsample:
            self.auto_reg_dnconv2.add_module('width2_dn',\
                        nn.ConvTranspose2d(hidden_size2+latent_size,hidden_size1,(4,1),stride=(2,1),padding=(1,0)))
        else:
            self.auto_reg_dnconv2.add_module('width2_dn',\
                        nn.ConvTranspose2d(hidden_size2+latent_size,hidden_size1,(3,1), padding=(1,0)))
        
        self.posterior_conv = nn.Conv2d(latent_size, hidden_size2,(3,1), padding=(1,0))
        self.posterior_relu = nn.ELU()
        
        self.posterior_mu = nn.Conv2d(hidden_size2,latent_size,(3,1), padding=(1,0))
        self.posterior_sig = nn.Conv2d(hidden_size2,latent_size,(3,1), padding=(1,0))
        
    def sample_gaussian(self, mean, logvar, eps=None):
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = Variable(torch.randn(mean.size())).cuda()
        sample = eps * std + mean
        # Can't use eps directly becase no grads for random variables
        log_qz =  ((sample - mean)**2 / torch.exp(logvar))  + logvar
        const_pi = Variable(torch.log(torch.FloatTensor([2*math.pi]))).cuda()
        log_qz = log_qz + const_pi
        log_qz = -0.5*log_qz
        log_q = log_qz.view(log_qz.size(0),-1).sum(1)
        entr = (.5 * (const_pi + 1 + logvar)).view(log_qz.size(0),-1).sum(1)
        kl = lambda p_mean, p_logvar: (.5 * (p_logvar - logvar) + (torch.exp(logvar) + (mean-p_mean)**2)/(2*torch.exp(p_logvar)) - .5).view(log_qz.size(0),-1).sum(1)
        return entr, kl, log_qz, log_q, sample
    
    def top_down(self, inp, qz):
        hidden = self.auto_reg_dnconv1(inp)
        h_det = hidden[:,:self.hidden_size2,:,:]
        prior_mean = hidden[:,self.hidden_size2:self.hidden_size2+self.latent_size,:,:]
        prior_logsd = hidden[:,self.hidden_size2+self.latent_size:self.hidden_width_dn_prior,:,:]
    
        pz_mean = hidden[:,self.hidden_width_dn_prior:self.hidden_width_dn_prior+self.latent_size,:,:]
        pz_logsd = hidden[:,self.hidden_width_dn_prior+self.latent_size:self.hidden_width_dn_prior+2*self.latent_size,:,:]
        down_context = hidden[:,self.hidden_width_dn_prior+2*self.latent_size:self.hidden_width_dn_prior+2*self.latent_size+self.hidden_size2,:,:]
#         entr, kl, log_qz, log_q, sample
        samples = self.sample_gaussian(qz['qz_mean']+pz_mean, 2*pz_logsd+qz['qz_logsd'])
        context = qz['up_context'] + down_context
        z = samples[4]
        logqz = samples[2]
        
        h = self.posterior_conv(z)
        
        h = context + h
        h = self.posterior_relu(h)
        arw_mean = self.posterior_mu(h)
        arw_logsd = self.posterior_sig(h)
        arw_mean = arw_mean*0.1
        arw_logsd = arw_logsd*0.1
        z = (z - arw_mean) / torch.exp(arw_logsd)
        logqz += arw_logsd
        h_det = torch.cat([h_det, z], 1)
        
        logpz = self.sample_gaussian(prior_mean, 2*prior_logsd, z)[2]
        if self.downsample:
            bs,nf,y,z = inp.size()
            inp = inp.view(bs, nf, y, 1, z)
            inp = torch.cat([inp,inp],3)
            inp = inp.view(bs, nf, -1, z)
        output = inp + 0.1 * self.auto_reg_dnconv2(h_det)
        
        return output, logqz-logpz
    
    def bottom_up(self, inp):
        hidden = self.auto_reg_upconv1(inp)
        h_det = hidden[:,:self.hidden_size1,:,:]
        qz_mean = hidden[:,self.hidden_size1:self.hidden_size1+self.latent_size,:,:]
        qz_logsd = hidden[:,self.hidden_size1+self.latent_size:self.hidden_size1+2*self.latent_size,:,:]
        up_context = hidden[:,self.hidden_size1+2*self.latent_size:,:,:]
        samples = self.sample_gaussian(qz_mean, 2*qz_logsd)
        if self.downsample:
            bs,nf,y,z = inp.size()
            inp = inp.view(bs, nf, int(y/2), 2, z)
            inp = inp.mean(3)
        output = inp + 0.1 * self.auto_reg_upconv2(h_det)
        return output, up_context, samples, qz_mean, qz_logsd
    
class ResNetMap(Map):
    def __init__(self, dim):
        super(ResNetMap,self).__init__(dim=dim)
        self.nin1 = 64
        self.nin2 = 64
        self.nlatent = dim
        self.layers = []
        self.depths = [2,2]
        self.kl_min = Variable(torch.FloatTensor([0.25])).cuda()
        
        for t in range(0,len(self.depths)):
            made_layer = []
            for j in range(0, self.depths[t]):
                downsample = (t > 0 and j == 0)
                layer = IAF(self.nin1, self.nin2, self.nlatent, downsample).cuda()
                made_layer.append(layer)   
            self.layers.append(made_layer)
        
        self.qz = [None]*len(self.depths)
   
    def forward(self, z, hidden): 
        for i in range(len(self.depths)):
            self.qz[i] = [None]*self.depths[i]
            for j in range(self.depths[i]):
                hidden, up_context, samples, qz_mean, qz_logsd = self.layers[i][j].bottom_up(hidden)
                qz= {}
                qz['qz_logsd'] = qz_logsd
                qz['qz_mean'] = qz_mean
                qz['up_context'] = up_context
                qz['samples'] = samples
                self.qz[i][j] = qz
        
        temp_h = Variable(torch.randn(1, self.nin1, 1, 1)).cuda()
        hidden = temp_h.repeat(torch.Size([hidden.size(0),1,hidden.size(2), hidden.size(3)]))
        
        obj_kl = 0
        for i in list(reversed(range(len(self.depths)))):
            for j in list(reversed(range(self.depths[i]))):
                hidden, kl = self.layers[i][j].top_down(hidden, self.qz[i][j])
                kl = kl.sum(3).sum(2).mean(0)
                obj_kl += torch.max(self.kl_min, kl).sum(0)
        
        return hidden, obj_kl

        
