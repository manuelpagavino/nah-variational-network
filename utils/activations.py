import torch
import math

import numpy as np

from torch import nn
from sklearn.neighbors import NearestNeighbors

from utils.params import create_activation_params

class CplxGaussianRBF(nn.Module):
    """Complex-valued Gaussian Radial Basis Function (RBF) activation according to Eqs. (28-30)."""

    def __init__(self, args):
        super(CplxGaussianRBF, self).__init__()
        self.num_weights = args.num_weights

        # RBF limits
        self.vmin = args.vmin
        self.vmax = args.vmax

        # Gaussian width
        delta = (self.vmax - self.vmin) / (math.isqrt(self.num_weights) - 1)
        self.delta = delta
        self.stddev = delta * delta

        # Gaussian centers
        mu = self.sunflower(self.num_weights, alpha=1) * self.vmax

        # Gaussian stddev by k-nearest neighbors (k=3)
        # see https://scikit-learn.org/stable/modules/neighbors.html
        X = np.stack((mu.real, mu.imag), -1)
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
        dist, idx = nbrs.kneighbors(X)
        delta = np.amin(dist[:,1:], 1) # discard the distance of the node to itself
        stddev = delta * delta
        
        # convert ndarray to tensor
        self.mu = torch.as_tensor(mu, dtype=torch.complex64).view(1,1,1,-1)
        self.stddev = torch.as_tensor(stddev, dtype=torch.float32)

        # RBF weights and biases
        self.weights = torch.nn.Parameter(create_activation_params(args), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1, args.num_filters, 1), dtype=torch.complex64))

    def sunflower(self, n, alpha):
        """Sunflower seed arangement.
        
        See https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
        adapted for distribution on cplx plane.

        Parameters
        ----------
        n : int
            Number of points in the point cloud.
        alpha : float
            Evenness of point cloud at the boundary.

        Returns
        -------
        out : ndarray, (n,)
            Complex-valued point cloud array arranged according to a sunflower seed pattern
            on the complex plane.
        """
        def radius(k,n,b):
            if k > (n-b):
                r = 1
            else:
                r = np.sqrt(k-1/2) / np.sqrt(n-(b+1)/2)
            return r

        b = np.round(alpha * np.sqrt(n))
        phi = (np.sqrt(5) + 1) * 0.5
        out = []
        for k in range(1, n+1):
            r = radius(k, n, b)
            theta = 2 * np.pi * k / phi ** 2
            out += [r*np.cos(theta) + 1j * r*np.sin(theta)]
        out = np.stack(out, 0)
        
        return out 

    def forward(self, x):
        """Apply RBF activation function according to Eq. (28).
        
        Parameters
        ----------
        x : Tensor, (B, C, H, W)
            Input tensor before RBF activation.

        Returns
        -------
        rbf : Tensor, (B, C, H, W)
            Output tensor after RBF activation.
        """
        B, C, H, W = x.shape
        x = x.view(B, C, -1, 1)

        self.mu = self.mu.to(x.device)
        self.stddev = self.stddev.to(x.device)

        dist = x - self.mu
        rbf = torch.exp(-(dist.conj() * dist) / (2 * self.stddev))

        rbf = torch.einsum('abcd,bd->abc', rbf, self.weights) + self.bias
        rbf = rbf.view(B, C, H, W)

        return rbf
    

