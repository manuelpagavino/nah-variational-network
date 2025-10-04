import torch
import time
import hydra 
import logging

import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from utils.metrics import rmse

logger = logging.getLogger(__name__)

class TotalGeneralizedVariation(nn.Module):
    """Total (Generalized) Variation (TV/TGV) solution via generalized Primal-Dual Hybrid Gradient (PDHG) / Condat-Vu algorithm."""

    def __init__(self, lambdaa=1.0, T=1, tau=1.0, rho=1.0, mu=0.0, beta=1.0, eval_every=1e3, order=1):
        super(TotalGeneralizedVariation, self).__init__()
        """Initialize hyperparameters.

        Parameters
        ----------
        lambdaa : float
            Regularization parameter λ of Total (Generalized) Variation constraint ||∇q||_1 or ||Δq||_1. 
        T : int
            Number of PDHG iterations.
        tau : float
            (Proximal) gradient descent step size.
        rho : float 
            Relaxation parameter of proximal gradient descent between [0, 1].
        mu : float
            Weight μ of fused spatial sparsity constraint μ*||q||_1 for compressive-sensing-based 
            Total (Generalized) Variation.
        beta : float
            Lipschitz constant of propagation matrix.
        eval_every : int
            Log RMSE every so iteration. Requires the user to provide a target reference.
        order : int
            The order of the n-th order derivative operators.
            Options:
                1: Total Variation (TV) based on first-order derivates (gradient). ||∇q||_1
                2: Total Generalized Variation (TGV) based on second-order derivates (Laplacian). ||Δq||_1
        """
        assert order in [1, 2]
        self.lambdaa = lambdaa
        self.T = T
        self.tau = tau
        self.rho = rho
        self.mu = mu
        self.beta = beta
        self.eval_every = eval_every
        self.order = order
        
        I = torch.eye(100)
        eps = torch.linalg.norm(self.diff_adjoint(self.diff(I)), ord=2)
        self.sigma = (1 / eps) * (1 / self.tau - 0.5 * self.beta)

        # optinally fuse TV with an additional spatial sparsity constraint for CS-TV
        if self.mu > 0:
            self.prox_fused = self.prox_l1
        else:
            self.prox_fused = self.prox_id

    def grad(self, q, p, G):
        """Compute gradient of least squares model mismatch ||Gq-p||_2^2."""
        Gq = G @ q.view(*q.shape[:2], -1, 1) 
        Gq_p = Gq  - p.view(*p.shape[:2], -1, 1)
        Gh_Gq_p = G.mH @ Gq_p
        return 2 * Gh_Gq_p.view(q.shape)

    def prox_id(self, q):
        """Placeholder proximal step for the identity operation Id(q)=q."""
        return q

    def prox_l1(self, q):
        """Compute proximal step to enforce l1-norm constraint μ*||q||_1."""
        return torch.sgn(q) * torch.clamp(q.abs() - 0.5 * self.tau * self.mu * self.lambdaa, 0)

    def proj_l2(self, u):
        """Compute projection step to enforce l2-norm constraint λ*||q||_2."""
        denom = torch.clamp(torch.linalg.norm(u, 2, -1, keepdims=True) / self.lambdaa, min=1)
        return u / denom

    def diff(self, q, dims=2):
        """Compute element-wise n-th order derivatives with zero boundary conditions."""
        qslice = [slice(x) for x in q.shape]

        # right-pad hologram axes with zeros
        q = F.pad(q, (0,self.order) * dims, mode='constant')

        diffs = []
        for dim in range(dims):
            # compute n-th order finite differences along hologram axes
            if self.order == 1:
                diff = torch.diff(q, n=1, dim=-1-dim)
            elif self.order == 2:
                diff = torch.roll(q, shifts=-2, dims=-1-dim) \
                    - 2 * torch.roll(q, shifts=-1, dims=-1-dim) \
                        + q
            diffs.append(diff[qslice]) # remove padding

        # stack finite differences along last axis
        diffs = torch.stack(diffs, -1)

        return diffs

    def diff_adjoint(self, u):
        """Compute adjoint of element-wise n-th order derivatives with zero boundary conditions."""
        dims = u.shape[-1]

        # left-pad hologram axes with zeros
        u = F.pad(u, (0,0) + (self.order,0) * dims, mode='constant')
        if self.order == 2:
            # requires zero edge padding but circular corner padding
            u[...,*[slice(2)] * dims,:] = u[...,*[slice(-2, None)] * dims,:]

        y = 0.
        for dim in range(dims):
            # compute n-th order finite differences along hologram axes
            if self.order == 1:
                # prepend diff dim by one to keep dims
                prepend = list(u.shape[:-1])
                prepend[-1-dim] = 1
                diff = torch.diff(-u[...,dim], n=1, dim=-1-dim, prepend=torch.zeros(prepend))
            elif self.order == 2:
                diff = torch.roll(u[...,dim], shifts=2, dims=-1-dim) \
                    - 2 * torch.roll(u[...,dim], shifts=1, dims=-1-dim) \
                        + u[...,dim]
            
            # sum up finite differences without padding to form adjoint
            y = y + diff[..., *[slice(self.order, None)] * dims]

        return y

    def forward(self, p, G, q0, q_ref=None):
        """Compute TGV/TV solution via generalized PDHG / Condat-Vu algorithm.

        Parameters
        ----------
        p : Tensor, (B, 1, M_y, M_x)
            Sound pressure measurements.
        G : Tensor, (B, 1, M, N)
            Propagation matrix.
        q0 : Tensor, (B, 1, N_y, N_x)
            Initial equivalent source strengths. Typically zero or random.
        q_ref : Tensor, (B, 1, N_y, N_x)
            Optional target reference equivalent source strengths 
            for RMSE computation to show progress.

        Returns
        -------
        q : Tensor, (B, 1, N_y, N_x)
            TV/TGV estimate of equivalent source strengths.
        """
        if self.order == 1:
            variant = "Total Variation (TV)"
        elif self.order == 2:
            variant = "Total Generalized Variation (TVG)"
        logger.info(f"Computing {variant} solution via PDHG ...")

        start = time.time()

        q_t, u_t = q0, torch.zeros(q0.shape + (2,))
        for t in tqdm(range(self.T)):
            # save current iterates
            q_t_1 = q_t
            u_t_1 = u_t
            
            # gradient step for model mismatch
            q_t = q_t - self.tau * self.grad(q_t, p, G)

            # subtract gradient adjoint of dual hologram
            q_t = q_t - self.tau * self.diff_adjoint(u_t)

            # optional proximal step for fused sparsity constraint
            q_t = self.prox_fused(q_t)

            # compute dual variable
            u_t = u_t + self.sigma * self.diff(2 * q_t - q_t_1)
            u_t = self.proj_l2(u_t)

            # relaxation steps
            q_t = self.rho * q_t + (1 - self.rho) * q_t_1
            u_t = self.rho * u_t + (1 - self.rho) * u_t_1

            # optionally log rmse if reference is provided
            if q_ref is not None and not t % self.eval_every:
                logger.info(f"RMSE @ iteration {t}: {rmse(q_t, q_ref):.2f}")
        q = q_t

        logger.info(
            f"Computation Finished | "
            f"Time: {time.time() - start:.2f}s"
            )
        
        return q

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(args):
    tgv = TotalGeneralizedVariation(
        lambdaa=1.0, T=100, tau=1.0, rho=1.0, mu=0.0,
        beta=1.0, eval_every=1e3, order=1
    )

    p = torch.randn((5,1,8,8), dtype=torch.complex64)
    G = torch.randn((5,1,8*8,41*41), dtype=torch.complex64)
    q0 = torch.randn((5,1,41,41), dtype=torch.complex64)
    q_ref = torch.randn_like(q0)

    q = tgv(p, G, q0, q_ref)

if __name__ == "__main__":
    main()