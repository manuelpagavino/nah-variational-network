import torch
import time
import hydra 
import logging

from torch import nn
from tqdm import tqdm

from utils.metrics import rmse

logger = logging.getLogger(__name__)

class LeastNorm(nn.Module):
    """Least Norm (LN) solution via Tikohonov/TSVD-regularized pseudoinverse or Proximal Gradient Descent."""

    def __init__(self, method="pseudo", lambdaa=1.0, T=1, tau=1.0, rho=1.0, eval_every=1e3):
        super(LeastNorm, self).__init__()
        """Initialize hyperparameters.

        Parameters
        ----------
        method : str
            Computational method of the regularized least norm solution.
            Options:
                `tikhonov`: Closed-form right inverse with Tikhonov regularization
                `tsvd`: Closed-form right inverse with Truncated Singular Value Decomposition (TSVD) regularization
                'prox': Regularized Proximal Gradient Descent / Forward-Backward-Splitting
        lambdaa : float
            Regularization parameter λ of squared l2-norm constraint ||q||_2^2. 
        T : int
            Number of iterations for proximal gradient descent. Only relevant if method="prox".
        tau : float
            (Proximal) gradient descent step size.
        rho : float 
            Relaxation parameter of proximal gradient descent between [0, 1].
        eval_every : int
            Log RMSE every so iteration. Requires the user to provide a target reference.
        """
        assert method in ["tikhonov", "tsvd", "prox"]
        self.method = method
        self.lambdaa = lambdaa
        self.T = T
        self.tau = tau
        self.rho = rho
        self.eval_every = eval_every

    def grad(self, q, p, G):
        """Compute gradient of least squares model mismatch ||Gq-p||_2^2."""
        Gq = G @ q.view(*q.shape[:2], -1, 1) 
        Gq_p = Gq  - p.view(*p.shape[:2], -1, 1)
        Gh_Gq_p = G.mH @ Gq_p
        return 2 * self.tau * Gh_Gq_p.view(q.shape)

    def prox(self, q):
        """Compute proximal step to enforce squares l2-norm constraint λ*||q||_2^2."""
        return q / (1 + self.tau * self.lambdaa)
         
    def forward(self, p, G, q0, q_ref=None):
        """Compute LN solution via Tikohonov/TSVD-regularized pseudoinverse or Proximal Gradient Descent.

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
            LN estimate of equivalent source strengths.
        """
        info = "Computing Least Norm (LN) solution via "
        if self.method == "tikhonov":
            info += "closed-form right inverse with Tikhonov regularization ..."
        elif self.method == "tsvd":
            info += "closed-form right inverse with TSVD regularization ..."
        elif self.method == "prox":
            info += "regularized proximal gradient descent / forward-backward-splitting ..."
        logger.info(info)

        start = time.time()
        if self.method == "tikhonov" or self.method == "tsvd":
            # singular value decomposition (SVD)
            U, S, Vh = torch.linalg.svd(G, full_matrices=False)
            V = Vh.mH
            
            # regularize and invert singular values
            if self.method == "tikhonov":
                # apply Tikhonov smoothing
                S_inv = S / (S ** 2 + self.lambdaa)
            elif self.method == "tsvd":
                # apply truncation mask to only consider singular values and 
                # corresponding left/right singular vectors above the truncation threshold
                mask = S ** 2 > self.lambdaa
                S_inv = 1 / S * mask

            # turn real-valued singular value vectors into complex-valued diagonal matrices
            S_inv = torch.diag_embed(S_inv.cfloat())

            # compute and apply Tikhonov/TSVD-regularized right inverse
            G_pinv = V @ (S_inv @ U.mH)
            q = G_pinv @ p.view(*p.shape[:2], -1, 1)
            q = q.view(q0.shape) # rasterize

            # optionally log rmse if reference is provided
            if q_ref is not None:
                logger.info(f"RMSE: {rmse(q_ref, q):.2f}")

        elif self.method == "prox":
            q_t = q0
            for t in tqdm(range(self.T)):
                # save current iterate for relaxation step
                q_t_1 = q_t 
                
                # gradient step for model mismatch
                q_t = q_t - self.grad(q_t, p, G)

                # proximal step for LN constraint
                q_t = self.prox(q_t)

                # relaxation step
                q_t = self.rho * q_t + (1 - self.rho) * q_t_1

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
    ln = LeastNorm(
        method="prox",
        T=100,
        lambdaa=1,
        tau=1,
        rho=0,
        eval_every=10,
    )

    p = torch.randn((5,1,8,8), dtype=torch.complex64)
    G = torch.randn((5,1,8*8,41*41), dtype=torch.complex64)
    q0 = torch.randn((5,1,41,41), dtype=torch.complex64)
    q_ref = torch.randn_like(q0)

    q = ln(p, G, q0, q_ref)

if __name__ == "__main__":
    main()