import math
import torch
import time
import hydra 
import logging

from torch import nn
from tqdm import tqdm

from utils.metrics import rmse

logger = logging.getLogger(__name__)

class CompressiveSensing(nn.Module):
    """Compressive Sensing (CS) based on Fast Iterative Shrinkage Thresholding Algorithm (FISTA)."""

    def __init__(self, lambdaa=1.0, T=1, tau=1.0, eval_every=1e3):
        super(CompressiveSensing, self).__init__()
        """Initialize hyperparameters.

        Parameters
        ----------
        lambdaa : float
            Regularization parameter λ of l1-norm constraint ||q||_1. 
        T : int
            Number of FISTA iterations.
        tau : float
            (Proximal) gradient descent step size.
        eval_every : int
            Log RMSE every so iteration. Requires the user to provide a target reference.
        """
        self.lambdaa = lambdaa
        self.T = T
        self.tau = tau
        self.eval_every = eval_every

    def grad(self, q, p, G):
        """Compute gradient of least squares model mismatch ||Gq-p||_2^2."""
        Gq = G @ q.view(*q.shape[:2], -1, 1) 
        Gq_p = Gq  - p.view(*p.shape[:2], -1, 1)
        Gh_Gq_p = G.mH @ Gq_p
        return 2 * self.tau * Gh_Gq_p.view(q.shape)

    def prox(self, q):
        """Compute proximal step to enforce l1-norm constraint λ*||q||_1."""
        return torch.sgn(q) * torch.clamp(q.abs() - self.tau * self.lambdaa, 0)

    def forward(self, p, G, q0, q_ref=None):
        """Compute CS solution via FISTA.

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
            CS estimate of equivalent source strengths.
        """
        logger.info("Computing Compressive Sensing (CS) solution via FISTA ...")
        
        start = time.time()

        # run iterations
        q_t, y_t, k_t = q0, q0, 1
        for t in tqdm(range(self.T)):           
            # save current iterates
            q_t_1 = q_t
            k_t_1 = k_t

            # gradient step for model mismatch
            q_t = y_t - self.grad(y_t, p, G)

            # proximal step for CS constraint
            q_t = self.prox(q_t)

            # FISTA param updates
            k_t = 0.5 * (1 + math.sqrt(1 + 4 * k_t_1 ** 2))
            y_t = q_t + (k_t_1 - 1) / k_t  * (q_t - q_t_1)

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
    cs = CompressiveSensing(
        T=100,
        lambdaa=1,
        tau=1,
        eval_every=10,
    )

    p = torch.randn((5,1,8,8), dtype=torch.complex64)
    G = torch.randn((5,1,8*8,41*41), dtype=torch.complex64)
    q0 = torch.randn((5,1,41,41), dtype=torch.complex64)
    q_ref = torch.randn_like(q0)

    q = cs(p, G, q0, q_ref)

if __name__ == "__main__":
    main()


