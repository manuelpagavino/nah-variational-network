import torch
import time
import logging

from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator

from utils.metrics import rmse
from methods.benchmarks.ln import LeastNorm

logger = logging.getLogger(__name__)

class PointNeuronNetwork(torch.nn.Module):
    """Point Neuron Network (PNN) solution according to Section 4.2.
    
    NOTE: 
    This implementation is based on the inofficial matlab code https://github.com/HanwenBi/Point-Neuron-Learning 
    provided by the corresponding authors of the paper
    "Point Neuron Learning: A New Physics-Informed Neural Network Architecture - Bi and Abhayapala, 2024"
    and is adapted to the use-case of nearfield acoustic holography.
    """
    def __init__(
            self, esm, T=1, step_q=1, step_x=1, step_y=1, step_z=1, psi=1, lambdaa=1, 
            init_ln=True, eval_every=1e3, stop_early=True, stop_tol=3
            ):
        """Initialize hyperparameters.

        Parameters
        ----------
        esm : EquivalentSourceModel
            The underlying equivalent source model (ESM).
        T : int
            Number of PNN iterations.
        step_q : float
            Gradient step size for equivalent source strengths (PNN weights).
        step_x : float
            Gradient step size for x-coordinate equivalent source positions (PNN biases).
        step_x : float
            Gradient step size for y-coordinate equivalent source positions (PNN biases).
        step_z : float
            Gradient step size for z-coordinate equivalent source positions (PNN biases).
        psi : float
            Regularization parameter ψ of spatial CS constraint ψ*||q||_1 according to Eq. (35).
        lambdaa : float
            Regularization parameter λ of squared l2-norm constraint ||q||_2^2 for the 
            Least Norm (LN) initializer. Only relevant if init_ln=True.
        init_ln : bool
            If True, use Least Norm (LN) initialzer to start optimization from a close 
            approximate solution.
        eval_every : int
            Log RMSE every so iteration. Requires the user to provide a target reference.
        stop_early : bool
            Stop training as soon as loss starts to monotonically increase for more 
            than the tolerated number of iterations defined by stop_tol.
        stop_tol : int
            The tolerated number of iterations where loss monotonically increases before 
            early stopping the training.
        """
        super(PointNeuronNetwork, self).__init__()
        dtype = torch.float32
        self.T = T
        self.lambdaa = lambdaa
        self.psi = psi
        self.grad_clip_thresh = 0.06
        self.eval_every = eval_every
        self.init_ln = init_ln

        # init least norm initializer
        if init_ln:    
            self.ln = LeastNorm(method="tikhonov", lambdaa=lambdaa)

        # early stopping
        self.stop_tol = stop_tol 
        self.stop_early = stop_early

        # init step sizes
        self.step_q = step_q 
        self.step_coord = torch.tensor(
            (step_x, step_y, step_z)
            ).view(-1,1,1)

        # extract equivalent source model 
        self.esm = esm
        self.k = esm.k
        self.omega = esm.omega
        self.rho = esm.rho

        # init source coordinates (variable)
        retreat_dist = 1e-3 # retreat distance avoids zero div during backpropagation
        x_src = torch.tensor(esm.X_src, dtype=dtype).view(-1,1)
        y_src = torch.tensor(esm.Y_src, dtype=dtype).view(-1,1)
        z_src = torch.zeros((esm.N, 1), dtype=dtype) - retreat_dist
        self.src_coords = torch.stack((x_src, y_src, z_src), 0)

        # init mic coordinates (fixed)
        x_mic = torch.tensor(esm.X_mic, dtype=dtype).view(-1,1)
        y_mic = torch.tensor(esm.Y_mic, dtype=dtype).view(-1,1)
        z_mic = esm.dist * torch.ones((esm.M, 1), dtype=dtype)
        self.mic_coords = torch.stack((x_mic, y_mic, z_mic), 0)

    def interpolate_src(self, x_src, y_src, q):
        """Perform 2d linear barycentric interpolation."""
        interp = LinearNDInterpolator(
            points=torch.cat((x_src, y_src), 1), values=q[:,0], fill_value=0
            )
        q = interp(self.esm.X_src, self.esm.Y_src) # interpolate
        q = torch.from_numpy(q).cfloat()
        q = q.view(-1, 1) # vectorize
        return q
        
    def forward(self, p, G, q0, q_ref=None):
        """Compute PNN solution.

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
            for RMSE computation to show progress. Required for early stopping.

        Returns
        -------
        q : Tensor, (B, 1, N_y, N_x)
            PNN estimate of equivalent source strengths.
        """
        logger.info("Computing Point Neuron Network (PNN) solution ...")
        if self.stop_early and q_ref is None:
            raise ValueError("Early stopping requires ground truth source data!")

        start = time.time()
        
        # optionally init with least norm (LN) solution 
        if self.init_ln:
            q0 = self.ln(p, G, q0)

        # restrict to single-batch processing
        assert q0.shape[:2] == p.shape[:2] == (1,1)
        src_shape = q0.shape
        q0 = q0.squeeze()
        p = p.squeeze()

        # vectorize inputs
        q0 = q0.view(-1, 1)
        p = p.view(-1, 1)

        # normalize measurements
        norm = p.abs().max() 
        p = p / norm 
        
        q_t, stop_cnt, loss_t = q0, 0., 0.
        for t in tqdm(range(self.T)):
            # differences between equivalent source and microphone positions
            coord_diff = self.src_coords.mT - self.mic_coords

            # distances between equivalent sources and microphones
            dist_s = torch.sqrt(coord_diff[0] ** 2 + coord_diff[1] ** 2 + coord_diff[2] ** 2)
            # distances between equivalent sources and the origin
            dist_e = torch.sqrt(self.src_coords[0] ** 2 + self.src_coords[1] ** 2 + self.src_coords[2] ** 2).T

            # update Green's function propagation matrix
            G = torch.exp(-1j * self.k * dist_s) / (4 * torch.pi * dist_s)
            G = G * 1j * self.omega * self.rho
            G = G / norm 

            # calculate normalization factor 𝛘 according to Eq. (24)
            # nearfield 
            scale = torch.ones_like(dist_e) # 𝛘=1
            # nearfield + farfield
            #scale = torch.sqrt(dist_e) * torch.exp(-1j * self.k * torch.sqrt(dist_e)) # 𝛘 according to Eq. (24)

            # calculate pressure estimate by forward propagation
            Gq_t = scale * G @ q_t

            # calculate model mismatch gradient
            Gq_t_p =  Gq_t - p
            Gh_Gq_t_p = 2 * (scale * G).H @ Gq_t_p            

            # update weights / source strengths
            q_t = q_t - self.step_q * Gh_Gq_t_p - self.step_q * self.psi * q_t / q_t.abs()

            # calculate gradients of biases / coordinates
            dhn = q_t.T * (1j * self.k * dist_s - 1) * torch.exp(1j * self.k * dist_s) / (4 * torch.pi * dist_s ** 2) * scale
            dpn = q_t.T * -(1j * self.k * dist_s - 1) * torch.exp(1j * self.k * dist_s) * G
            grad_coord = ((coord_diff / dist_s * dhn + q_t.T / dist_e * dpn).mH @ Gq_t_p).real

            # clip gradients of biases / coordinates
            mask = torch.abs(grad_coord * self.step_coord) > self.grad_clip_thresh
            grad_coord[mask] = self.grad_clip_thresh * (grad_coord / torch.abs(grad_coord * self.step_coord))[mask]

            # update biases / coordinates
            self.src_coords = self.src_coords - self.step_coord * grad_coord

            if t == (self.T - 1) or not t % self.eval_every:
                # interpolate estimate if necessary
                q = q_t
                if self.step_coord[0] > 0 or self.step_coord[1] > 0:
                    q = self.interpolate_src(self.src_coords[0], self.src_coords[1], q_t)
                q = q.view(src_shape) # retrieve input shape

                if q_ref is not None:
                    loss_t_1 = loss_t
                    loss_t = rmse(q, q_ref)

                    # optionally log rmse
                    if not t % self.eval_every:
                        logger.info(f"RMSE @ iteration {t}: {loss_t:.2f}")

                    # stop training when loss starts to monotonically increase for more 
                    # than the tolerated number of iterations
                    if self.stop_early:
                        if loss_t_1 - loss_t < 0:
                            if stop_cnt > self.stop_tol:
                                logger.info(f"Training of PNN stopped early after {t} iterations at RMSE: {loss_t:.2f}")
                                break 
                            else:
                                stop_cnt += 1
                        else:
                            stop_cnt = 0 # reset

        logger.info(
            f"Computation Finished | "
            f"Time: {time.time() - start:.2f}s"
            )

        return q