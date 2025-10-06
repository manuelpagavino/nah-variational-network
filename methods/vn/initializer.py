import torch
import math

import torch.nn as nn

from cplxmodule.cplx import conv2d

from utils.params import create_filter
from utils.activations import CplxGaussianRBF
from utils.norm import init_instance_norm

class Initializer(nn.Module):
    """Data-driven continued truncated singular value decomposition (TSVD) initializer according to Eq. (31)."""

    def __init__(self, args):
        super(Initializer, self).__init__()
        self.args = args

        # filters
        self.filter_u1 = create_filter(args.filter_u1)
        self.filter_u2 = create_filter(args.filter_u2)
        self.filter_u3 = create_filter(args.filter_u3)
        self.filters = [self.filter_u1, self.filter_u2, self.filter_u3]

        # activations
        self.activation_u1 = CplxGaussianRBF(args.activation_u1)
        self.activation_u2 = CplxGaussianRBF(args.activation_u2)
        self.activation_u3 = CplxGaussianRBF(args.activation_u3)
        self.activations = [self.activation_u1, self.activation_u2, self.activation_u3]

        # instance normalization
        self.norm_in = init_instance_norm(args.activation_u1.num_filters)
        self.norm_out = init_instance_norm(args.activation_u3.num_filters)
        self.norms = [self.norm_in, self.norm_in, self.norm_out]

    def layer(self, x, kernel, activation, norm):
        """Compute initializer layer."""
        # filter
        x = conv2d(
            input=x,
            weight=kernel.to(x.device),
            bias=None,
            padding="same",
            padding_mode="zeros",
        ) 

        # instance normalization
        x = norm(x)
        
        # activation
        x = torch.complex(x.real, x.imag)
        x = activation(x) 
        
        return x

    def calc_lambdaa(self, p, snr, s_max, N):
        """Calculate Tikhonov regularization parameter λ according to Halds formula Eq. (25).
        
        Parameters
        ----------
        p : Tensor, (B, 1, M_y, M_x)
            Sound pressure measurements.
        snr : Tensor, (B,)
            Estimated signal-to-noise ratio of each measurement p.
        s_max : Tensor, (B,)
            Largest singular value of the propagation matrix G linked each measurement p.
        N : int
            Total number of equivalent sources.

        Returns
        -------
        lambdaa : Tensor, (B,)
            The Tikhonov regularization parameter λ according to Halds formula Eq. (25)
            for each measurement.
        """
        # Tikhonov regularization parameter according to Hald
        p_norm = torch.linalg.norm(p.flatten(1,3), 2, -1)
        lambdaa = s_max * 10 ** (-snr / 20) * p_norm / math.sqrt(N)

        return lambdaa
    
    def tsvd(self, U, S, V, lambdaa, tikhonov=True):
        """Calculate truncated singular value decomposition (TSVD) according to Eq. (13).

        Parameters
        ----------
        U : Tensor, (B, 1, M, M)
            Matrix of left singular vectors of the propagation matrix G.
        S : Tensor, (B, 1, M)
            Vector of singular values of the propagation matrix G.
        V : Tensor, (B, 1, N, M)
            Matrix of right singular vectors of the propagation matrix G.
        lambdaa : Tensor, (B, 1, 1)
            The regularization parameter λ serving as the truncation threshold.
        tikhonov : bool
            Optionally apply additional Tikhonov regularization for further smoothing.
        

        Returns
        -------
        G_pinv_high : Tensor, (B, 1, N, M)
            The TSVD-regularized right inverse considering only singular values
            and corresponding left/right singular vectors above the truncation threshold.
        G_pinv_low : Tensor, (B, 1, N, M) 
            The TSVD-regularized right inverse considering only singular values
            and corresponding left/right singular vectors below the truncation threshold.
        """
        # compute truncation mask to only consider singular values and 
        # corresponding left/right singular vectors above the truncation threshold
        mask = S ** 2 > lambdaa

        # split matrix of left singular vectors at threshold
        U_high = U * mask.unsqueeze(1)
        U_low = U - U_high

        # split matrix of right singular vectors at threshold
        V_high = V * mask.unsqueeze(1)
        V_low = V - V_high

        # split vector of singular values at threshold
        if tikhonov:
            S_inv = S / (S ** 2 + lambdaa)
        else:
            S_inv = 1 / S
        S_inv_high = S_inv * mask 
        S_inv_low = S_inv - S_inv_high

        # turn real-valued singular value vectors into complex-valued diagonal matrices
        S_inv_high = torch.diag_embed(S_inv_high.cfloat())
        S_inv_low = torch.diag_embed(S_inv_low.cfloat())

        # compute TSVD(+Tikhonov)-regularized right inverse and residual 
        G_pinv_high = V_high @ (S_inv_high @ U_high.mH) 
        G_pinv_low = V_low @ (S_inv_low @ U_low.mH) 

        return G_pinv_high, G_pinv_low

    def forward(self, p, G, q0, snr):
        """Apply data-driven continued truncated singular value decomposition (TSVD) according to Eq. (31).
        
        Parameters
        ----------
        p : Tensor, (B, 1, M_y, M_x)
            Sound pressure measurements.
        G : Tensor, (B, 1, M, N)
            Propagation matrix.
        q0 : Tensor, (B, 1, N_y, N_x)
            Initial equivalent source strengths used for shape extraction. 
            Will be overwritten at the output.
        snr : Tensor, (B,)
            Estimated signal-to-noise ratio of each measurement p.

        Returns
        -------
        q0 : Tensor, (B, 1, N_y, N_x)
            The source strengths resulting from the data-driven continued truncated singular 
            value decomposition (TSVD) according to Eq. (31).
        s_max : Tensor, (B,)
            Largest singular value of the propagation matrix G linked each measurement p.
        """
        # singular value decomposition (SVD)
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        V = Vh.mH

        # largest singular value, needed for scaling of least squares gradient (model mismatch)
        s_max = torch.amax(S, (1,2))

        # Tikhonov regularization parameter according to Halds formula Eq. (25)
        lambdaa  = self.calc_lambdaa(
            p=p, 
            snr=snr, 
            s_max=s_max, 
            N=G.shape[-1] or q0.shape[-2:].numel()
            ).view(-1,1,1,1)

        # truncated singular value decomposition (TSVD) according to Eq. (13)
        G_pinv_high, G_pinv_low = self.tsvd(U, S, V, lambdaa.squeeze(-1))
        q_tsvd = G_pinv_high @ p.view(*p.shape[:2], -1, 1) 

        # data-driven cont'd TSVD according to Eq. (31)
        p_amax = torch.amax(p.abs(), dim=(2, 3), keepdim=True)
        x = p / p_amax

        for kernel, activation, norm in zip(
            self.filters, self.activations, self.norms
            ):
            x = self.layer(x, kernel, activation, norm)
        
        # weighted skip connection
        x = 0.5 * (x * p_amax + p)
        
        # compose output by combining TSVD with its data-driven continuation
        q_contd = G_pinv_low @ x.view(*x.shape[:2], -1, 1) 
        q0 = (q_tsvd + q_contd).view(q0.shape)

        return q0, s_max

