import torch
import hydra

import torch.nn as nn
import torch.nn.functional as F

from cplxmodule.cplx import conv2d

from utils.params import create_filter
from utils.activations import CplxGaussianRBF
from utils.norm import init_instance_norm

class Mismatch(nn.Module):
    """Data fidelity or model mismatch network according to Eq. (19)."""

    def __init__(self, args, layer):
        super(Mismatch, self).__init__()
        self.args = args
        self.layer = layer

        # pad
        self.pad_A = args.filter_A.pad
        self.pad_B = args.filter_B.pad
        self.pad_mode_A = args.filter_A.pad_mode
        self.pad_mode_B = args.filter_A.pad_mode

        # filters
        self.filter_A = create_filter(args.filter_A)
        self.filter_B = create_filter(args.filter_B)

        # activations
        self.activation = CplxGaussianRBF(args.activation)

        # instance normalization
        self.norm = init_instance_norm(args.activation.num_filters)
        
        # step size for unconditioned model mismatch
        self.tau = torch.nn.Parameter(torch.as_tensor(0.3), requires_grad=True)

    def forward(self, q, p, G, s_max):
        """Apply model mismatch network to estimate the model mismatch gradients according to Eq. (19).
        
        Parameters
        ----------
        q : Tensor, (B, 1, N_y, N_x)
            Equivalent source strengths.
        p : Tensor, (B, 1, M_y, M_x)
            Sound pressure measurements.
        G : Tensor, (B, 1, M, N)
            Propagation matrix.
        s_max : Tensor, (B,)
            Largest singular value of the propagation matrix G linked each measurement p.

        Returns
        -------
        Dq : Tensor, (B, N_D, N_y, N_x)
            A set of N_D model mismatch gradient estimates according to Eq. (19).
        """
        # fixed unconditioned Least Squares model mismatch

        Gq = G @ q.view(*q.shape[:2], -1, 1) 
        Gq_p = Gq  - p.view(*p.shape[:2], -1, 1)
        Gh_Gq_p = G.mH @ Gq_p

        grad_scale = 1 / s_max.view(-1,1,1,1)
        Dq_fixed = self.tau * (grad_scale ** 2) * Gh_Gq_p.view(q.shape)

        # learned parametric model mismatch according to Eq. (19)

        # right preconditioning
        q = F.pad(q, [self.pad_B] * 4, self.pad_mode_B)
        q = conv2d(
            input=q,
            weight=self.filter_B, 
            bias=None,
            padding=self.args.filter_B.kernel_size // 2,
            padding_mode="zeros"
        )
        q = q[:, :, self.pad_B:-self.pad_B, self.pad_B:-self.pad_B]

        # model mismatch
        q = torch.complex(q.real, q.imag)
        Gq = G @ q.view(*q.shape[:2], -1, 1) 
        Gq_p = Gq.view(*q.shape[:2], *p.shape[2:])  - p

        # scale gradient
        Gq_p = grad_scale * Gq_p
        
        # left preconditioning
        Gq_p = F.pad(Gq_p, [self.pad_A] * 4, self.pad_mode_A)
        Gq_p = conv2d(
            input=Gq_p,
            weight=self.filter_A,
            bias=None,
            padding=self.args.filter_A.kernel_size // 2,
            padding_mode="zeros",
        )
        
        # instance norm
        Gq_p = self.norm(Gq_p) 

        # activation
        Gq_p = torch.complex(Gq_p.real, Gq_p.imag) # convert Cplx to torch.complex
        Gq_p = self.activation(Gq_p)

        # left preconditioning transposed
        Gq_p = conv2d(
            input=Gq_p,
            weight=torch.flip(self.filter_A, dims=(2,3)).conj(),
            bias=None,
            padding=self.args.filter_A.kernel_size // 2, 
            padding_mode="zeros",
        )
        Gq_p = Gq_p[:, :, self.pad_A:-self.pad_A, self.pad_A:-self.pad_A]
        
        # propagator transposed
        Gq_p = torch.complex(Gq_p.real, Gq_p.imag)
        Gh_Gq_p = G.mH @ Gq_p.view(*q.shape[:2], -1, 1)
        Gh_Gq_p = grad_scale * Gh_Gq_p.view(q.shape)

        # right preconditioning transposed
        Gh_Gq_p = F.pad(Gh_Gq_p, [self.pad_B] * 4, self.pad_mode_B)
        Dq = conv2d(
            input=Gh_Gq_p,
            weight=torch.flip(self.filter_B, dims=(2,3)).conj(),
            bias=None,
            padding=self.args.filter_B.kernel_size // 2,
            padding_mode="zeros",
            groups=self.args.activation.num_filters # depthwise convolution
        )
        Dq = torch.complex(Dq.real, Dq.imag)

        # remove padding and normalize
        Dq = Dq[:, :, self.pad_B:-self.pad_B, self.pad_B:-self.pad_B]
        Dq = Dq / self.args.activation.num_filters
        
        # combine fixed and learned model mismatch
        Dq = torch.cat((Dq, Dq_fixed), 1)

        return Dq


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(args):
    mismatch = Mismatch(args=args.methods.vn.mismatch, layer=1)

    p = torch.randn((5,1,8,8), dtype=torch.complex64)
    q = torch.randn((5,1,41,41), dtype=torch.complex64)
    G = torch.randn((5,1,8*8,41*41), dtype=torch.complex64)
    s_max = torch.randn((5,), dtype=torch.float32)

    Dq = mismatch(q, p, G, s_max)

if __name__ == "__main__":
    main()
