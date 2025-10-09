import torch
import hydra

import torch.nn.functional as F

from torch import nn
from cplxmodule.cplx import conv2d

from utils.params import create_filter
from utils.activations import CplxGaussianRBF
from utils.norm import init_instance_norm

class Regularizer(nn.Module):
    """Regularizer network according to Eq. (20)."""

    def __init__(self, args, layer):
        super(Regularizer, self).__init__()
        self.args = args
        self.layer = layer

        # pad
        self.pad = args.filter_K.pad
        self.pad_mode = args.filter_K.pad_mode

        # filters
        self.filter_K = torch.nn.Parameter(create_filter(args.filter_K), requires_grad=True)
        self.filter_L = torch.nn.Parameter(create_filter(args.filter_L), requires_grad=True)

        # activation
        self.activation = CplxGaussianRBF(args.activation)

        # instance normalization
        self.norm = init_instance_norm(args.activation.num_filters)

    def forward(self, q):
        """Apply regularizer network to estimate the constraint gradients according to Eq. (20).
        
        Parameters
        ----------
        q : Tensor, (B, 1, N_y, N_x)
            Equivalent source strengths.

        Returns
        -------
        Rq : Tensor, (B, N_R, N_y, N_x)
            A set of N_R constraint gradient estimates according to Eq. (20).
        """
        # filters / domain transformations
        q = F.pad(q, [self.pad] * 4, self.pad_mode)
        q = conv2d(
            input=q,
            weight=self.filter_L.to(q.device),
            bias=None,
            padding=self.args.filter_L.kernel_size // 2,
            padding_mode="zeros",
        )
        
        q = conv2d(
            input=q,
            weight=self.filter_K.to(q.device), 
            bias=None,
            padding=self.args.filter_K.kernel_size // 2,
            padding_mode="zeros",
        )

        # instance normalization
        q = self.norm(q)

        # activation / penalty function
        q = torch.complex(q.real, q.imag) # convert Cplx to torch.complex
        q = self.activation(q)

        # transposed filters / domain transformations
        q = conv2d(
            input=q,
            weight=torch.flip(self.filter_K, dims=(2,3)).conj().to(q.device), 
            bias=None,
            padding=self.args.filter_K.kernel_size // 2,
            padding_mode="zeros",
        )

        Rq = conv2d(
            input=q,
            weight=torch.flip(self.filter_L, dims=(2,3)).conj().to(q.device),
            bias=None,
            padding=self.args.filter_L.kernel_size // 2, 
            padding_mode="zeros",
            groups=self.args.activation.num_filters, # depthwise convolution
        )
        Rq = torch.complex(Rq.real, Rq.imag)

        # remove padding and normalize
        Rq = Rq[:, :, self.pad:-self.pad, self.pad:-self.pad]
        Rq /= self.args.activation.num_filters
        
        return Rq

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(args):
    regularizer = Regularizer(args=args.methods.vn.regularizer, layer=1)
    q = torch.randn((5,1,41,41), dtype=torch.complex64)
    Rq = regularizer(q)

if __name__ == "__main__":
    main()
