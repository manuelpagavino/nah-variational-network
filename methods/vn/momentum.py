import torch
import hydra

import torch.nn as nn

from cplxmodule.cplx import conv2d

from utils.params import create_filter

class Momentum(nn.Module):
    """Gradient momentum network according to Eq. (21)."""

    def __init__(self, args, layer):
        super(Momentum, self).__init__()
        self.args = args
        self.layer = layer
        self.store_all_grads = args.store_all_grads

        # pad
        self.pad = args.filter.pad
        self.pad_mode = args.filter.pad_mode

        # filters
        self.filter = create_filter(
            args.filter, 
            multiplier=self.layer if self.store_all_grads else 1
            )

    def forward(self, Rq, Dq, grad_cache):
        """Apply gradient momentum network to re-weigh and sum up past and current gradients according to Eq. (21).
        
        Parameters
        ----------
        Rq : Tensor, (B, N_R, N_y, N_x)
            The regularizer network gradients of the current layer t.
        Dq : Tensor, (B, N_D, M_y, M_x)
            The model mismatch network gradients of the current layer t.
        grad_cache : list 
            if store_all_grads:
                List containing the regularizer and model mismatch gradients of all past layers l
                with 0 <= l <= t-1, excluding the current layer t.
            else:
                List containing the regularizer and model mismatch gradients of the previous layer t-1.

        Returns
        -------
        grad : Tensor, (B, 1, N_y, N_x)
            The re-weighed sum of past and current gradients according to Eq. (21).
        grad_cache : list 
            if store_all_grads:
                List containing the original regularizer and model mismatch gradients of past and current 
                layers l with 0 <= l <= t, including the current layer t.
            else:
                List containing the original regularizer and model mismatch gradients of the current layer t.
        """
        if self.layer:            
            # re-weigh past gradients
            grad_cache_weighted = torch.cat(grad_cache, 1)
            grad_cache_weighted = conv2d(
                input=grad_cache_weighted,
                weight=self.filter,
                bias=None,
                padding=self.pad,
                padding_mode=self.pad_mode,
                groups=grad_cache_weighted.shape[1], # depthwise convolution
            )
            grad_cache_weighted = torch.complex(grad_cache_weighted.real, grad_cache_weighted.imag)

            # sum up re-weighted gradients
            grad = torch.cat((Rq, Dq, grad_cache_weighted), 1)
            grad = torch.sum(grad, 1, keepdim=True)
            
            # store gradients in cache
            if self.store_all_grads:
                grad_cache = [Rq, Dq] + grad_cache
            else:
                grad_cache = [Rq, Dq]
        else:
            # store gradients in cache
            grad_cache = [Rq, Dq]

            # sum up gradients
            grad = torch.cat(grad_cache, 1)
            grad = torch.sum(grad, 1, keepdim=True)

        return grad, grad_cache

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(args):
    args.methods.vn.momentum.filter.out_channels = 32 + 8
    momentum = Momentum(args=args.methods.vn.momentum, layer=1)

    Rq = torch.randn((5,32,41,41), dtype=torch.complex64)
    Dq = torch.randn((5,8,41,41), dtype=torch.complex64)

    grad, grad_cache = momentum(Rq, Dq, [Rq, Dq])

if __name__ == "__main__":
    main()
