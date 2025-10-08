

import torch
import hydra

from torch import nn

from methods.vn.regularizer import Regularizer
from methods.vn.mismatch import Mismatch
from methods.vn.momentum import Momentum
from methods.vn.initializer import Initializer

class VariationalNetwork(nn.Module):
    """The T-layer Variational Network (VN) according to Eq. (18)."""

    def __init__(self, args):
        super(VariationalNetwork, self).__init__()
        self.args = args
        
        # data-driven cont'd TSVD initializer
        self.initializer = Initializer(args.initializer)
        
        # init all layers
        self.layers = nn.ModuleList(
            VariationalNetworkLayer(args=args, layer=layer) 
            for layer in range(self.args.T)
            )
                
    def forward(self, p, G, q0, snr=30, return_intermediate=False):
        """Apply the T-layer VN according to Eq. (18).
        
        Parameters
        ----------
        p : Tensor, (B, 1, M_y, M_x)
            Sound pressure measurements.
        G : Tensor, (B, 1, M, N)
            Propagation matrix.
        q0 : Tensor, (B, 1, N_y, N_x)
            Initial equivalent source strengths used for shape extraction. 
            Will be overwritten by the intitializer.
        snr : Tensor, (B,)
            Estimated signal-to-noise ratio of each measurement p.
        return_intermediate : bool
            if return_intermediate:
                Return equivalent source strengths estimates of all VN layers (final + intermediate layers).
            else:
                Return only equivalent source strengths estimate of the final VN layer. (default)

        Returns
        -------
        q : Tensor, (B, 1, N_y, N_x)
            if return_intermediate:
                Equivalent source strengths estimates of all VN layers (final + intermediate layers).
            else:
                Equivalent source strengths estimate of the final VN layer. (default)
        """
        # store intermediate outputs
        q_t = []

        # convert single value snr to batched tensor
        if isinstance(snr, (int, float)):
            snr = torch.empty(p.shape[0]).fill_(snr)

        # initialize equivalent sources
        q, s_max = self.initializer(p, G, q0, snr)
        q_t += [q] # store initializer output

        # normalize
        q_max = torch.amax(q.abs(), dim=(2, 3), keepdim=True)
        q = q / q_max
        p = p / q_max

        # execute all layers
        grad_cache = None
        for i, layer in enumerate(self.layers):
            q, grad_cache = layer(q, p, G, grad_cache, s_max)
            if i < self.args.T - 1:
                q_t += [q] # store intermediate layer outputs

        # denormalize
        # NOTE: power compression improves magnitude approximation of point sources
        q = q * torch.where(q_max <= 1, q_max ** 0.5, q_max)
        q_t += [q] # store final layer output

        if return_intermediate:
            # list of intermediate outputs starting from the initializer layer
            assert len(q_t) == self.args.T + 1 
            return q_t
        else:
            return q
    
class VariationalNetworkLayer(nn.Module):
    """A single Variational Network (VN) layer t according to Eq. (18)."""

    def __init__(self, args, layer):
        super(VariationalNetworkLayer, self).__init__()
        self.args = args
        self.layer = layer

        # init gradient networks
        self.regularizer = Regularizer(args.regularizer, layer)
        self.mismatch = Mismatch(args.mismatch, layer)
        self.momentum = Momentum(args.momentum, layer)


    def forward(self, q, p, G, grad_cache, s_max):
        """Apply a single VN layer t according to Eq. (18).
        
        Parameters
        ----------
        p : Tensor, (B, 1, M_y, M_x)
            Sound pressure measurements.
        q : Tensor, (B, 1, N_y, N_x)
            Eequivalent source strengths estimate of previous VN layer t-1.
        G : Tensor, (B, 1, M, N)
            Propagation matrix.
        grad_cache : list 
            if momentum.store_all_grads:
                List containing the regularizer and model mismatch gradients of all past layers l
                with 0 <= l <= t-1, excluding the current layer t.
            else:
                List containing the regularizer and model mismatch gradients of the previous layer t-1.
        s_max : Tensor, (B,)
            Largest singular value of the propagation matrix G linked each measurement p.

        Returns
        -------
        q : Tensor, (B, 1, N_y, N_x)
            Equivalent source strengths estimate of the current VN layer t.
        """
        # compute regularizer gradient
        Rq = self.regularizer(q)

        # compute mismatch gradient
        Dq = self.mismatch(q, p, G, s_max)

        # apply gradient momentum
        grad, grad_cache = self.momentum(Rq, Dq, grad_cache)

        # perform gradient step to update equivalent source estimate
        q = q - grad

        return q, grad_cache

    

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(args):
    vn = VariationalNetwork(args=args.methods.vn)

    p = torch.randn((5,1,8,8), dtype=torch.complex64)
    q0 = torch.randn((5,1,41,41), dtype=torch.complex64)
    G = torch.randn((5,1,8*8,41*41), dtype=torch.complex64)

    q = vn(p, G, q0)

if __name__ == "__main__":
    main()
