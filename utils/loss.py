import torch
import logging

import numpy as np

from torch import nn

from utils.metrics import ncc

logger = logging.getLogger(__name__)
class CompositeLoss(nn.Module):
    """The composite loss function according to Eq. (34)."""

    def __init__(self, args):
        super(CompositeLoss, self).__init__()
        self.args = args
        self.show_loss = args.show_loss

        # init loss fn with kwargs from conf
        klass = getattr(nn, self.args.klass)
        self.loss_fn = klass(**self.args.kwargs)

        # loss factors
        self.loss_abs_weight = args.loss_abs_weight
        self.loss_real_weight = args.loss_real_weight
        self.loss_imag_weight = args.loss_imag_weight
        self.loss_ncc_weight = args.loss_ncc_weight

    def ncc_loss_fn(self, estimate, target, zero_mean=True):
        """Use Normalized Cross Correlation (NCC) of Eq. (33) as loss via (1-NCC)."""
        # compute (zero-mean) normalized cross correlation    
        ncc_metric = ncc(
            estimate, target, zero_mean, reduce=False, percent=False
            )
        
        # compute reduced loss
        ncc_loss = (1 - ncc_metric).mean()
        return ncc_loss

    def forward(self, estimate, target):
        """Compute composite loss function according to Eq. (34).
        
        Parameters
        ----------
        estimate : Tensor, (B, C, H, W)
            Prediction.
        target : Tensor, (B, C, H, W)
            Target reference.

        Returns
        -------
        loss : float
            Single loss value.
        """
        # compute magnitudes
        estimate_abs = estimate.abs()
        target_abs = target.abs()

        # calculate and weigh sublosses
        loss, loss_abs, loss_real, loss_imag, loss_ncc = torch.zeros(5)
        if self.loss_abs_weight:
            loss_abs = self.loss_fn(estimate_abs, target_abs) 
            loss_abs = loss_abs * self.loss_abs_weight
        if self.loss_real_weight:
            loss_real = self.loss_fn(estimate.real, target.real)
            loss_real = loss_real * self.loss_real_weight
        if self.loss_imag_weight:
            loss_imag = self.loss_fn(estimate.imag, target.imag)
            loss_imag = loss_imag * self.loss_imag_weight
        if self.loss_ncc_weight:
            loss_ncc = self.ncc_loss_fn(estimate_abs, target_abs)
            loss_ncc = loss_ncc * self.loss_ncc_weight
        
        # sum up weighted sublosses
        loss = loss_abs + loss_real + loss_imag + loss_ncc

        # optionally display losses in each iteration to ease selection of weighting factors
        if self.show_loss:
            loss_info = {
                "loss": loss,
                "loss_abs": loss_abs,
                "loss_real": loss_real,
                "loss_imag": loss_imag,
                "loss_ncc": loss_ncc,
            }
            loss_info = " ".join(f"{k}={v:.6f} \n" for k, v in loss_info.items())
            logger.info(loss_info)

        return loss

def atan2_stable(y, x):
    """Stable computation of gradients via custom optimized torch atan2 function.

    NOTE: Standard atan/atan2 yields nan gradients for edge case x == y == 0.
          For this edge case, the function provides angle and input gradients of zero!

    Parameters
    ----------
    y : Tensor
        Imaginary part.
    x : Tensor
        Real part.
    """
    # boolean masks for (strictly) positive, negative, and zero values of x and y
    y_strictly_positive = y > 0
    y_positive = y >= 0
    y_strictly_negative = y < 0
    x_strictly_negative = x < 0
    x_zero = x == 0 
    y_zero = y == 0

    # constants
    pi = np.pi
    pi_half = 0.5 * pi
    eps = 1e-7

    # NOTE: regularize not only zero denominators, but also the ones of very small values 
    #       see https://discuss.pytorch.org/t/how-to-avoid-nan-output-from-atan2-during-backward-pass/176890/6
    #       This avoids bfloat16 nans due to overflow/underflow rounding errors!
    # eps = eps * x_zero # only regularize zero denominators
    sign = x.sign() # -1 for negative, +1 for positive, 0 for zero
    sign[sign == 0] = 1 # make sure not to use 0, but 1 in case of zero input
    x_eps = (x.abs() < eps) 
    x_eps = (eps - x.abs()) * sign * x_eps # clamp to +-eps with correct sign
    eps = x_eps

    # Quadrant I & IV
    angle = torch.atan(y / (x + eps))

    # Quadrant II
    angle += (y_positive & x_strictly_negative) * pi
    
    # Quadrant III
    angle -= (y_strictly_negative & x_strictly_negative) * pi
    
    # +90°
    mask = (y_strictly_positive & x_zero)
    angle *= ~mask # set mask indizes to zero
    angle += mask * pi_half

    # -90°
    mask = (y_strictly_negative & x_zero)
    angle *= ~mask # set mask indizes to zero
    angle -= mask * pi_half
    
    # Stabilized computation of otherwise invalid edge case x == y == 0
    # Forces zero output while still providing gradients that are not NaN!
    angle *= ~(x_zero & y_zero) # set mask indizes to zero

    return angle
