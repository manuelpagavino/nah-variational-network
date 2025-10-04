
import torch

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def rmse(estimate, target, reduce=True, percent=True):
    """Root Mean Squared Error (RMSE) according to Eq. (36)."""
    # vectorize
    target = target.flatten(2,3)
    estimate = estimate.flatten(2,3)

    num = torch.linalg.norm(target - estimate, 2, -1)
    denom = torch.linalg.norm(target, 2, -1)
    rmse = (num / denom)

    if percent:
        rmse = rmse * 100

    if reduce:
        return torch.mean(rmse)
    else:
        return rmse

def ncc(estimate, target, zero_mean=False, reduce=True, percent=True):
    """Normalized Cross Correlation (NCC) according to Eq. (33)."""
    # vectorize inputs
    estimate = estimate.flatten(2,3)
    target = target.flatten(2,3)
    
    # force zero mean
    if zero_mean:
        estimate = estimate - estimate.mean((1,2), keepdim=True)
        target = target - target.mean((1,2), keepdim=True)
    
    # compute norms for denom
    target_norm = torch.linalg.norm(target, 2, -1, keepdim=True)
    estimate_norm = torch.linalg.norm(estimate, 2, -1, keepdim=True)
    
    # compute (zero-mean) normalized cross correlation
    ncc = (target @ estimate.mT).abs() / (target_norm * estimate_norm)
    
    if percent:
        ncc = ncc * 100

    if reduce:
        return torch.mean(ncc)
    else:
        return ncc
    
def ssim(estimate, target, data_range=1.0, reduce=True, percent=True):
    """Structual Similarity Index Measure (SSIM)."""
    ssim = StructuralSimilarityIndexMeasure(
        data_range=data_range,
        reduction="elementwise_mean" if reduce else None
        )
    ssim = ssim(torch.abs(estimate), torch.abs(target))
    if percent:
        ssim = ssim * 100
    return ssim

def psnr(estimate, target, data_range=1.0, reduce=True):
    """Peak Signal-to-Noise Ratio (PSNR)."""
    psnr = PeakSignalNoiseRatio(
        data_range=data_range,
        reduction="elementwise_mean" if reduce else None
        )
    psnr = psnr(torch.abs(estimate), torch.abs(target))
    return psnr