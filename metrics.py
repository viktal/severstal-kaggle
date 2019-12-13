import torch
from torch import Tensor


def dice(pred_mask: Tensor, true_mask: Tensor):
    """Средний dice коэффициент для каждого изображения в батче."""
    assert pred_mask.shape == true_mask.shape and \
           pred_mask.dim() == true_mask.dim() == 4, "[B, C, H, W] tensors expected"
    with torch.no_grad():
        intersection = (pred_mask * true_mask).sum(dim=[2, 3])
        union = pred_mask.sum(dim=[2, 3]) + true_mask.sum(dim=[2, 3])
        result = torch.where(union == 0, torch.ones_like(intersection), 2 * intersection / union)
    return result
