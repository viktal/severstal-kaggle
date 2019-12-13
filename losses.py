import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


def dice_loss(predict: Tensor, target: Tensor, smooth=1e-3):
    '''
    :param predict: B,C,H,W: probabilities
    :param target: B,C,H,W: one-hot encoded ground true labels
    :return: tensor: B, C - loss value
    '''
    B, C = predict.shape[:2]
    predict, target = predict.view(B, C, -1), target.view(B, C, -1)
    intersection = (predict * target).sum(dim=-1)  # B,C

    predict_area, target_area = predict.sum(dim=-1), target.sum(dim=-1)  # B,C
    loss = 1 - (2.0 * intersection + smooth) / (predict_area + target_area + smooth)
    return loss


def focal_loss(predict: Tensor, target: Tensor, gamma=2):
    batch, num_classes = predict.shape[:2]

    probs = predict.view(batch, num_classes, -1)  # N,C,H*W
    probs = probs.transpose(1, 2)    # N,H*W,C
    probs = probs.contiguous().view(-1, num_classes)   # N,H*W,C => N*H*W,C

    target = target.argmax(dim=1).view(-1, 1).type(torch.int64)  # Indices must be long
    probs = probs.gather(dim=1, index=target) + 1e-6  # 1e-6 for numerical stability

    loss = -1 * torch.pow((1 - probs), gamma) * torch.log(probs)
    loss = loss.view(batch, 1, *predict.shape[-2:])
    return loss


def iou(y_pred, y_true, smooth=1e-3):
    """Intersection over union"""
    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (y_pred * y_true).sum(axis=(2, 3))
    total = (y_pred + y_true).sum(axis=(2, 3))
    union = total - intersection
    iou_ = (intersection + smooth) / (union + smooth)
    return 1 - iou_.mean(axis=1).mean()


def cross_entropy(y_pred: Tensor, y_true: Tensor, weights: Tensor = None, smooth=1e-3):
    batch, classes = y_pred.shape[:2]
    y_pred, y_true = y_pred.reshape(batch, classes, -1), y_true.reshape(batch, classes, -1)
    loss = -(y_true * torch.log(y_pred + smooth))
    if weights:
        assert weights.dim() == 1 and len(weights) == classes
        loss = loss * weights[None, ..., None]
    return loss.sum(axis=1).mean()

# And more https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#BCE-Dice-Loss ...
