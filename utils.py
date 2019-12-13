import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from torch import Tensor
from skmultilearn.model_selection import iterative_train_test_split


def plot_segmentation(images, masks, figsize=(8, 8)):
    assert len(images) == len(masks)
    N = len(images)

    fig, axes = plt.subplots(N, 2, figsize=figsize)
    axes = axes.reshape(-1, 2)
    to_HWC = lambda ndarray: np.transpose(ndarray, [1, 2, 0])
    for img, mask, ax in zip(images, masks, axes):
        img, mask = to_HWC(img), to_HWC(mask)
        ax[0].imshow(img)
        plot_mask(img, mask, ax[1])
    plt.tight_layout()
    plt.show()


def stratified_multilabel_split(rles, nclasses):
    # Трансформируем метки в one-hot encoding, т.к. иначе эта странная библиотека не работает
    to_one_hot = lambda classes: [cls in classes for cls in range(nclasses)]
    X, Y = np.arange(len(rles))[..., None], [to_one_hot(y) for y in rles]
    train_indices, _, val_indices, _ = iterative_train_test_split(X, np.asarray(Y), test_size=0.25)
    train_indices, val_indices = train_indices[:, 0], val_indices[:, 0]
    return train_indices, val_indices


def compute_weights(rles):
    # Балансировка между классами при обучении
    file_classes = [tuple(x.keys()) for x in rles]
    counts = Counter(file_classes)
    unique_cls_combinations = len(counts.keys())
    weights = [1 / (unique_cls_combinations * counts[cls]) for cls in file_classes]
    assert abs(sum(weights) - 1) < 1e-3
    return weights


def to_one_hot_mask(pred_mask: Tensor):
    with torch.no_grad():
        pred_cls = pred_mask.argmax(axis=1, keepdims=True)
        on_hot_mask = torch.zeros_like(pred_mask)
        on_hot_mask.scatter_(1, pred_cls, 1)
    return on_hot_mask


def parse_data(img_directory, csv_labels=None):
    if csv_labels:
        df = pd.read_csv(csv_labels)
        rles, files = [], []
        for imgfile, group in df.groupby('ImageId'):
            rles.append(dict(zip(group.ClassId, group.EncodedPixels)))
            files.append(os.path.join(img_directory, imgfile))
        return files, rles
    else:
        files = os.listdir(img_directory)
        files = [os.path.join(img_directory, f) for f in files]
        return files


def plot_mask(img: np.ndarray, mask: np.ndarray, ax, alpha=0.5):
    classes = mask.shape[-1] - 1
    cmap = plt.cm.get_cmap('hsv', classes)

    masked = np.zeros_like(img, dtype=np.float32)
    overlap_masks = mask.sum(axis=-1, keepdims=True)
    pixelwise_alpha = alpha / overlap_masks

    # Don't shadow the background
    pixelwise_alpha[(mask[:, :, 0, None] > 0) & (overlap_masks == 1)] = 0

    for cls in range(1, classes):
        color = np.asarray(cmap(cls)[:3]) * 255
        clsmask = mask[:, :, cls] > 0
        masked[clsmask] += color
    masked = (1 - pixelwise_alpha) * img + pixelwise_alpha * masked
    ax.imshow(masked.astype(np.uint8))


def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, height, width):
    rle = rle.split(" ")
    positions = map(int, rle[0::2])
    length = map(int, rle[1::2])
    mask = np.zeros(height * width, dtype=np.uint8)
    for pos, le in zip(positions, length):
        mask[pos:(pos + le)] = 1
    mask = mask.reshape(height, width, order='F')
    return mask
