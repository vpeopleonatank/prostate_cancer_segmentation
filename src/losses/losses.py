from typing import Tuple

import torch
import torch.functional as F
from torch import nn


class DenseCrossEntropy(nn.Module):
    # Taken from: https://www.kaggle.com/pestipeti/plant-pathology-2020-pytorch
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()


class CutMixLoss:
    # https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/losses/cutmix.py
    def __init__(self, reduction: str = 'mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
        self, predictions: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float], train: bool = True
    ) -> torch.Tensor:
        if train:
            targets1, targets2, lam = targets
            loss = lam * self.criterion(predictions, targets1) + (1 - lam) * self.criterion(predictions, targets2)
        else:
            loss = self.criterion(predictions, targets)
        return loss


class MixupLoss:
    # https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/losses/mixup.py
    def __init__(self, reduction: str = 'mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
        self, predictions: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float], train: bool = True
    ) -> torch.Tensor:
        if train:
            targets1, targets2, lam = targets
            loss = lam * self.criterion(predictions, targets1) + (1 - lam) * self.criterion(predictions, targets2)
        else:
            loss = self.criterion(predictions, targets)
        return loss


class CrossEntropy2D(nn.CrossEntropyLoss):
    """Use the torch.nn.CrossEntropyLoss loss to calculate mean loss per image or per pixel.
    Deeplab models calculate mean loss per image.

    Inputs:
        - inputs (Tensor): Raw output of network (without softmax applied).
        - targets (Tensor): Ground truth, containing a int class index in the range :math:`[0, C-1]` as the
          `target` for each pixel.

    Shape and dtype:
        - Input: [B, C, H, W], where C = num_classes. dtype=float16/32
        - Target: [B, H, W]. dtype=int32/64
        - Output: scalar.

    Args:
        loss_per_image (bool, optional):
        ignore_index (int, optional): Defaults to 255. The pixels with this labels do not contribute to loss

    References:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
    """

    def __init__(self, loss_per_image: bool = True, ignore_index: int = 255):
        if loss_per_image:
            reduction = "sum"
        else:
            reduction = "mean"

        super().__init__(reduction=reduction, ignore_index=ignore_index)
        self.loss_per_image = loss_per_image

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = F.cross_entropy(
            inputs, targets, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )

        if self.loss_per_image:
            batch_size = inputs.shape[0]
            loss = loss / batch_size

        return loss
