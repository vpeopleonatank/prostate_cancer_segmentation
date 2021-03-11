from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.utils.technical_utils import load_obj
from src.losses.losses import CrossEntropy2D
from src.metrics.iou import Iou


class LitSemanticSegmentation(pl.LightningModule):
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
        super(LitSemanticSegmentation, self).__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams
        self.model = load_obj(cfg.model.class_name)(**self.cfg.model.params)

        self.loss = load_obj(cfg.loss.class_name)(**self.cfg.loss.params)
        if not cfg.metric.params:
            self.metric = load_obj(cfg.metric.class_name)()
        else:
            self.metric = load_obj(cfg.metric.class_name)(**cfg.metric.params)

        self.iou_train = Iou(num_classes=self.cfg.training.n_classes)
        self.iou_val = Iou(num_classes=self.cfg.training.n_classes)
        self.iou_test = Iou(num_classes=self.cfg.training.n_classes)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        # if 'decoder_lr' in self.cfg.optimizer.params.keys():
        #     params = [
        #         {'params': self.model.decoder.parameters(), 'lr': self.cfg.optimizer.params.lr},
        #         {'params': self.model.encoder.parameters(), 'lr': self.cfg.optimizer.params.decoder_lr},
        #     ]
        #     optimizer = load_obj(self.cfg.optimizer.class_name)(params)

        # else:
        #     optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        # scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        # return (
        #     [optimizer],
        #     [{'scheduler': scheduler, 'interval': self.cfg.scheduler.step, 'monitor': self.cfg.scheduler.monitor}],
        # )

        optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        ret_opt = {"optimizer": optimizer}

        if self.cfg.scheduler.class_name != 'None':
            sch = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

            if sch is not None:
                scheduler = {
                    "scheduler": sch,  # The LR scheduler instance (required)
                    "interval": "epoch",  # The unit of the scheduler's step size
                    "frequency": 1,  # The frequency of the scheduler
                    "reduce_on_plateau": False,  # For ReduceLROnPlateau scheduler
                    "monitor": "Val/mIoU",  # Metric for ReduceLROnPlateau to monitor
                    "strict": True,  # Whether to crash the training if `monitor` is not found
                    "name": None,  # Custom name for LearningRateMonitor to use
                }

                ret_opt.update({"lr_scheduler": scheduler})

        return ret_opt

    def training_step(self, batch, *args, **kwargs):  # type: ignore

        """Defines the train loop. It is independent of forward().
        Don’t use any cuda or .to(device) calls in the code. PL will move the tensors to the correct device.
        """
        inputs, labels = batch
        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)

        # Calculate Loss
        loss = self.loss(outputs, labels)
        logs = {'train_loss': loss}

        """Log the value on GPU0 per step. Also log average of all steps at epoch_end."""
        # self.log("Train/loss", loss, on_step=True, on_epoch=True)
        """Log the avg. value across all GPUs per step. Also log average of all steps at epoch_end.
        Alternately, you can use the ops 'sum' or 'avg'.
        Using sync_dist is efficient. It adds extremely minor overhead for scalar values.
        """
        # self.log("Train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, sync_dist_op="avg")

        # Calculate Metrics
        # self.iou_train(predictions, labels)
        return {
            'loss': loss,
            'progress_bar': logs,
        }

    # def training_epoch_end(self, outputs):
    #     metrics_avg = self.iou_train.compute()
    #     self.log("Train/mIoU", metrics_avg.miou)
    #     self.iou_train.reset()
        # logs = {'train_iou': metrics_avg.miou}
        # return {
        #     'progress_bar': logs,
        # }

    def validation_step(self, batch, *args, **kwargs):  # type: ignore

        inputs, labels = batch
        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)

        # Calculate Loss
        loss = self.loss(outputs, labels)
        self.log("Val/loss", loss)

        # Calculate Metrics
        self.iou_val(predictions, labels)

        logs = {'valid_loss': loss}

        return {
            'val_loss': loss,
            'progress_bar': logs,
        }

    def validation_epoch_end(self, outputs):
        # Compute and log metrics across epoch
        metrics_avg = self.iou_val.compute()
        self.log("Val/mIoU", metrics_avg.miou)
        self.iou_val.reset()
        logs = {'val_iou': metrics_avg.miou}
        return {
            'progress_bar': logs,
        }
