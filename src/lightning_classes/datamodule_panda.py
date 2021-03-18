from typing import Dict
import os

import pytorch_lightning as pl
import cv2
from omegaconf import DictConfig
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.datasets.mnist_dataset import MnistDataset


class PANDADataset(Dataset):
    def __init__(
        self,
        df,
        labels,
        images_in_path="",
        labels_in_path="",
        transform=None,
        rand=False,
    ):
        self.df = df
        self.labels = labels
        self.transform = transform
        self.rand = rand

        self.gleason_replace_dict = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4}

        self.images_in_path = images_in_path
        self.labels_in_path = labels_in_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_name_label = self.labels.values[idx]
        file_path = f'{self.images_in_path}{file_name}.png'
        file_path_label = f'{self.labels_in_path}{file_name_label}'

        big_img = cv2.imread(file_path)
        big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
        big_mask = cv2.imread(file_path_label, cv2.IMREAD_UNCHANGED)

        big_img_tensor = transforms.ToTensor()(big_img)
        big_mask_tensor = torch.from_numpy(big_mask[:, :, 0])

        return big_img_tensor, big_mask_tensor.long()


class PANDADataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, hparams: Dict[str, float], data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):

        self.BASE_PATH = self.cfg.datamodule.base_image_path

        self.train_df = pd.read_csv(self.BASE_PATH + self.cfg.datamodule.train_csv)
        self.test_df = pd.read_csv(self.BASE_PATH + self.cfg.datamodule.test_csv)
        self.train_image_path = self.cfg.datamodule.train_image_path
        self.train_labels_path = self.cfg.datamodule.train_labels_path
        train_images_absolute_path = self.BASE_PATH + self.train_image_path
        train_labels_absolute_path = self.BASE_PATH + self.train_labels_path

        X = self.train_df.drop(['isup_grade'], axis=1)
        Y = self.train_df['mask_file_name']
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X, Y, test_size=self.cfg.datamodule.valid_size, random_state=1234
        )

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.panda_train = PANDADataset(
                self.X_train, self.y_train, train_images_absolute_path, train_labels_absolute_path
            )
            self.panda_val = PANDADataset(
                self.X_valid, self.y_valid, train_images_absolute_path, train_labels_absolute_path
            )

        # if stage == 'test' or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.panda_train, batch_size=self.cfg.datamodule.batch_size,
                num_workers=self.cfg.datamodule.num_workers, pin_memory=self.cfg.datamodule.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.panda_val, batch_size=self.cfg.datamodule.batch_size,
                num_workers=self.cfg.datamodule.num_workers, pin_memory=self.cfg.datamodule.pin_memory, drop_last=True)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=32)
