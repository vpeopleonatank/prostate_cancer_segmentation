from typing import Dict
import os

import pytorch_lightning as pl
import numpy as np
import cv2
from omegaconf import DictConfig
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.datasets.mnist_dataset import MnistDataset

def get_tiles(img, tile_size, n_tiles, mask, mode=0):
    t_sz = tile_size
    h, w, c = img.shape
    pad_h = (t_sz - h % t_sz) % t_sz + ((t_sz * mode) // 2)
    pad_w = (t_sz - w % t_sz) % t_sz + ((t_sz * mode) // 2)

    img2 = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    img3 = img2.reshape(img2.shape[0] // t_sz, t_sz, img2.shape[1] // t_sz, t_sz, 3)
    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, t_sz, t_sz, 3)

    mask3 = None
    if mask is not None:
        mask2 = np.pad(
            mask,
            [
                [pad_h // 2, pad_h - pad_h // 2],
                [pad_w // 2, pad_w - pad_w // 2],
                [0, 0],
            ],
            constant_values=0,
        )
        mask3 = mask2.reshape(
            mask2.shape[0] // t_sz, t_sz, mask2.shape[1] // t_sz, t_sz, 3
        )
        mask3 = mask3.transpose(0, 2, 1, 3, 4).reshape(-1, t_sz, t_sz, 3)

    n_tiles_with_info = (
        img3.reshape(img3.shape[0], -1).sum(1) < t_sz ** 2 * 3 * 255
    ).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(
            img3,
            [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]],
            constant_values=255,
        )
        if mask is not None:
            mask3 = np.pad(
                mask3,
                [[0, n_tiles - len(mask3)], [0, 0], [0, 0], [0, 0]],
                constant_values=0,
            )
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    if mask is not None:
        mask3 = mask3[idxs]
    return img3, mask3, n_tiles_with_info >= n_tiles

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

        # image saved in BGR, cv2 imread get image in RGB
        big_img = cv2.imread(file_path)
        big_mask = cv2.imread(file_path_label, cv2.IMREAD_UNCHANGED)


        tiles, masks, _ = get_tiles(
            img=big_img, tile_size=256, n_tiles=36, mask=big_mask, mode=0
        )

        big_img_tensor = transforms.ToTensor()(big_img)
        big_mask_tensor = torch.from_numpy(big_mask[:, :, 0])

        tiles = [transforms.ToTensor()(t) for t in tiles]
        masks = [torch.from_numpy(m[:, :, 0]).long() for m in masks]

        return tiles, masks


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

        if self.cfg.datamodule.load_full_csv:
            masks = os.listdir(train_labels_absolute_path)
            df_masks = pd.Series(masks).to_frame()
            df_masks.columns = ["mask_file_name"]
            df_masks["image_id"] = df_masks.mask_file_name.apply(lambda x: x.split("_")[0])
            df_train = pd.merge(self.train_df, df_masks, on="image_id", how="outer")
            del df_masks

            # replace gleason score
            gleason_replace_dict = {0:0, 1:1, 3:2, 4:3, 5:4}

            def process_gleason(gleason):
                if gleason == 'negative': gs = (1, 1)
                else: gs = tuple(gleason.split('+'))
                return [gleason_replace_dict[int(g)] for g in gs]

            df_train.gleason_score = df_train.gleason_score.apply(process_gleason) 
            df_train['gleason_primary'] = ''
            df_train['gleason_secondary'] = ''

            for idx in range(0, len(df_train.gleason_score)):
                df_train['gleason_primary'][idx] = df_train['gleason_score'][idx][0]
                df_train['gleason_secondary'][idx] = df_train['gleason_score'][idx][1]
                
            df_train = df_train.drop(['gleason_score'], axis=1)
            # remove not exist masks in csv
            df_train.dropna(subset=['mask_file_name'], inplace=True, axis=0)
            X = df_train.drop(['isup_grade'], axis=1)
            Y = df_train['mask_file_name']
        else:
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
