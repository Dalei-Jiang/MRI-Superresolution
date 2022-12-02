import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
# import pdb

class BasicDataset(Dataset):
    def __init__(self, low_dir: str, high_dir: str, scale: float = 1.0):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(low_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {low_dir}, make sure you put your lows there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img_ndarray, scale, is_high):
        img_ndarray
        if not is_high:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2,0,1))

            img_ndarray = img_ndarray / 255
        else:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2,0,1))
        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return np.load(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        high_name = name[-4:]
        high_file = list(self.high_dir.glob(high_name + '.*'))  # here
        low_file = list(self.low_dir.glob(name + '.*'))
        assert len(low_file) == 1, f'Either no low or multiple lows found for the ID {name}: {low_file}'
        assert len(high_file) == 1, f'Either no high or multiple highs found for the ID {name}: {high_file}'
        high = self.load(high_file[0])
        low = self.load(low_file[0])

        low = self.preprocess(low, self.scale, is_high=False)
        high = self.preprocess(high, self.scale, is_high=True)
        return {
            'low': torch.as_tensor(low.copy()).float().contiguous(),
            'high': torch.as_tensor(high.copy()).long().contiguous(),
            'idx': str(low_file[0])
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, low_dir, high_dir, scale=1):
        super().__init__(low_dir, high_dir, scale)
