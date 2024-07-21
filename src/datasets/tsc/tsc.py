import os
import torch
import zipfile
import numpy as np
import pytorch_lightning as pl

from typing import Optional
from hydra.utils import to_absolute_path
from aeon.datasets import load_classification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import datasets_utils


class TimeSeriesClassification(pl.LightningDataModule):

    def __init__(self,
                 download_dir: str,
                 download_url: str,
                 file_name: str,
                 batch_size: int,
                 num_workers: int,
                 dataset_name: str
                 ):
        """

        :param download_dir: the directory to save the data
        :param download_url: from where to download the data
        :param file_name: the name of the downloaded file
        :param batch_size:
        :param num_workers:
        """
        super().__init__()
        self.train_size = None
        self.save_hyperparameters(logger=False)
        self.train_dataset = None
        self.test_dataset = None
        self.output_size = None
        self.time_series_size = None
        self.channels = None
        self.file_path = None
        self.shape = None
        self.batch_size = batch_size
        self.download_dir = download_dir
        self.file_name = file_name
        self.dataset_name = dataset_name

    @staticmethod
    def _transform_labels(y_train, y_test):
        """
        Transform label to min equal zero and continuous
        For example if we have [1,3,4] --->  [0,1,2]
        """
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # re-split the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test

    def _load_data(self):
        x_train, y_train = load_classification(extract_path=os.path.join(self.file_path, self.dataset_name),
                                               name=self.dataset_name,
                                               return_metadata=False, split='train')
        x_test, y_test = load_classification(extract_path=os.path.join(self.file_path, self.dataset_name),
                                             name=self.dataset_name,
                                             return_metadata=False, split='test')
        y_train, y_test = self._transform_labels(y_train, y_test)
        return x_train, x_test, y_train, y_test

    def prepare_data(self):
        if not os.path.exists(to_absolute_path(self.hparams.download_dir)):
            os.makedirs(to_absolute_path(self.hparams.download_dir))
        self.file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        if not os.path.exists(self.file_path):
            datasets_utils.download_file(self.hparams.download_url, self.file_path)

            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                zip_ref.extractall(self.hparams.download_dir)
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            self.file_path = os.path.join(self.download_dir, zip_ref.filelist[0].filename)
        x_train, _, y_train, y_test = self._load_data()
        self.time_series_size = x_train.shape[-1]
        self.shape = x_train.shape
        self.train_size = x_train.shape[0]
        self.channels = x_train.shape[-2] if len(x_train.shape) > 2 else 1
        self.output_size = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    def setup(self, stage: Optional[str] = None):
        x_train, x_test, y_train, y_test = self._load_data()
        y_train, y_test = self._transform_labels(y_train, y_test)
        is_univariate = self.channels == 1
        if is_univariate:
            x_train = x_train.transpose(0, 2, 1)
            x_test = x_test.transpose(0, 2, 1)
        else:
            ss = StandardScaler()
            x_train = x_train.transpose(0, 2, 1).reshape(-1, self.channels)
            x_test = x_test.transpose(0, 2, 1).reshape(-1, self.channels)
            x_train = ss.fit_transform(x_train).reshape(-1, self.shape[-1], self.channels)
            x_test = ss.transform(x_test).reshape(-1, self.shape[-1], self.channels)

        train_tensors = [torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)]
        test_tensors = [torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)]

        self.train_dataset = TensorDataset(*train_tensors)
        self.test_dataset = TensorDataset(*test_tensors)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=False)


if __name__ == '__main__':
    uer = TimeSeriesClassification(download_dir=r'C:\Developments\GP-VIB\data\tsc\uea',
                                   download_url='https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip',
                                   file_name='Multivariate2018_ts.zip',
                                   batch_size=64,
                                   num_workers=1,
                                   dataset_name='BasicMotions')

    uer.prepare_data()
    uer.setup()

    ucr = TimeSeriesClassification(download_dir=r'C:\Developments\GP-VIB\data\tsc\ucr',
                                   download_url='https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip',
                                   file_name='Univariate2018_ts.zip',
                                   batch_size=64,
                                   num_workers=1,
                                   dataset_name='Crop')

    ucr.prepare_data()
    ucr.setup()
