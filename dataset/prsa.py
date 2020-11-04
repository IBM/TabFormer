import torch
from torch.utils.data import DataLoader, Dataset
import glob
import pandas as pd
import numpy as np
from dataset.vocab import Vocabulary
from sklearn.preprocessing import MinMaxScaler
import tqdm
from os import path
import logging

logger = logging.getLogger(__name__)
log = logger


class PRSADataset(Dataset):
    def __init__(self,
                 data_root: str = "./data/prsa/",
                 seq_len=10,
                 stride=5,
                 nbins=50,
                 vocab_dir="output",
                 mlm=True,
                 return_labels=False,
                 use_station=False,
                 transform_date=True,
                 flatten=False):

        self.stride = stride
        self.seq_len = seq_len
        self.data_root = data_root
        self.nbins = nbins
        self.vocab_dir = vocab_dir

        self.mlm = mlm
        self.return_labels = return_labels
        self.use_station = use_station
        self.transform_date = transform_date
        self.flatten = flatten

        self.vocab = Vocabulary()
        self.encoding_fn = {}
        self.target_cols = ['PM2.5', 'PM10']

        self.setup()

    def __getitem__(self, index):
        if self.flatten:
            return_data = torch.tensor(self.samples[index], dtype=torch.long)
        else:
            return_data = torch.tensor(self.samples[index], dtype=torch.long).reshape(self.seq_len, -1)

        if self.return_labels:
            target = self.targets[index]
            return_data = return_data, torch.tensor(target, dtype=torch.float32)

        return return_data

    def __len__(self):
        return len(self.samples)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.nbins, 1 / self.nbins)
        bin_edges = np.quantile(data, qtls, axis=0)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.nbins) - 1
        return quant_inputs

    @staticmethod
    def time_fit_transform(column):
        mfit = MinMaxScaler()
        mfit.fit(column)
        return mfit, mfit.transform(column)

    @staticmethod
    def timeEncoder(X):
        d = pd.to_datetime(dict(year=X['year'], month=X['month'], day=X['day'], hour=X['hour'])).astype(int)
        return pd.DataFrame(d)

    def setup(self):
        data = self.read_data(self.data_root)

        '''
        year 	month 	day 	hour 	PM2.5 	PM10 	SO2 	NO2 	
        CO 	O3 	TEMP 	PRES 	DEWP 	RAIN 	wd 	WSPM 	station
        '''

        cols_for_bins = []
        if self.transform_date:
            cols_for_bins += ['timestamp']

            data_cols = ['year', 'month', 'day', 'hour']
            timestamp = self.timeEncoder(data[data_cols])
            timestamp_fit, timestamp = self.time_fit_transform(timestamp)
            self.encoding_fn['timestamp'] = timestamp_fit
            data['timestamp'] = timestamp

        cols_for_bins += ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        for col in cols_for_bins:
            col_data = np.array(data[col])
            bin_edges, bin_centers, bin_widths = self._quantization_binning(col_data)
            data[col] = self._quantize(col_data, bin_edges)
            self.encoding_fn[col] = [bin_edges, bin_centers, bin_widths]

        final_cols = cols_for_bins + ['wd', 'station', 'PM2.5', 'PM10']

        self.data = data[final_cols]
        self.init_vocab()
        self.prepare_samples()
        self.save_vocab(self.vocab_dir)

    def prepare_samples(self):
        self.samples, self.targets = [], []
        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        groups = self.data.groupby('station')
        for group in tqdm.tqdm(groups):
            station_name, station_data = group

            nrows = station_data.shape[0]
            nrows = nrows - self.seq_len

            log.info(f"{station_name} : {nrows}")
            for sample_id in range(0, nrows, self.stride):
                sample, target = [], []
                for tid in range(0, self.seq_len):
                    row = station_data.iloc[sample_id + tid]
                    for col_name, col_value in row.iteritems():
                        if not self.use_station:
                            if col_name == "station":
                                continue
                        if col_name not in self.target_cols:
                            vocab_id = self.vocab.get_id(col_value, col_name)
                            sample.append(vocab_id)

                    if self.mlm:
                        sample.append(sep_id)
                    target.append(row[self.target_cols].tolist())

                self.samples.append(sample)
                self.targets.append(target)

        assert len(self.samples) == len(self.targets)
        log.info(f"total samples {len(self.samples)}")

        self.ncols = len(self.vocab.field_keys)

    def init_vocab(self):
        cols = list(self.data.columns)

        if not self.use_station:
            cols.remove('station')

        for col in self.target_cols:
            cols.remove(col)

        self.vocab.set_field_keys(cols)

        for column in cols:
            unique_values = self.data[column].value_counts(sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        print(f"columns used for vocab: {list(cols)}")
        print(f"total vocabulary size: {len(self.vocab.id2token)}")

        for column in cols:
            vocab_size = len(self.vocab.token2id[column])
            print(f"column : {column}, vocab size : {vocab_size}")

    def read_data(self, root):
        all_stations = None
        fnames = glob.glob(f"{root}/*.csv")
        for fname in fnames:
            station_data = pd.read_csv(fname)

            if all_stations is None:
                all_stations = station_data
            else:
                all_stations = all_stations.append(station_data, ignore_index=True)

        all_stations.drop(columns=['No'], inplace=True, axis=1)
        log.info(f"shape (original)   : {all_stations.shape}")
        all_stations = all_stations.dropna()
        log.info(f"shape (after nan removed): {all_stations.shape}")
        return all_stations

    def save_vocab(self, vocab_dir):
        file_name = path.join(vocab_dir, f'vocab.nb')
        log.info(f"saving vocab at {file_name}")
        self.vocab.save_vocab(file_name)