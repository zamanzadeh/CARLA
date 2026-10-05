import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.mypath import MyPath

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSL(Dataset):
    """
    NASA MSL Dataset — interface compatible dengan SMD dan pipeline CARLA.

    Normalisasi:
      - Train : data disimpan RAW; mean/std dihitung dan disimpan.
                AugmentedDataset yang melakukan normalisasi.
      - Test  : dinormalisasi menggunakan mean/std dari train set (mean_data, std_data).

    __getitem__ mengembalikan:
        {'ts_org': Tensor, 'target': Tensor, 'meta': dict}
    """

    base_folder = ''

    def __init__(
        self,
        fname,
        root=MyPath.db_root_dir('msl'),
        train=True,
        transform=None,
        sanomaly=None,
        mean_data=None,
        std_data=None,
        window_size=200,
        stride=5,
    ):
        super().__init__()

        self.root = root
        self.fname = fname
        self.train = train
        self.transform = transform
        self.sanomaly = sanomaly
        self.classes = ['Normal', 'Anomaly']

        wsz = window_size
        stride = stride

        # --------------------------------------------------
        # 1. Load raw data
        # --------------------------------------------------
        split = 'train' if train else 'test'
        file_path = os.path.join(root, 'data', split, fname + '.npy')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MSL file not found: {file_path}")

        temp = np.load(file_path).astype(np.float64)

        if np.isnan(temp).any():
            print(f'[WARNING] NaN terdeteksi di {fname}, diganti dengan 0')
            temp = np.nan_to_num(temp)

        # --------------------------------------------------
        # 2. Labels
        # --------------------------------------------------
        if train:
            labels = np.zeros(len(temp), dtype=int)
        else:
            labels = self._load_anomaly_labels(fname)

        # --------------------------------------------------
        # 3. Normalisasi
        # --------------------------------------------------
        self.mean, self.std = mean_data, std_data

        if train:
            # Hitung statistik dari train; data TIDAK dinormalisasi di sini
            # (AugmentedDataset yang menormalisasi melalui get_info())
            self.mean = np.mean(temp, axis=0)
            self.std = np.std(temp, axis=0)
        else:
            # Normalisasi test dengan statistik train
            assert mean_data is not None and std_data is not None, \
                "Berikan mean_data dan std_data dari train set untuk test MSL"
            self.std = np.where(self.std == 0.0, 1.0, self.std)
            temp = (temp - self.mean) / (self.std + 1e-8)

        # --------------------------------------------------
        # 4. Windowing
        # --------------------------------------------------
        self.data = np.asarray(temp)
        self.targets = np.asarray(labels)
        self.data, self.targets = self._create_windows(wsz, stride)

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _load_anomaly_labels(self, fname):
        csv_path = os.path.join(self.root, 'labeled_anomalies.csv')
        df = pd.read_csv(csv_path)
        row = df[df['chan_id'] == fname].iloc[0]
        anomalies = ast.literal_eval(row['anomaly_sequences'])
        length = row['num_values']
        labels = np.zeros(length, dtype=int)
        for start, end in anomalies:
            labels[start:end + 1] = 1
        return labels

    def _create_windows(self, w, stride):
        n = self.data.shape[0]
        sz = int((n - w) / stride)
        windows, wlabels = [], []
        for i in range(sz):
            st = i * stride
            window = self.data[st:st + w]
            label = 1 if (self.targets[st:st + w] > 0).any() else 0
            windows.append(window)
            wlabels.append(label)
        return np.stack(windows), np.stack(wlabels)

    # --------------------------------------------------
    # Dataset API
    # --------------------------------------------------
    def __getitem__(self, index):
        ts_org = torch.from_numpy(self.data[index]).float().to(device)

        if len(self.targets) > 0:
            target = torch.tensor(int(self.targets[index]), dtype=torch.long).to(device)
            class_name = self.classes[int(target)]
        else:
            target = torch.tensor(0, dtype=torch.long).to(device)
            class_name = ''

        ts_size = (ts_org.shape[0], ts_org.shape[1]) if ts_org.ndim == 2 else (ts_org.shape[0],)

        return {
            'ts_org': ts_org,
            'target': target,
            'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name},
        }

    def __len__(self):
        return len(self.data)

    # --------------------------------------------------
    # Utilities (dibutuhkan oleh pipeline CARLA)
    # --------------------------------------------------
    def get_info(self):
        """Mengembalikan (mean, std) dari train set."""
        return self.mean, self.std

    def get_ts(self, index):
        return self.data[index]

    def concat_ds(self, new_ds):
        """Gabungkan dataset lain (dipakai untuk mode fname='All')."""
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train else "Test")
