
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.mypath import MyPath
from sklearn.preprocessing import StandardScaler, RobustScaler


class Power(Dataset):
    """`power demand <https://www>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ```` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
    """
    base_folder = ''

    def __init__(self, root=MyPath.db_root_dir('power'), fname='', train=True, transform=None, panomaly= None, sanomaly= None, sanomaly2=None, mean_data=None, std_data=None):

        super(Power, self).__init__()
        self.root = root
        self.transform = transform
        self.panomaly = panomaly
        self.sanomaly = sanomaly
        self.sanomaly2 = sanomaly2
        self.train = train  # training set or test set
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []
        wsize, wstride = 100, 1

        self.mean, self.std = mean_data, std_data

        with open(os.path.join(self.root, fname), 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')

        sp_points = (fname.removesuffix('.txt')).split('_')

        d_sz = csv_reader.shape
        tr_point = int(sp_points[-3])
        start_anom, end_anom = int(sp_points[-2]), int(sp_points[-1])
        labels = np.zeros(d_sz[0])
        ts = csv_reader.to_numpy() #.ravel()

        if self.train:
            TRAIN_TS = ts[:tr_point]
            self.data = np.asarray(TRAIN_TS)
            self.mean = TRAIN_TS.mean()
            self.std = TRAIN_TS.std()

        else:
            TEST_TS = ts[tr_point:]
            if self.std == 0.0: self.std = 1.0
            temp = (TEST_TS - self.mean) / self.std
            self.data = np.asarray(temp)
            labels[(start_anom - tr_point):(end_anom - tr_point)] = 1

        self.targets = np.asarray(labels)
        self.data, self.targets = self.convert_to_windows(wsize, wstride)

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.data.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.data[st:st+w_size]
            if sum(self.targets[st:st+w_size]) > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        ts = self.data[index]
        if len(self.targets) > 0:
            target = self.targets[index].astype(int)
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = len(ts)

        if self.transform is not None:
            ts = self.transform(ts)

        out = {'ts_org': ts, 'target': target, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def get_info(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")