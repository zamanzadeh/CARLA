
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.mypath import MyPath
from sklearn.preprocessing import StandardScaler, RobustScaler


class KPI(Dataset):
    """`SMD <https://www>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ```` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a ts
            and returns a transformed version.
    """
    base_folder = ''

    def __init__(self, fname, root=MyPath.db_root_dir('kpi'), train=True, transform=None, sanomaly= None, mean_data=None, std_data=None):

        super(KPI, self).__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []
        wsize, wstride = 200, 5

        if self.train:
            self.base_folder += 'train'
        else:
            self.base_folder += 'test'

        file_path = os.path.join(self.root, self.base_folder, fname)
        temp = pd.read_csv(file_path)
        temp = temp.set_index(['timestamp']).sort_index()
        data = np.asarray(temp['value'])
        labels = np.asarray(temp['label'])

        if np.any(sum(np.isnan(data))!=0):
            print('Data contains NaN which replaced with zero')
            data = np.nan_to_num(data)

        self.mean, self.std = mean_data, std_data
        if self.train:
            self.mean = np.mean(data)
            self.std = np.std(data)
        else:
            if self.std == 0.0: self.std = 1.0
            data = (data - self.mean) / self.std

        self.targets = labels
        self.data = data
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
        ts_org = torch.from_numpy(self.data[index]).float().to(device)  # cuda
        if len(self.targets) > 0:
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long).to(device)
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = len(ts)

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