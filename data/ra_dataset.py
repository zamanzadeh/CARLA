import os
import numpy as np
from torch.utils.data import Dataset


class SaveAugmentedDataset(Dataset):

    def __init__(self, data, target):
        super(SaveAugmentedDataset, self).__init__()
        self.classes = ['Normal', 'Anomaly', 'Noise', 'Point', 'Subseq', 'Subseq2']
        self.targets = target
        self.data = data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        ts = self.data[index]
        if len(self.targets) > 0:
            target = int(self.targets[index])
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = (ts.shape[0])

        out = {'ts_org': ts, 'target': target, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def __len__(self):
        return len(self.data)