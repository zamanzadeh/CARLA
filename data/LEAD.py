import os
import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DEFAULT_ROOT = os.path.join(
    os.environ.get('CARLA_DATA_ROOT', r'd:\CARLA\datasets'),
    'LEAD', 'processed'
)


class LEAD(Dataset):
    """
    LEAD (Large-scale Energy Anomaly Detection) dataset untuk CARLA.

    Struktur folder (output dari preprocessing):
        datasets/LEAD/processed/
            train/{id}.npy          → (T_train, 39)  float32
            test/{id}.npy           → (T_test,  39)  float32
            test_label/{id}.npy     → (T_test,)      int8
            building_list.txt

    fname  : nama file building, misal '1.npy'
    train  : True → load train split; False → load test split
    """

    classes = ['Normal', 'Anomaly']

    def __init__(
        self,
        fname,
        root=_DEFAULT_ROOT,
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

        wsz, st = window_size, stride

        # --------------------------------------------------
        # 1. Load raw data
        # --------------------------------------------------
        split = 'train' if train else 'test'
        file_path = os.path.join(root, split, fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LEAD file not found: {file_path}")

        temp = np.load(file_path).astype(np.float64)

        if np.isnan(temp).any():
            print(f'[WARNING] NaN di {fname} ({split}), diganti 0')
            temp = np.nan_to_num(temp)

        # --------------------------------------------------
        # 2. Labels
        # --------------------------------------------------
        if train:
            labels = np.zeros(len(temp), dtype=int)
        else:
            label_path = os.path.join(root, 'test_label', fname)
            labels = np.load(label_path).astype(int)

        # --------------------------------------------------
        # 3. Normalisasi
        # --------------------------------------------------
        self.mean, self.std = mean_data, std_data

        if train:
            self.mean = np.mean(temp, axis=0)
            self.std = np.std(temp, axis=0)
        else:
            assert mean_data is not None and std_data is not None, \
                "Berikan mean_data dan std_data dari train set untuk LEAD test"
            safe_std = np.where(self.std == 0.0, 1.0, self.std)
            temp = (temp - self.mean) / (safe_std + 1e-8)

        # --------------------------------------------------
        # 4. Windowing
        # --------------------------------------------------
        self.data = np.asarray(temp)
        self.targets = np.asarray(labels)
        self.data, self.targets = self._create_windows(wsz, st)

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _create_windows(self, w, stride):
        n = self.data.shape[0]
        sz = int((n - w) / stride)
        windows, wlabels = [], []
        for i in range(sz):
            s = i * stride
            window = self.data[s:s + w]
            label = 1 if (self.targets[s:s + w] > 0).any() else 0
            windows.append(window)
            wlabels.append(label)
        if not windows:
            # Dataset terlalu kecil untuk satu window — kembalikan satu window kosong
            windows = [np.zeros((w, self.data.shape[1]), dtype=np.float64)]
            wlabels = [0]
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
    # Utilities (dibutuhkan pipeline CARLA)
    # --------------------------------------------------
    def get_info(self):
        """Kembalikan (mean, std) dari train set."""
        return self.mean, self.std

    def get_ts(self, index):
        return self.data[index]

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train else "Test")
