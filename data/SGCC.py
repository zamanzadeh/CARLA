import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DEFAULT_ROOT = os.path.join(
    os.environ.get('CARLA_DATA_ROOT', r'd:\CARLA\datasets'),
    'SGCC', 'processed'
)


def _load_file_labels(root):
    """Baca file_labels.csv -> dict {fname: label(int 0/1)}."""
    path = os.path.join(root, 'file_labels.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"file_labels.csv tidak ditemukan: {path}")
    labels = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['fname'].strip()] = int(row['label'])
    return labels


class SGCC(Dataset):
    """
    Dataset SGCC (State Grid Corporation of China) — deteksi pencurian listrik.
    Label per CONSUMER (file), bukan per timestamp.

    Struktur folder (output preprocess_sgcc.py):
        datasets/SGCC/processed/
            data/{CONS_NO}.npy   -> (T, 1) float32, raw kWh harian TANPA normalisasi
            file_labels.csv      -> fname, label   (0=normal, 1=pencurian)
            train_list.txt       -> consumer NORMAL untuk pooled training
            val_list.txt         -> held-out untuk pilih checkpoint terbaik
            test_list.txt        -> held-out untuk evaluasi akhir (tidak pernah dilihat
                                    selama training)

    Perbedaan dari MSL/SMD/LEAD:
      - Label per FILE (consumer), bukan per timestamp/window.
      - T sangat pendek (26 hari), sehingga window_size harus kecil (≤ 10).
      - Self-normalize per consumer (skala kWh beda tiap pelanggan).

    Pipeline normalisasi (sama persis dengan PLN — KRITIS untuk order injeksi anomali):
      train=True  : data disimpan RAW kWh. get_info() mengembalikan (mean, std) asli.
                    AugmentedDataset: attack(ts_raw) → normalize dengan mean/std ini.
                    Order BENAR (Jokar 2016): EAI di ruang kWh asli, bukan ternormalisasi.
      train=False : data dinormalisasi di __init__ (untuk eval/inferensi tanpa AugmentedDataset).

    Window size:
      Dengan T=26 dan stride=1, positive pairs (index > 10) ada sebanyak T - W - 11.
      Agar ≥ 5 positive pairs per consumer: W ≤ 10.
      Rekomendasi default: window_size=7 (weekly pattern, 8 positive pairs).
    """

    classes = ['Normal', 'Theft']

    def __init__(
        self,
        fname,
        root=_DEFAULT_ROOT,
        train=True,
        transform=None,
        sanomaly=None,
        mean_data=None,
        std_data=None,
        window_size=10,    # 7 hari = weekly pattern; ≤ 10 untuk ≥ 5 positive pairs (T=26)
        stride=1,
    ):
        super().__init__()

        self.root      = root
        self.fname     = fname
        self.train     = train
        self.transform = transform
        self.sanomaly  = sanomaly

        wsz, st = window_size, stride

        # ------------------------------------------------------------------
        # 1. Load raw data
        # ------------------------------------------------------------------
        file_path = os.path.join(root, 'data', fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SGCC file not found: {file_path}")

        temp = np.load(file_path).astype(np.float64)   # (T, 1) raw kWh
        if np.isnan(temp).any():
            print(f'[WARNING] NaN di {fname}, diganti 0')
            temp = np.nan_to_num(temp)
        if temp.ndim == 1:
            temp = temp[:, None]                        # (T,) → (T, 1)

        # ------------------------------------------------------------------
        # 2. Label per-file (consumer)
        # ------------------------------------------------------------------
        if train:
            file_label = 0
        else:
            file_labels = _load_file_labels(root)
            if fname not in file_labels:
                raise KeyError(f"'{fname}' tidak ada di file_labels.csv")
            file_label = int(file_labels[fname])
        self.file_label = file_label

        # ------------------------------------------------------------------
        # 3. Normalisasi per-consumer (self-normalize)
        # ------------------------------------------------------------------
        if mean_data is not None and std_data is not None:
            self.mean, self.std = mean_data, std_data
        else:
            self.mean = np.mean(temp, axis=0)   # (1,) — mean per fitur (hanya 1 fitur)
            self.std  = np.std(temp,  axis=0)   # (1,)

        if not train:
            # Test/eval: normalisasi di sini agar model langsung terima input ternormalisasi
            # (konsisten dengan MSL/SMD — test data dinormalisasi sebelum inferensi).
            safe_std = np.where(self.std == 0.0, 1.0, self.std)
            temp = (temp - self.mean) / (safe_std + 1e-8)
        # Train: TIDAK dinormalisasi — AugmentedDataset yang menormalisasi SETELAH injeksi.
        # Order yang benar: attack(Ec_raw) → normalize, bukan normalize → attack(Ec_norm).

        # ------------------------------------------------------------------
        # 4. Windowing — semua window mewarisi label file
        # ------------------------------------------------------------------
        self.data = np.asarray(temp)
        self.data, self.targets = self._create_windows(wsz, st)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_windows(self, w, stride):
        n  = self.data.shape[0]
        sz = int((n - w) / stride)
        if sz <= 0:
            # Data terlalu pendek — kembalikan satu window zero (fallback aman)
            print(f'[WARNING] {self.fname}: T={n} < W={w}, dikembalikan window zero.')
            windows = [np.zeros((w, self.data.shape[1]), dtype=np.float64)]
            return np.stack(windows), np.array([self.file_label])

        windows, wlabels = [], []
        for i in range(sz):
            s = i * stride
            windows.append(self.data[s:s + w])
            wlabels.append(self.file_label)
        return np.stack(windows), np.stack(wlabels)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        ts_org = torch.from_numpy(self.data[index]).float().to(device)
        target = torch.tensor(int(self.targets[index]), dtype=torch.long).to(device)
        class_name = self.classes[int(target)]
        ts_size = (ts_org.shape[0], ts_org.shape[1]) if ts_org.ndim == 2 else (ts_org.shape[0],)
        return {
            'ts_org':  ts_org,
            'target':  target,
            'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name},
        }

    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    # Utilities (dibutuhkan pipeline CARLA)
    # ------------------------------------------------------------------
    def get_info(self):
        """
        Kembalikan (mean, std) per-consumer untuk AugmentedDataset.

        Saat train=True, self.data berisi window RAW (belum dinormalisasi).
        AugmentedDataset.create_pairs() akan:
            ts_ss_augment = attack(ts_raw)          ← injeksi di ruang kWh asli ✓
            normalized    = (ts_ss_augment - mean) / std   ← baru normalize ✓
        """
        return self.mean, self.std

    def get_ts(self, index):
        return self.data[index]

    def concat_ds(self, new_ds):
        self.data    = np.concatenate((self.data,    new_ds.data),    axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def extra_repr(self):
        return f"Split: {'Train' if self.train else 'Test'}  FileLabel: {self.file_label}"
