import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DEFAULT_ROOT = os.path.join(
    os.environ.get('CARLA_DATA_ROOT', r'd:\CARLA\datasets'),
    'PLN', 'processed'
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


class PLN(Dataset):
    """
    Dataset PLN (load profile listrik, label per FILE bukan per timestamp).

    Struktur folder (output preprocessing):
        datasets/PLN/processed/
            data/{fname}.npy   -> (T, n_features) float32, satu file = satu load profile lengkap
                                   (timestamp tiap 15 menit, multi-kolom)
            file_labels.csv    -> kolom: fname,label   (label: 0=normal, 1=anomali)
            train_list.txt     -> daftar fname NORMAL untuk dipool jadi training (unsupervised)
            test_list.txt      -> daftar fname held-out (campuran normal+anomali) untuk evaluasi

    Beda mendasar dengan MSL/SMD/LEAD: di sana train/test adalah belahan WAKTU dari satu
    entitas yang sama, dan label tersedia di tiap timestamp. Di PLN, train/test adalah file
    yang BERBEDA, dan hanya ada SATU label per file (mewakili seluruh durasi file itu).
    Konsekuensinya:
      - train=True  : file ini diasumsikan NORMAL (dipilih lewat train_list.txt). Label tiap
                       window dipaksa 0 - konsisten dengan paradigma pretext CARLA yang memang
                       tidak pernah memakai label asli saat training (selalu unsupervised).
                       Data disimpan RAW (belum dinormalisasi) — AugmentedDataset yang
                       menormalisasi SETELAH injeksi anomali, sesuai order fisik Jokar 2016:
                       attack(Ec) → normalize, bukan normalize → attack(Ec_norm).
      - train=False : label asli file (dari file_labels.csv) diwariskan ke SETIAP window di
                       file ini. evaluate.py lalu mengagregasi skor window->satu skor file,
                       dan membandingkannya ke label file ini sebagai ground truth.
                       Data dinormalisasi langsung di __init__ agar eval/inferensi tanpa
                       AugmentedDataset tetap mendapat input ternormalisasi untuk model.
      - Normalisasi dilakukan PER FILE (mean/std file itu sendiri, self-normalize), BUKAN
        dengan mean/std lintas file. Ini supaya model membandingkan BENTUK pola pemakaian,
        bukan skala absolut (yang bisa berbeda jauh antar pelanggan/feeder).
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
        window_size=96,   # 96 x 15menit = 1 hari. Sesuaikan ke karakteristik data PLN Anda.
        stride=4,
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
        file_path = os.path.join(root, 'data', fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PLN file not found: {file_path}")

        temp = np.load(file_path).astype(np.float64)
        if np.isnan(temp).any():
            print(f'[WARNING] NaN di {fname}, diganti 0')
            temp = np.nan_to_num(temp)
        if temp.ndim == 1:
            temp = temp[:, None]

        # --------------------------------------------------
        # 2. Label per-file
        # --------------------------------------------------
        if train:
            file_label = 0
        else:
            file_labels = _load_file_labels(root)
            if fname not in file_labels:
                raise KeyError(f"'{fname}' tidak ada di file_labels.csv")
            file_label = int(file_labels[fname])
        self.file_label = file_label

        # --------------------------------------------------
        # 3. Normalisasi (self-normalize per file)
        # --------------------------------------------------
        if mean_data is not None and std_data is not None:
            self.mean, self.std = mean_data, std_data
        else:
            self.mean = np.mean(temp, axis=0)
            self.std = np.std(temp, axis=0)

        if not train:
            # Test/eval: normalisasi di sini agar eval/inferensi langsung
            # (tanpa AugmentedDataset) mendapat data ternormalisasi untuk model.
            # Konsisten dengan MSL/SMD/LEAD: test data dinormalisasi dengan
            # mean/std train (atau self-normalize untuk PLN karena per-pelanggan).
            safe_std = np.where(self.std == 0.0, 1.0, self.std)
            temp = (temp - self.mean) / (safe_std + 1e-8)
        # Train: data TIDAK dinormalisasi — AugmentedDataset menormalisasi
        # SETELAH injeksi anomali via get_info(). Order yang benar (Jokar 2016):
        #   attack(Ec_raw) → normalize  bukan  normalize → attack(Ec_norm)

        # --------------------------------------------------
        # 4. Windowing - semua window mewarisi label file
        # --------------------------------------------------
        self.data = np.asarray(temp)
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
            windows.append(window)
            wlabels.append(self.file_label)
        if not windows:
            # Dataset terlalu kecil untuk satu window - kembalikan satu window kosong
            windows = [np.zeros((w, self.data.shape[1]), dtype=np.float64)]
            wlabels = [self.file_label]
        return np.stack(windows), np.stack(wlabels)

    # --------------------------------------------------
    # Dataset API
    # --------------------------------------------------
    def __getitem__(self, index):
        ts_org = torch.from_numpy(self.data[index]).float().to(device)
        target = torch.tensor(int(self.targets[index]), dtype=torch.long).to(device)
        class_name = self.classes[int(target)]
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
        """
        Kembalikan (mean, std) per-file yang dipakai AugmentedDataset untuk
        menormalisasi ts_org DAN ts_ss_augment SETELAH injeksi anomali.

        [FIX] Sebelumnya mengembalikan (0, 1) karena PLN sudah pre-normalize
        di __init__ untuk semua kasus (train dan test). Ini menyebabkan
        AugmentedDataset.create_pairs() menginjeksi anomali di ruang ternormalisasi,
        bukan di ruang konsumsi asli Ec(t) seperti yang didefinisikan Jokar 2016.

        Sekarang konsisten dengan MSL/SMD/LEAD:
          - train=True  : self.data berisi window RAW (belum dinormalisasi)
                          get_info() mengembalikan (mean, std) asli per-file
                          AugmentedDataset: attack(ts_raw) → normalize dengan ini
          - train=False : self.data berisi window ternormalisasi (langsung di __init__)
                          get_info() tetap mengembalikan mean/std yang sama, tapi
                          jarang dipanggil pada path eval (data sudah siap pakai)
        """
        return self.mean, self.std

    def get_ts(self, index):
        return self.data[index]

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def extra_repr(self):
        return "Split: {}  FileLabel: {}".format("Train" if self.train else "Test", self.file_label)
