"""
utils/dataset_channels.py

- MSL / SMAP : channel list dari labeled_anomalies.csv (kolom 'spacecraft')
- SMD        : machine list dari folder train/ di direktori dataset
- PLN        : file list dari train_list.txt / test_list.txt + label dari file_labels.csv
- SGCC       : file list dari train_list.txt / val_list.txt / test_list.txt + label dari
               file_labels.csv (sama pola dengan PLN — label per consumer/file)

Digunakan oleh run_msl.py, run_smap.py, run_smd.py, run_pln.py, run_sgcc.py, dan evaluate.py.
"""

import os
import csv
import pandas as pd
from utils.mypath import MyPath


# ------------------------------------------------------------------ #
#  MSL / SMAP  (via labeled_anomalies.csv)
# ------------------------------------------------------------------ #

def get_channels(spacecraft: str) -> list:
    """
    Kembalikan list chan_id untuk 'MSL' atau 'SMAP'
    berdasarkan labeled_anomalies.csv.
    """
    spacecraft = spacecraft.upper()
    assert spacecraft in ('MSL', 'SMAP'), \
        f"spacecraft harus 'MSL' atau 'SMAP', bukan '{spacecraft}'"

    csv_path = os.path.join(MyPath.db_root_dir('msl'), 'labeled_anomalies.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labeled_anomalies.csv tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    channels = df[df['spacecraft'] == spacecraft]['chan_id'].tolist()

    if not channels:
        raise ValueError(f"Tidak ada channel untuk spacecraft '{spacecraft}' di CSV")

    return sorted(channels)


def get_spacecraft(chan_id: str) -> str:
    """Kembalikan 'MSL' atau 'SMAP' untuk chan_id tertentu."""
    csv_path = os.path.join(MyPath.db_root_dir('msl'), 'labeled_anomalies.csv')
    df = pd.read_csv(csv_path)
    rows = df[df['chan_id'] == chan_id]
    if rows.empty:
        raise ValueError(f"Channel '{chan_id}' tidak ditemukan di labeled_anomalies.csv")
    return rows.iloc[0]['spacecraft']


# ------------------------------------------------------------------ #
#  SMD  (via listing folder train/)
# ------------------------------------------------------------------ #

def get_smd_machines() -> list:
    """
    Kembalikan list nama file machine SMD (tanpa path),
    dibaca dari folder datasets/SMD/train/.
    Contoh: ['machine-1-1.txt', 'machine-1-2.txt', ...]
    """
    train_dir = os.path.join(MyPath.db_root_dir('smd'), 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Folder SMD train tidak ditemukan: {train_dir}")

    machines = sorted(
        f for f in os.listdir(train_dir)
        if f.startswith('machine-') and f.endswith('.txt')
    )

    if not machines:
        raise ValueError(f"Tidak ada file machine-*.txt di {train_dir}")

    return machines


# ------------------------------------------------------------------ #
#  LEAD  (via building_list.txt)
# ------------------------------------------------------------------ #

def get_lead_buildings(eval_only: bool = False) -> list:
    """
    Kembalikan list nama file building LEAD.

    eval_only=False  → building_list.txt      (semua 198 building, untuk training)
    eval_only=True   → building_eval_list.txt  (hanya yang punya anomali di test set,
                                                untuk evaluasi)
    """
    fname = 'building_eval_list.txt' if eval_only else 'building_list.txt'
    list_path = os.path.join(MyPath.db_root_dir('lead'), fname)
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"{fname} tidak ditemukan: {list_path}")

    with open(list_path) as f:
        buildings = [line.strip() for line in f if line.strip()]

    if not buildings:
        raise ValueError(f"{fname} kosong: {list_path}")

    return buildings


# ------------------------------------------------------------------ #
#  PLN  (label per FILE, via train_list.txt / val_list.txt / test_list.txt /
#  file_labels.csv)
# ------------------------------------------------------------------ #

def _read_pln_list(fname: str) -> list:
    list_path = os.path.join(MyPath.db_root_dir('pln'), fname)
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"{fname} tidak ditemukan: {list_path}")
    with open(list_path) as f:
        items = [line.strip() for line in f if line.strip()]
    if not items:
        raise ValueError(f"{fname} kosong: {list_path}")
    return items


def get_pln_train_files() -> list:
    """
    Daftar file PLN NORMAL yang dipool jadi training (unsupervised pretext +
    sumber 'apa itu normal' untuk classification stage). Dibaca dari
    datasets/PLN/processed/train_list.txt.
    """
    return _read_pln_list('train_list.txt')


def get_pln_test_files() -> list:
    """
    Daftar file PLN held-out (campuran normal + anomali) untuk evaluasi
    file-level FINAL (evaluate.py). Dibaca dari datasets/PLN/processed/test_list.txt.
    File-file ini TIDAK boleh dilihat/dipakai selama training maupun untuk
    memilih checkpoint terbaik - lihat get_pln_val_files() untuk itu.
    """
    return _read_pln_list('test_list.txt')


def get_pln_val_files() -> list:
    """
    Daftar file PLN held-out (campuran normal + anomali) untuk MEMILIH
    checkpoint terbaik selama training (val_dataset di carla_pretext.py /
    carla_classification.py). Dibaca dari datasets/PLN/processed/val_list.txt.

    Kompatibilitas: kalau val_list.txt belum ada (data diproses dengan
    preprocess_pln.py versi LAMA, sebelum 3-way split ditambahkan), fallback
    ke test_list.txt seperti perilaku original - dengan WARNING, karena ini
    berarti file yang sama dipakai untuk pilih checkpoint terbaik DAN untuk
    evaluasi akhir (data leakage, metrik akhir jadi bias optimis). Jalankan
    ulang preprocess_pln.py untuk split yang benar.
    """
    list_path = os.path.join(MyPath.db_root_dir('pln'), 'val_list.txt')
    if not os.path.exists(list_path):
        print("[WARNING] val_list.txt tidak ditemukan - fallback memakai test_list.txt "
              "sebagai val_dataset (perilaku lama). Ini menyebabkan data leakage: model "
              "dipilih (best checkpoint) berdasarkan data yang sama dengan yang dipakai "
              "evaluate.py untuk melaporkan metrik akhir. Jalankan ulang preprocess_pln.py "
              "untuk menghasilkan val_list.txt terpisah.")
        return _read_pln_list('test_list.txt')
    return _read_pln_list('val_list.txt')





def get_pln_file_labels() -> dict:
    """
    Kembalikan dict {fname: label(0/1)} dari datasets/PLN/processed/file_labels.csv.
    """
    csv_path = os.path.join(MyPath.db_root_dir('pln'), 'file_labels.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"file_labels.csv tidak ditemukan: {csv_path}")
    labels = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['fname'].strip()] = int(row['label'])
    return labels


def get_pln_label(fname: str) -> int:
    """Kembalikan label (0/1) untuk satu file PLN tertentu."""
    labels = get_pln_file_labels()
    if fname not in labels:
        raise KeyError(f"'{fname}' tidak ada di file_labels.csv")
    return labels[fname]


# ------------------------------------------------------------------ #
#  SGCC  (label per consumer/FILE, via train_list.txt / val_list.txt /
#  test_list.txt / file_labels.csv — pola sama dengan PLN)
# ------------------------------------------------------------------ #

def _read_sgcc_list(fname: str) -> list:
    list_path = os.path.join(MyPath.db_root_dir('sgcc'), fname)
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"{fname} tidak ditemukan: {list_path}")
    with open(list_path) as f:
        items = [line.strip() for line in f if line.strip()]
    if not items:
        raise ValueError(f"{fname} kosong: {list_path}")
    return items


def get_sgcc_train_files() -> list:
    """Consumer SGCC NORMAL yang dipool jadi training. Dari datasets/SGCC/processed/train_list.txt."""
    return _read_sgcc_list('train_list.txt')


def get_sgcc_val_files() -> list:
    """
    Consumer SGCC held-out untuk memilih checkpoint terbaik selama training.
    Dari datasets/SGCC/processed/val_list.txt. Fallback ke test_list.txt kalau
    val_list.txt belum ada (dengan WARNING — ini menyebabkan data leakage).
    """
    list_path = os.path.join(MyPath.db_root_dir('sgcc'), 'val_list.txt')
    if not os.path.exists(list_path):
        print("[WARNING] SGCC val_list.txt tidak ditemukan — fallback memakai test_list.txt "
              "sebagai val (data leakage). Jalankan ulang preprocess_sgcc.py.")
        return _read_sgcc_list('test_list.txt')
    return _read_sgcc_list('val_list.txt')


def get_sgcc_test_files() -> list:
    """
    Consumer SGCC held-out untuk evaluasi akhir (evaluate.py). TIDAK PERNAH dilihat
    selama training. Dari datasets/SGCC/processed/test_list.txt.
    """
    return _read_sgcc_list('test_list.txt')


def get_sgcc_file_labels() -> dict:
    """Kembalikan dict {fname: label(0/1)} dari datasets/SGCC/processed/file_labels.csv."""
    csv_path = os.path.join(MyPath.db_root_dir('sgcc'), 'file_labels.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"file_labels.csv tidak ditemukan: {csv_path}")
    labels = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['fname'].strip()] = int(row['label'])
    return labels


# ------------------------------------------------------------------ #
#  SGCC_LARGE  (dataset besar 42K consumers, window=365, stride=14)
#  Pola identik dengan SGCC tetapi root → processed_large/
# ------------------------------------------------------------------ #

def _read_sgcc_large_list(fname: str) -> list:
    list_path = os.path.join(MyPath.db_root_dir('sgcc_large'), fname)
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"{fname} tidak ditemukan: {list_path}")
    with open(list_path) as f:
        items = [line.strip() for line in f if line.strip()]
    if not items:
        raise ValueError(f"{fname} kosong: {list_path}")
    return items


def get_sgcc_large_train_files() -> list:
    """Consumer SGCC_LARGE NORMAL untuk training. Dari processed_large/train_list.txt."""
    return _read_sgcc_large_list('train_list.txt')


def get_sgcc_large_val_files() -> list:
    """Consumer SGCC_LARGE held-out untuk pilih checkpoint. Dari processed_large/val_list.txt."""
    list_path = os.path.join(MyPath.db_root_dir('sgcc_large'), 'val_list.txt')
    if not os.path.exists(list_path):
        print("[WARNING] SGCC_LARGE val_list.txt tidak ditemukan — fallback ke test_list.txt "
              "(data leakage). Jalankan ulang preprocess_sgcc_large.py.")
        return _read_sgcc_large_list('test_list.txt')
    return _read_sgcc_large_list('val_list.txt')


def get_sgcc_large_test_files() -> list:
    """Consumer SGCC_LARGE untuk evaluasi akhir. Dari processed_large/test_list.txt."""
    return _read_sgcc_large_list('test_list.txt')


def get_sgcc_large_file_labels() -> dict:
    """Kembalikan dict {fname: label(0/1)} dari processed_large/file_labels.csv."""
    csv_path = os.path.join(MyPath.db_root_dir('sgcc_large'), 'file_labels.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"file_labels.csv tidak ditemukan: {csv_path}")
    labels = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['fname'].strip()] = int(row['label'])
    return labels
