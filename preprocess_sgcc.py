"""
preprocess_sgcc.py - Preprocessing dataset SGCC (State Grid Corporation of China)
untuk pipeline CARLA deteksi pencurian listrik.

Format raw data:
    datasets/SGCC/raw/datasetsmall.csv.zip  (atau .csv setelah diekstrak)
        Format WIDE: satu baris = satu konsumen
        Kolom:
          - 26 kolom tanggal (01/01/2014 .. 01/26/2014): konsumsi harian dalam kWh
          - CONS_NO : ID konsumen (hash string)
          - FLAG    : label 0=normal, 1=anomali/pencurian

    Catatan: format berbeda dari PLN (yang per-file long-format).
    Di sini satu row CSV = satu "file" PLN, satu consumer = satu time series.

Output (siap dipakai run_sgcc.py / evaluate.py --dataset SGCC):
    datasets/SGCC/processed/data/{CONS_NO}.npy  -> (26, 1) float32, RAW kWh TANPA normalisasi
    datasets/SGCC/processed/file_labels.csv     -> fname,label
    datasets/SGCC/processed/train_list.txt      -> consumer NORMAL, dipool untuk training
    datasets/SGCC/processed/val_list.txt        -> held-out untuk memilih checkpoint terbaik
    datasets/SGCC/processed/test_list.txt       -> held-out untuk evaluasi akhir (tidak pernah
                                                    dilihat selama training)

PENTING - urutan pipeline CARLA:
    1. raw kWh (T,1) disimpan TANPA normalisasi ke .npy
    2. SGCC(train=True) mengembalikan raw kWh; SGCC(train=False) menormalisasi per-consumer
    3. AugmentedDataset.create_pairs(): attack(ts_raw) → normalize pakai SGCC.get_info()
    Ini KONSISTEN dengan order Jokar 2016: serangan EAI dilakukan di ruang kWh asli,
    bukan di ruang ternormalisasi. Jangan pindahkan normalisasi ke sini.

Cara pakai:
    # Ekstrak zip dulu (kalau belum):
    cd datasets/SGCC/raw && unzip datasetsmall.csv.zip

    python preprocess_sgcc.py
    python preprocess_sgcc.py --raw_csv datasets/SGCC/raw/datasetsmall.csv \\
                               --out_dir datasets/SGCC/processed \\
                               --train_frac 0.8 --val_frac 0.5 --seed 42
"""

import os
import csv
import sys
import zipfile
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()

# Kolom non-fitur yang akan di-drop sebelum menyimpan time series
META_COLS = {'CONS_NO', 'FLAG'}


class _Tee:
    """Duplikasi stdout ke file log (terminal + file sekaligus)."""
    def __init__(self, filepath):
        self._file = open(filepath, 'w', encoding='utf-8')
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def _load_raw_csv(raw_csv: Path) -> pd.DataFrame:
    """
    Baca SGCC wide CSV. Kalau path .zip langsung diberikan, ekstrak in-memory.
    Kembalikan DataFrame dengan kolom tanggal (kWh), CONS_NO, FLAG.
    """
    p = Path(raw_csv)
    if p.suffix == '.zip':
        with zipfile.ZipFile(p) as zf:
            names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not names:
                raise FileNotFoundError(f"Tidak ada .csv di dalam {p}")
            with zf.open(names[0]) as f:
                df = pd.read_csv(f)
        print(f"  Baca dari ZIP: {p.name} → {names[0]}")
    else:
        df = pd.read_csv(p)
        print(f"  Baca CSV: {p.name}")

    # Validasi kolom wajib
    for col in ('CONS_NO', 'FLAG'):
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan. Pastikan format SGCC benar.")

    return df


def _get_date_cols(df: pd.DataFrame) -> list:
    """Kembalikan daftar kolom tanggal (semua kolom kecuali META_COLS)."""
    return [c for c in df.columns if c not in META_COLS]


def _fill_nan(row_series: pd.Series) -> np.ndarray:
    """
    Isi NaN pada satu baris (time series satu consumer) dengan interpolasi linear.
    Ujung yang masih NaN (leading/trailing) diisi dengan forward/backward fill.
    Kalau semua NaN, isi dengan 0 (fallback).

    TIDAK ada normalisasi di sini - data keluar dalam satuan kWh asli.
    """
    s = row_series.copy().astype(float)
    s = s.interpolate(method='linear', limit_direction='both')
    s = s.ffill().bfill().fillna(0.0)
    return s.to_numpy(dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Preprocess SGCC untuk CARLA")
    parser.add_argument(
        '--raw_csv',
        default=str(BASE_DIR / 'datasets' / 'SGCC' / 'raw' / 'datasetsmall.csv.zip'),
        help="Path ke CSV atau ZIP SGCC. Default: datasets/SGCC/raw/datasetsmall.csv.zip"
    )
    parser.add_argument(
        '--out_dir',
        default=str(BASE_DIR / 'datasets' / 'SGCC' / 'processed'),
        help="Direktori output. Default: datasets/SGCC/processed"
    )
    parser.add_argument(
        '--train_frac', type=float, default=0.8,
        help="Fraksi konsumen NORMAL untuk train_list.txt (default: 0.8)"
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.5,
        help="Fraksi held-out (normal sisa + semua anomali) untuk val_list.txt (default: 0.5)"
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = BASE_DIR / 'results' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'preprocess_sgcc_{ts}.txt'

    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        _run(args, ts, log_path)
    finally:
        tee.close()


def _run(args, ts, log_path):
    print(f"Run        : {ts}")
    print(f"Log file   : {log_path}")
    print(f"Raw CSV    : {args.raw_csv}")
    print(f"Out dir    : {args.out_dir}")
    print(f"train_frac : {args.train_frac}")
    print(f"val_frac   : {args.val_frac}")
    print(f"seed       : {args.seed}\n")

    out_dir  = Path(args.out_dir)
    data_dir = out_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Baca raw CSV
    # ------------------------------------------------------------------
    df = _load_raw_csv(Path(args.raw_csv))
    date_cols = _get_date_cols(df)
    T = len(date_cols)
    N = len(df)
    n_nan_total = int(df[date_cols].isna().sum().sum())
    n_anomaly = int((df['FLAG'] == 1).sum())
    n_normal  = int((df['FLAG'] == 0).sum())

    print(f"\nDataset SGCC:")
    print(f"  Jumlah consumer : {N:,}")
    print(f"  Normal (FLAG=0) : {n_normal:,}  ({100*n_normal/N:.1f}%)")
    print(f"  Anomali (FLAG=1): {n_anomaly:,}  ({100*n_anomaly/N:.1f}%)")
    print(f"  Jumlah timestamp: {T}  (kolom: {date_cols[0]} .. {date_cols[-1]})")
    print(f"  Total NaN       : {n_nan_total:,}  ({100*n_nan_total/(N*T):.2f}% dari seluruh nilai)")
    print(f"\nStatistik kWh (RAW, sebelum fill NaN):")
    raw_vals = df[date_cols].values.flatten()
    raw_vals = raw_vals[~np.isnan(raw_vals)]
    print(f"  min={raw_vals.min():.1f}  max={raw_vals.max():.1f}  "
          f"mean={raw_vals.mean():.1f}  std={raw_vals.std():.1f}")

    # ------------------------------------------------------------------
    # 2. Per-consumer: isi NaN → simpan .npy (26, 1) float32 TANPA normalisasi
    # ------------------------------------------------------------------
    print(f"\nMenyimpan {N:,} consumer ke {data_dir} ...")
    fnames_npy = []
    label_map  = {}
    n_nan_fixed = 0

    for i, row in df.iterrows():
        cons_no = str(row['CONS_NO']).strip()
        flag    = int(row['FLAG'])
        series  = row[date_cols]

        n_missing = int(series.isna().sum())
        if n_missing > 0:
            n_nan_fixed += n_missing

        arr = _fill_nan(series)             # (T,) float32, RAW kWh
        arr = arr[:, np.newaxis]            # → (T, 1) agar konsisten dengan multi-feature

        # Validasi: tidak ada normalisasi di sini
        assert arr.dtype == np.float32, "dtype harus float32"
        # (nilai bisa > 1 atau sangat besar - itu benar, ini raw kWh)

        fname = f"{cons_no}.npy"
        np.save(data_dir / fname, arr)
        fnames_npy.append(fname)
        label_map[fname] = flag

        if (i + 1) % 5000 == 0 or (i + 1) == N:
            print(f"  [{i+1:,}/{N:,}] consumer diproses ...")

    print(f"  Total NaN yang diisi interpolasi: {n_nan_fixed:,}")
    print(f"  Shape tiap .npy: ({T}, 1)  — raw kWh, TANPA normalisasi")

    # ------------------------------------------------------------------
    # 3. file_labels.csv
    # ------------------------------------------------------------------
    with open(out_dir / 'file_labels.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fname', 'label'])
        for fname in fnames_npy:
            writer.writerow([fname, label_map[fname]])
    print(f"\nfile_labels.csv ditulis: {len(fnames_npy)} entri")

    # ------------------------------------------------------------------
    # 4. Train / val / test split
    # ------------------------------------------------------------------
    normal_files  = [f for f in fnames_npy if label_map[f] == 0]
    anomaly_files = [f for f in fnames_npy if label_map[f] == 1]

    rng = np.random.RandomState(args.seed)
    rng.shuffle(normal_files)
    rng.shuffle(anomaly_files)

    # train_frac dari normal → train_list.txt
    n_train = max(1, int(round(len(normal_files) * args.train_frac)))
    if len(normal_files) > 1:
        n_train = min(n_train, len(normal_files) - 1)
    train_files   = normal_files[:n_train]
    holdout_normal = normal_files[n_train:]

    # val_frac dari held-out (normal sisa + semua anomali), proporsional per kelas
    n_val_normal  = int(round(len(holdout_normal) * args.val_frac))
    n_val_anomaly = int(round(len(anomaly_files)  * args.val_frac))
    val_files  = holdout_normal[:n_val_normal]  + anomaly_files[:n_val_anomaly]
    test_files = holdout_normal[n_val_normal:]  + anomaly_files[n_val_anomaly:]

    for fname, lst in [('train_list.txt', train_files),
                        ('val_list.txt',   val_files),
                        ('test_list.txt',  test_files)]:
        with open(out_dir / fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lst) + ('\n' if lst else ''))

    n_val_anom  = sum(1 for f in val_files  if label_map[f] == 1)
    n_test_anom = sum(1 for f in test_files if label_map[f] == 1)

    print(f"\nSplit hasil (seed={args.seed}):")
    print(f"  train_list.txt : {len(train_files):,} consumer normal "
          f"(dipool untuk training, tidak dipakai val/test)")
    print(f"  val_list.txt   : {len(val_files):,} consumer  "
          f"({len(val_files)-n_val_anom:,} normal + {n_val_anom:,} anomali) "
          f"— hanya untuk pilih checkpoint terbaik selama training")
    print(f"  test_list.txt  : {len(test_files):,} consumer  "
          f"({len(test_files)-n_test_anom:,} normal + {n_test_anom:,} anomali) "
          f"— hanya untuk evaluate.py, tidak dilihat selama training")

    if n_val_anom == 0:
        print("  [WARNING] val_list.txt tidak punya anomali — pemilihan checkpoint terbaik "
              "berdasarkan F1 selama training tidak bermakna. Naikkan --val_frac.")
    if n_test_anom == 0:
        print("  [WARNING] test_list.txt tidak punya anomali — metrik akhir tidak bermakna.")

    # ------------------------------------------------------------------
    # 5. Ringkasan window size
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"CATATAN WINDOW SIZE (penting untuk configs):")
    print(f"  T = {T} timestamp per consumer, stride = 1")
    print(f"  Jumlah window = T - W = {T} - W")
    print(f"  Positive pairs (index > 10): jumlah = {T-1} - W - 10 = {T-11} - W")
    for w in [4, 5, 6, 7, 8, 10, 12, 14]:
        n_win  = T - w
        n_pos  = max(0, T - 1 - w - 10)
        status = '✓' if n_pos >= 5 else '✗'
        print(f"    W={w:2d}: {n_win} windows, {n_pos} positive pairs  {status}")
    print(f"  → W ≤ 10 agar ≥ 5 positive pairs per consumer")
    print(f"  → Rekomendasi: W=7 (weekly pattern, 8 pairs) atau W=10 (5 pairs tepat)")
    print(f"{'='*60}")
    print(f"\nSelesai. Log: {log_path}")


if __name__ == '__main__':
    main()
