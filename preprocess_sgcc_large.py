"""
preprocess_sgcc_large.py - Preprocessing dataset SGCC besar (42K consumers, 1034 hari)
untuk pipeline CARLA dengan window_size=365, stride=14.

Format raw data:
    datasets/SGCC/raw/data set.csv.zip
        Format WIDE: satu baris = satu consumer
        Kolom: 1034 kolom tanggal (1/1/2014 .. 10/31/2016) + CONS_NO + FLAG

Strategy:
    - Stratified sampling ~2000 consumers: jaga rasio fraud/normal (~8.5%)
    - Windowing (365 hari, stride 14) dilakukan di SGCC dataset class saat load
    - Simpan full (1034, 1) time series TANPA normalisasi ke processed_large/data/
    - Log starttime, endtime, totaltime setiap tahap

Output (siap dipakai run_sgcc_large.py):
    datasets/SGCC/processed_large/data/{CONS_NO}.npy  -> (1034, 1) float32, raw kWh
    datasets/SGCC/processed_large/file_labels.csv
    datasets/SGCC/processed_large/train_list.txt       -> normal only (untuk training)
    datasets/SGCC/processed_large/val_list.txt         -> held-out (normal + fraud)
    datasets/SGCC/processed_large/test_list.txt        -> held-out (normal + fraud)

Estimasi waktu pipeline keseluruhan (CPU, 2000 consumers):
    Preprocess : ~5 menit
    Pretext    : ~10 menit (5 epoch x 1463 train consumers x 48 windows)
    k-NN       : ~5 menit
    Klasifikasi: ~25 menit (15 epoch x con_data ~140K)
    Evaluate   : ~2 menit
    Total      : ~47 menit  (<1 jam OK)

Cara pakai:
    python preprocess_sgcc_large.py
    python preprocess_sgcc_large.py --n_consumers 2000 --train_frac 0.8 --seed 42
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
META_COLS = {'CONS_NO', 'FLAG'}


# ---------------------------------------------
# Timing helper
# ---------------------------------------------
def _now():
    return datetime.datetime.now()


def _log_timing(stage, t0, t1=None):
    if t1 is None:
        t1 = _now()
    elapsed = (t1 - t0).total_seconds()
    print(f"  [TIMING] {stage}")
    print(f"           starttime  = {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"           endtime    = {t1.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"           totaltime  = {elapsed:.1f}s  ({elapsed/60:.2f} menit)")
    return t1


# ---------------------------------------------
# Tee (stdout -> terminal + log file)
# ---------------------------------------------
class _Tee:
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


# ---------------------------------------------
# IO helpers
# ---------------------------------------------
def _load_raw_csv(raw_zip: Path) -> pd.DataFrame:
    """Baca SGCC large CSV dari ZIP in-memory."""
    with zipfile.ZipFile(raw_zip) as zf:
        names = [n for n in zf.namelist() if n.endswith('.csv')]
        if not names:
            raise FileNotFoundError(f"Tidak ada .csv di dalam {raw_zip}")
        print(f"  Membaca {names[0]} dari {raw_zip.name} ...")
        with zf.open(names[0]) as f:
            df = pd.read_csv(f)
    for col in ('CONS_NO', 'FLAG'):
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan.")
    return df


def _fill_nan(row_series: pd.Series) -> np.ndarray:
    """Interpolasi linear + forward/backward fill. Tidak ada normalisasi."""
    s = row_series.copy().astype(float)
    s = s.interpolate(method='linear', limit_direction='both')
    s = s.ffill().bfill().fillna(0.0)
    return s.to_numpy(dtype=np.float32)


# ---------------------------------------------
# Stratified sampling
# ---------------------------------------------
def _stratified_sample(df: pd.DataFrame, n_target: int, seed: int):
    """
    Pilih n_target consumer dengan menjaga rasio fraud/normal.
    Kembalikan subset DataFrame yang sudah di-shuffle.
    """
    normal_idx  = df.index[df['FLAG'] == 0].tolist()
    fraud_idx   = df.index[df['FLAG'] == 1].tolist()

    total       = len(df)
    fraud_ratio = len(fraud_idx) / total

    n_fraud  = min(max(1, round(n_target * fraud_ratio)), len(fraud_idx))
    n_normal = min(n_target - n_fraud, len(normal_idx))
    n_fraud  = min(n_fraud, n_target - n_normal)   # pastikan total tidak melebihi

    rng = np.random.RandomState(seed)
    rng.shuffle(normal_idx)
    rng.shuffle(fraud_idx)

    selected = normal_idx[:n_normal] + fraud_idx[:n_fraud]
    rng.shuffle(selected)

    print(f"  Stratified sample: target={n_target}")
    print(f"    Normal  dipilih : {n_normal:,}  (dari {len(normal_idx):,})")
    print(f"    Fraud   dipilih : {n_fraud:,}  (dari {len(fraud_idx):,})")
    print(f"    Total   dipilih : {n_normal + n_fraud:,}")
    print(f"    Fraud ratio     : {n_fraud / (n_normal + n_fraud) * 100:.2f}%  "
          f"(original {fraud_ratio*100:.2f}%)")

    return df.loc[selected].reset_index(drop=True)


# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess SGCC large untuk CARLA (window=365, stride=14)")
    parser.add_argument(
        '--raw_zip',
        default=str(BASE_DIR / 'datasets' / 'SGCC' / 'raw' / 'data set.csv.zip'),
        help="Path ke ZIP SGCC large. Default: datasets/SGCC/raw/data set.csv.zip"
    )
    parser.add_argument(
        '--out_dir',
        default=str(BASE_DIR / 'datasets' / 'SGCC' / 'processed_large'),
        help="Direktori output. Default: datasets/SGCC/processed_large"
    )
    parser.add_argument(
        '--n_consumers', type=int, default=2000,
        help="Jumlah consumer yang diambil (stratified). Default: 2000"
    )
    parser.add_argument(
        '--train_frac', type=float, default=0.8,
        help="Fraksi consumer NORMAL untuk train_list.txt. Default: 0.8"
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.5,
        help="Fraksi held-out (normal sisa + fraud) untuk val vs test. Default: 0.5"
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup log
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = BASE_DIR / 'results' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'preprocess_sgcc_large_{ts_str}.txt'

    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        _run(args, ts_str, log_path)
    finally:
        tee.close()


def _run(args, ts_str, log_path):
    t_total_start = _now()

    print("=" * 65)
    print("  PREPROCESS SGCC LARGE  -  window=365, stride=14")
    print("=" * 65)
    print(f"  Run       : {ts_str}")
    print(f"  Log file  : {log_path}")
    print(f"  Raw ZIP   : {args.raw_zip}")
    print(f"  Out dir   : {args.out_dir}")
    print(f"  n_consumers (stratified) : {args.n_consumers}")
    print(f"  train_frac : {args.train_frac}")
    print(f"  val_frac   : {args.val_frac}")
    print(f"  seed       : {args.seed}")
    print()

    out_dir  = Path(args.out_dir)
    data_dir = out_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------
    # TAHAP 1: Baca CSV
    # ------------------------------------------
    print("-" * 65)
    print("  [1/4] BACA RAW CSV")
    print("-" * 65)
    t0 = _now()

    df_full = _load_raw_csv(Path(args.raw_zip))
    date_cols = [c for c in df_full.columns if c not in META_COLS]
    T = len(date_cols)
    N_full = len(df_full)

    n_fraud_full  = int((df_full['FLAG'] == 1).sum())
    n_normal_full = int((df_full['FLAG'] == 0).sum())

    print(f"\n  Dataset penuh:")
    print(f"    Total consumer  : {N_full:,}")
    print(f"    Normal (FLAG=0) : {n_normal_full:,}  ({100*n_normal_full/N_full:.1f}%)")
    print(f"    Fraud  (FLAG=1) : {n_fraud_full:,}  ({100*n_fraud_full/N_full:.1f}%)")
    print(f"    Jumlah hari (T) : {T}  ({date_cols[0]} .. {date_cols[-1]})")

    # Hitung NaN
    n_nan = int(df_full[date_cols].isna().sum().sum())
    print(f"    Total NaN       : {n_nan:,}  ({100*n_nan/(N_full*T):.2f}% dari semua nilai)")

    # Statistik kWh
    sample_vals = df_full[date_cols].values[:500].flatten()
    sample_vals = sample_vals[~np.isnan(sample_vals)]
    print(f"    kWh (sample 500 rows): min={sample_vals.min():.2f}  "
          f"max={sample_vals.max():.2f}  mean={sample_vals.mean():.2f}")

    _log_timing("Baca CSV", t0)

    # ------------------------------------------
    # TAHAP 2: Stratified sampling
    # ------------------------------------------
    print("\n" + "-" * 65)
    print("  [2/4] STRATIFIED SAMPLING")
    print("-" * 65)
    t0 = _now()

    df = _stratified_sample(df_full, args.n_consumers, args.seed)
    del df_full   # bebaskan RAM

    N = len(df)
    n_windows_per = (T - 365) // 14 + 1
    n_windows_est = int(N * 0.8 * (N / N * n_windows_per))   # rough estimate train
    print(f"\n  Windows per consumer (W=365, s=14): {n_windows_per}")
    print(f"  Estimasi train windows (~80% normal): "
          f"~{round(N*0.8*0.915)*n_windows_per:,} windows")

    _log_timing("Stratified sampling", t0)

    # ------------------------------------------
    # TAHAP 3: Simpan .npy per consumer
    # ------------------------------------------
    print("\n" + "-" * 65)
    print("  [3/4] SIMPAN .NPY PER CONSUMER")
    print("-" * 65)
    t0 = _now()
    print(f"  Menyimpan {N:,} consumer ke {data_dir} ...")

    fnames_npy = []
    label_map  = {}
    n_nan_fixed = 0

    for i, row in df.iterrows():
        cons_no = str(row['CONS_NO']).strip()
        flag    = int(row['FLAG'])
        series  = row[date_cols]

        n_missing = int(series.isna().sum())
        n_nan_fixed += n_missing

        arr = _fill_nan(series)      # (T,) float32, raw kWh
        arr = arr[:, np.newaxis]     # -> (T, 1)

        fname = f"{cons_no}.npy"
        np.save(data_dir / fname, arr)
        fnames_npy.append(fname)
        label_map[fname] = flag

        if (i + 1) % 200 == 0 or (i + 1) == N:
            print(f"    [{i+1:>4}/{N}] selesai  (NaN diisi sejauh ini: {n_nan_fixed:,})")

    print(f"  Total NaN diisi interpolasi: {n_nan_fixed:,}")
    print(f"  Shape tiap .npy: ({T}, 1)  - raw kWh, TANPA normalisasi")

    _log_timing("Simpan .npy", t0)

    # ------------------------------------------
    # TAHAP 4: file_labels.csv + split
    # ------------------------------------------
    print("\n" + "-" * 65)
    print("  [4/4] BUAT FILE LIST + SPLIT")
    print("-" * 65)
    t0 = _now()

    # file_labels.csv
    with open(out_dir / 'file_labels.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fname', 'label'])
        for fname in fnames_npy:
            writer.writerow([fname, label_map[fname]])
    print(f"  file_labels.csv: {len(fnames_npy)} entri")

    # Split
    normal_files  = [f for f in fnames_npy if label_map[f] == 0]
    fraud_files   = [f for f in fnames_npy if label_map[f] == 1]

    rng = np.random.RandomState(args.seed)
    rng.shuffle(normal_files)
    rng.shuffle(fraud_files)

    # train: train_frac dari normal saja (unsupervised pretext)
    n_train = max(1, int(round(len(normal_files) * args.train_frac)))
    n_train = min(n_train, len(normal_files) - 1)
    train_files     = normal_files[:n_train]
    holdout_normal  = normal_files[n_train:]

    # val/test: stratified dari holdout_normal + fraud
    n_val_normal = int(round(len(holdout_normal) * args.val_frac))
    n_val_fraud  = int(round(len(fraud_files)    * args.val_frac))
    val_files  = holdout_normal[:n_val_normal] + fraud_files[:n_val_fraud]
    test_files = holdout_normal[n_val_normal:] + fraud_files[n_val_fraud:]

    for list_fname, lst in [('train_list.txt', train_files),
                             ('val_list.txt',   val_files),
                             ('test_list.txt',  test_files)]:
        with open(out_dir / list_fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lst) + ('\n' if lst else ''))

    n_val_fraud_count  = sum(1 for f in val_files  if label_map[f] == 1)
    n_test_fraud_count = sum(1 for f in test_files if label_map[f] == 1)

    print(f"\n  Split (seed={args.seed}):")
    print(f"    train_list.txt : {len(train_files):>5} consumer normal  "
          f"-> dipool jadi 1 model")
    print(f"    val_list.txt   : {len(val_files):>5} consumer  "
          f"({len(val_files)-n_val_fraud_count} normal + {n_val_fraud_count} fraud)  "
          f"-> pilih checkpoint terbaik")
    print(f"    test_list.txt  : {len(test_files):>5} consumer  "
          f"({len(test_files)-n_test_fraud_count} normal + {n_test_fraud_count} fraud)  "
          f"-> evaluate.py saja")

    if n_val_fraud_count == 0:
        print("  [WARNING] val_list.txt tidak punya fraud - naikkan --val_frac")
    if n_test_fraud_count == 0:
        print("  [WARNING] test_list.txt tidak punya fraud - metrik akhir tidak bermakna")

    _log_timing("Buat file list", t0)

    # ------------------------------------------
    # Ringkasan akhir
    # ------------------------------------------
    t_total_end = _now()
    print()
    print("=" * 65)
    print("  SELESAI")
    print("=" * 65)
    _log_timing("TOTAL PREPROCESS", t_total_start, t_total_end)

    print()
    print("  Konfigurasi yang diperlukan di YAML:")
    print(f"    window_size  : 365")
    print(f"    stride       : 14")
    print(f"    Windows/consumer        : {n_windows_per}")
    print(f"    Train consumers (normal): {len(train_files)}")
    print(f"    Train windows approx    : {len(train_files) * n_windows_per:,}")
    print()
    print(f"  Langkah selanjutnya:")
    print(f"    python run_sgcc_large.py")
    print(f"    python evaluate.py --dataset SGCC_LARGE")
    print(f"  Log: {log_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
