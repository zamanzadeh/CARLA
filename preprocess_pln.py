"""
preprocess_pln.py - Template preprocessing data PLN untuk CARLA.

Asumsi format raw data (SESUAIKAN dengan skema CSV PLN Anda yang sebenarnya):

    datasets/PLN/raw/{fname}.csv
        Satu file = satu load profile (satu pelanggan/feeder/trafo), dengan:
          - 1 kolom timestamp (interval 15 menit) - akan di-drop otomatis kalau
            namanya match salah satu di TIMESTAMP_COL_CANDIDATES
          - N kolom numerik (kVA, kWh, arus, tegangan, dst.)

    datasets/PLN/raw/labels.csv
        Kolom: fname,label
          fname  -> nama file TANPA ekstensi .csv
          label  -> 0 = normal, 1 = anomali (untuk SELURUH file itu)

Output (siap dipakai run_pln.py / evaluate.py --dataset PLN):

    datasets/PLN/processed/data/{fname}.npy   (T, n_features) float32
    datasets/PLN/processed/file_labels.csv
    datasets/PLN/processed/train_list.txt     (file NORMAL, dipool untuk training)
    datasets/PLN/processed/val_list.txt       (held-out, normal+anomali - dipakai HANYA
                                               untuk memilih checkpoint terbaik selama
                                               training, carla_pretext.py/carla_classification.py)
    datasets/PLN/processed/test_list.txt      (held-out, normal+anomali - dipakai HANYA oleh
                                               evaluate.py di akhir, TIDAK PERNAH dilihat
                                               selama training, supaya metrik akhir tidak
                                               bias optimis/data leakage)

train/val/test dipisah TOTAL (tidak ada file yang sama di lebih dari satu list):
  1. train_frac dari file NORMAL -> train_list.txt.
  2. Sisa file normal + SEMUA file anomali dibagi lagi jadi val/test lewat
     --val_frac (proporsional terhadap masing-masing kelas, supaya val dan
     test sama-sama punya campuran normal+anomali).

Cara pakai:
    python preprocess_pln.py
    python preprocess_pln.py --raw_dir datasets/PLN/raw --train_frac 0.8 --val_frac 0.5
"""

import os
import csv
import sys
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()


class _Tee:
    """
    Redirect stdout supaya setiap print() tetap tampil di terminal SEKALIGUS
    ditulis ke file log. Dipakai agar output preprocessing (shape per file,
    warning NaN/kolom non-numerik, hasil train/test split) tersimpan permanen
    ke file .txt dengan nama UNIK (timestamp), supaya bisa dipakai untuk
    dokumentasi/evaluasi nanti (jejak persis data apa yang masuk ke training).
    """
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

TIMESTAMP_COL_CANDIDATES = {'timestamp', 'datetime', 'date', 'waktu', 'time', 'tanggal'}

# Sel yang dianggap "kosong"/missing selain NaN bawaan pandas
MISSING_MARKERS = {'', '-', '--', 'na', 'n/a', 'null', 'none', '?','#value!'}


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Cast satu kolom ke numerik dengan toleran:
      - Kalau sudah numerik, kembalikan langsung.
      - Kalau string: trim spasi, anggap marker umum (NA/-/null/dst) sebagai missing,
        lalu deteksi format desimal koma ala Indonesia/Eropa:
          "1.234,56" -> '.' = pemisah ribuan, ',' = desimal  -> 1234.56
          "1234,56"  -> ',' = desimal                         -> 1234.56
          "1234.56"  -> sudah format titik standar            -> 1234.56
      - Sisanya yang masih gagal di-parse -> NaN (dihitung & dilaporkan, bukan didiamkan).
    """
    if pd.api.types.is_numeric_dtype(s):
        return s

    def _norm(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        if v.lower() in MISSING_MARKERS:
            return np.nan
        has_comma, has_dot = ',' in v, '.' in v
        if has_comma and has_dot:
            v = v.replace('.', '').replace(',', '.')
        elif has_comma and not has_dot:
            v = v.replace(',', '.')
        return v

    return pd.to_numeric(s.map(_norm), errors='coerce')


def load_one_csv(csv_path: Path, verbose: bool = True) -> np.ndarray:
    """
    Baca satu file CSV load profile -> matrix numerik (T, n_features), dengan:
      - cast string->numerik (termasuk format desimal koma ID/Eropa)
      - kolom yang TETAP tidak bisa dijadikan numerik (semua nilainya gagal di-parse,
        misal kolom teks/ID) di-drop dengan warning, BUKAN didiamkan
      - missing value diisi pakai interpolasi linear di sumbu waktu (limit_direction='both'
        biar ujung depan/belakang yang kosong juga terisi), baru fallback ke mean kolom
        kalau masih ada NaN setelah interpolasi
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)

    drop_cols = [c for c in df.columns if c.strip().lower() in TIMESTAMP_COL_CANDIDATES]
    df = df.drop(columns=drop_cols, errors='ignore')

    numeric_cols, dropped_cols = {}, []
    for col in df.columns:
        cleaned = _clean_numeric_series(df[col])
        if cleaned.notna().sum() == 0:
            dropped_cols.append(col)
            continue
        numeric_cols[col] = cleaned

    if not numeric_cols:
        raise ValueError(
            f"{csv_path.name}: tidak ada kolom yang bisa dijadikan numerik setelah drop "
            f"timestamp. Cek TIMESTAMP_COL_CANDIDATES / skema kolom / format desimal."
        )
    if dropped_cols and verbose:
        print(f"  [WARNING] {csv_path.name}: kolom non-numerik dibuang: {dropped_cols}")

    out = pd.DataFrame(numeric_cols)

    n_missing = int(out.isna().sum().sum())
    if n_missing > 0:
        if verbose:
            print(f"  [INFO] {csv_path.name}: {n_missing} nilai missing -> diinterpolasi (linear)")
        out = out.interpolate(method='linear', limit_direction='both', axis=0)
        still_missing = out.isna().sum()
        for col in out.columns[still_missing > 0]:
            out[col] = out[col].fillna(out[col].mean())
        out = out.fillna(0.0)  # fallback terakhir kalau satu kolom 100% NaN (seharusnya sudah didrop di atas)

    return out.to_numpy(dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Preprocess data PLN untuk CARLA")
    parser.add_argument('--raw_dir', default=str(BASE_DIR / 'datasets' / 'PLN' / 'raw'))
    parser.add_argument('--out_dir', default=str(BASE_DIR / 'datasets' / 'PLN' / 'processed'))
    parser.add_argument('--labels_csv', default=None,
                        help="Default: {raw_dir}/labels.csv")
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help="Fraksi file NORMAL yang dipakai untuk training pooled "
                             "(sisanya, dicampur semua file anomali, dibagi lagi jadi "
                             "val_list.txt + test_list.txt lewat --val_frac)")
    parser.add_argument('--val_frac', type=float, default=0.5,
                        help="Dari file held-out (normal sisa + semua anomali), fraksi "
                             "yang masuk val_list.txt - dipakai HANYA untuk memilih "
                             "checkpoint terbaik selama training (BUKAN untuk training "
                             "itu sendiri). Sisanya masuk test_list.txt, dipakai HANYA "
                             "oleh evaluate.py di akhir - tidak pernah dilihat selama "
                             "training, supaya metrik akhir tidak bias optimis.")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = BASE_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Nama file UNIK (timestamp) - tidak menimpa log preprocessing run sebelumnya,
    # jadi tiap run punya jejak tertulis sendiri (file mana yang diproses, shape,
    # warning NaN/kolom non-numerik, hasil train/test split) untuk evaluasi nanti.
    log_path = log_dir / f"preprocess_pln_{ts}.txt"

    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        _run_preprocess(args, ts, log_path)
    finally:
        tee.close()


def _run_preprocess(args, ts, log_path):
    print(f"Run        : {ts}")
    print(f"Log file   : {log_path}")
    print(f"Raw dir    : {args.raw_dir}")
    print(f"Out dir    : {args.out_dir}\n")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    data_dir = out_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = Path(args.labels_csv) if args.labels_csv else raw_dir / 'labels.csv'
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv tidak ditemukan: {labels_csv}")

    labels_df = pd.read_csv(labels_csv)
    label_map = {str(r['fname']).strip(): int(r['label']) for _, r in labels_df.iterrows()}

    fnames_npy = []
    for fname, label in label_map.items():
        csv_path = raw_dir / f"{fname}.csv"
        if not csv_path.exists():
            print(f"[WARNING] {csv_path} tidak ditemukan, dilewati")
            continue
        arr = load_one_csv(csv_path)
        if np.isnan(arr).any():
            n_nan = int(np.isnan(arr).sum())
            print(f"[WARNING] {fname}: {n_nan} NaN diganti 0")
            arr = np.nan_to_num(arr)
        npy_name = f"{fname}.npy"
        np.save(data_dir / npy_name, arr.astype(np.float32))
        fnames_npy.append(npy_name)
        print(f"  {fname}: shape={arr.shape} label={label}")

    if not fnames_npy:
        raise RuntimeError("Tidak ada file yang berhasil diproses.")

    # file_labels.csv (pakai nama .npy, dipakai oleh data/PLN.py)
    with open(out_dir / 'file_labels.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fname', 'label'])
        for fname in fnames_npy:
            base = fname[:-4]
            writer.writerow([fname, label_map[base]])

    normal_files = [f for f in fnames_npy if label_map[f[:-4]] == 0]
    anomaly_files = [f for f in fnames_npy if label_map[f[:-4]] == 1]

    if not normal_files:
        raise RuntimeError("Tidak ada file berlabel normal (0) - butuh minimal beberapa "
                            "file normal untuk training pooled.")

    rng = np.random.RandomState(args.seed)
    normal_files = list(normal_files)
    anomaly_files = list(anomaly_files)
    rng.shuffle(normal_files)
    rng.shuffle(anomaly_files)

    # 1) train_frac dari NORMAL -> train_list.txt (dipool untuk training, TIDAK
    #    pernah masuk val/test - model tidak boleh "melihat" file ini lagi).
    n_train = max(1, int(round(len(normal_files) * args.train_frac)))
    n_train = min(n_train, len(normal_files) - 1) if len(normal_files) > 1 else len(normal_files)
    train_files = normal_files[:n_train]
    holdout_normal = normal_files[n_train:]

    # 2) Sisa normal + SEMUA anomali (sama-sama belum pernah dilatih) dibagi lagi
    #    jadi val/test lewat val_frac, masing-masing KELAS dibagi proporsional
    #    (bukan digabung dulu baru dibagi) supaya val dan test sama-sama dapat
    #    campuran normal+anomali, bukan satu kelas doang di salah satu split.
    n_val_normal = int(round(len(holdout_normal) * args.val_frac))
    n_val_anomaly = int(round(len(anomaly_files) * args.val_frac))
    val_files = holdout_normal[:n_val_normal] + anomaly_files[:n_val_anomaly]
    test_files = holdout_normal[n_val_normal:] + anomaly_files[n_val_anomaly:]

    with open(out_dir / 'train_list.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_files) + ('\n' if train_files else ''))
    with open(out_dir / 'val_list.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_files) + ('\n' if val_files else ''))
    with open(out_dir / 'test_list.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_files) + ('\n' if test_files else ''))

    n_val_anomaly_actual = sum(1 for f in val_files if label_map[f[:-4]] == 1)
    n_test_anomaly_actual = sum(1 for f in test_files if label_map[f[:-4]] == 1)

    print(f"\nSelesai. {len(fnames_npy)} file diproses ({len(normal_files)} normal, "
          f"{len(anomaly_files)} anomali).")
    print(f"  train_list.txt : {len(train_files)} file (normal, dipool untuk training - "
          f"TIDAK PERNAH dipakai utk val/test)")
    print(f"  val_list.txt   : {len(val_files)} file ({len(val_files) - n_val_anomaly_actual} "
          f"normal + {n_val_anomaly_actual} anomali) - HANYA dipakai untuk memilih "
          f"checkpoint terbaik selama training")
    print(f"  test_list.txt  : {len(test_files)} file ({len(test_files) - n_test_anomaly_actual} "
          f"normal + {n_test_anomaly_actual} anomali) - HANYA dipakai evaluate.py di "
          f"akhir, tidak pernah dilihat selama training")
    if val_files and n_val_anomaly_actual == 0:
        print("  [WARNING] val_list.txt tidak punya file anomali - pemilihan checkpoint "
              "terbaik (best_f1) selama training bisa tidak bermakna. Pertimbangkan "
              "--val_frac lebih besar atau tambah data anomali.")
    if test_files and n_test_anomaly_actual == 0:
        print("  [WARNING] test_list.txt tidak punya file anomali - precision/recall/F1/AUROC "
              "akhir bisa tidak bermakna. Pertimbangkan --val_frac lebih kecil atau tambah "
              "data anomali.")
    print(f"\n  Jangan lupa: set 'in_channels' di configs/pretext/carla_pretext_pln.yml dan "
          f"configs/classification/carla_classification_pln.yml = {load_one_csv(raw_dir / (fnames_npy[0][:-4] + '.csv')).shape[1]} "
          f"(jumlah kolom fitur numerik).")
    print(f"\n  Log preprocessing ini tersimpan di: {log_path}")


if __name__ == '__main__':
    main()
