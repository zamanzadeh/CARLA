"""
run_sgcc.py — Jalankan pipeline CARLA pada dataset SGCC (State Grid Corporation of China)

Sama pola dengan run_pln.py: SGCC menggunakan label per CONSUMER (file), bukan per
timestamp, sehingga pipeline melatih SATU model bersama (fname='All') dari pool semua
consumer normal (train_list.txt). Checkpoint terbaik dipilih berdasarkan val_list.txt.
Evaluasi akhir dilakukan oleh evaluate.py terhadap test_list.txt (terpisah dari val,
sehingga metrik akhir tidak bias optimis / data leakage).

Pastikan sebelum menjalankan ini:
  1. preprocess_sgcc.py sudah dijalankan:
       python preprocess_sgcc.py
  2. datasets/SGCC/processed/ berisi:
       data/{CONS_NO}.npy   -> (26, 1) float32 per consumer, RAW kWh
       file_labels.csv      -> fname, label
       train_list.txt       -> consumer NORMAL untuk training
       val_list.txt         -> consumer held-out untuk pilih checkpoint terbaik
       test_list.txt        -> consumer held-out untuk evaluasi akhir

Cara pakai:
  python run_sgcc.py
"""

import sys
import subprocess
import yaml
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from utils.dataset_channels import get_sgcc_train_files, get_sgcc_val_files, get_sgcc_test_files

CONFIG_ENV = BASE_DIR / "configs" / "env.yml"
CONFIG_PRE = BASE_DIR / "configs" / "pretext"        / "carla_pretext_sgcc.yml"
CONFIG_CLS = BASE_DIR / "configs" / "classification" / "carla_classification_sgcc.yml"


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


def _dir_fname(config_exp: Path, fname: str = "All") -> str:
    """Suffix folder checkpoint sesuai anomaly_kwargs.type di config."""
    cfg = yaml.safe_load(open(config_exp))
    anomaly_type = cfg.get('anomaly_kwargs', {}).get('type', 'subseq')
    suffix = '' if anomaly_type in (None, 'subseq') else f'_{anomaly_type}'
    return fname + suffix


def run_stage(script: str, config_exp: Path) -> bool:
    cmd = [
        sys.executable, str(BASE_DIR / script),
        "--config_env", str(CONFIG_ENV),
        "--config_exp", str(config_exp),
        "--fname",      "All",
    ]
    print("  CMD:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=str(BASE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    return proc.returncode == 0


def main():
    dir_fname = _dir_fname(CONFIG_CLS)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = BASE_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_sgcc_{dir_fname}_{ts}.txt"

    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        train_files = get_sgcc_train_files()
        val_files   = get_sgcc_val_files()
        test_files  = get_sgcc_test_files()

        print(f"Run             : {ts}")
        print(f"Checkpoint dir  : results/sgcc/{dir_fname}/")
        print(f"Log file        : {log_path}")
        print(f"\nDataset      : SGCC")
        print(f"Train files  : {len(train_files)} consumers (normal, dipool jadi 1 model)")
        print(f"Val files    : {len(val_files)} consumers (held-out, pilih checkpoint terbaik)")
        print(f"Test files   : {len(test_files)} consumers (held-out TERPISAH, evaluasi akhir)\n")

        print("=" * 60)
        print("  [1/2] Pretext stage (shared model, fname=All)")
        print("=" * 60)
        if not run_stage("carla_pretext.py", CONFIG_PRE):
            print("[ERROR] Pretext stage gagal.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("  [2/2] Classification stage (shared model, fname=All)")
        print("=" * 60)
        if not run_stage("carla_classification.py", CONFIG_CLS):
            print("[ERROR] Classification stage gagal.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print(f"  Selesai. Model tersimpan di results/sgcc/{dir_fname}/")
        print(f"  Jalankan: python evaluate.py --dataset SGCC")
        print(f"  Log run ini tersimpan di: {log_path}")
        print("=" * 60)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
