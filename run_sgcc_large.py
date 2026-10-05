"""
run_sgcc_large.py - Pipeline CARLA untuk dataset SGCC besar (42K consumers, 1034 hari)
dengan window_size=365, stride=14. Seluruh pipeline ditargetkan selesai < 1 jam.

Tahap:
  [0] Preprocessing   - preprocess_sgcc_large.py   (~5 menit)
  [1] Pretext         - carla_pretext.py            (~10 menit, 5 epoch)
  [2] Classification  - carla_classification.py     (~25 menit, 15 epoch)
  [3] Evaluate        - evaluate.py                 (~2 menit)

Log dengan starttime, endtime, totaltime dicetak di setiap tahap.

Cara pakai:
  python run_sgcc_large.py
  python run_sgcc_large.py --skip_preprocess   # kalau processed_large/ sudah ada
  python run_sgcc_large.py --n_consumers 1500  # kurangi ukuran untuk < 40 menit
"""

import sys
import subprocess
import yaml
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

CONFIG_ENV  = BASE_DIR / "configs" / "env.yml"
CONFIG_PRE  = BASE_DIR / "configs" / "pretext"        / "carla_pretext_sgcc_large.yml"
CONFIG_CLS  = BASE_DIR / "configs" / "classification" / "carla_classification_sgcc_large.yml"
PROCESSED_LARGE = BASE_DIR / "datasets" / "SGCC" / "processed_large"


# ---------------------------------------------
# Helpers
# ---------------------------------------------
class _Tee:
    """Duplikasi stdout ke terminal + file log."""
    def __init__(self, filepath):
        self._file   = open(filepath, 'w', encoding='utf-8')
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


def _now():
    return datetime.datetime.now()


def _log_timing(stage: str, t0: datetime.datetime, t1=None):
    if t1 is None:
        t1 = _now()
    elapsed = (t1 - t0).total_seconds()
    print(f"\n  +- TIMING: {stage}")
    print(f"  |  starttime = {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  |  endtime   = {t1.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  +- totaltime = {elapsed:.1f}s  ({elapsed/60:.2f} menit)")
    return elapsed


def _dir_fname(config_exp: Path) -> str:
    cfg = yaml.safe_load(open(config_exp))
    anomaly_type = cfg.get('anomaly_kwargs', {}).get('type', 'subseq')
    suffix = '' if anomaly_type in (None, 'subseq') else f'_{anomaly_type}'
    return 'All' + suffix


def run_stage(script: str, config_exp: Path, extra_args: list = None) -> bool:
    """
    Jalankan satu stage sebagai subprocess.
    Gunakan PYTHONUNBUFFERED=1 agar output child process tidak di-buffer.
    """
    cmd = [
        sys.executable, "-u", str(BASE_DIR / script),
        "--config_env", str(CONFIG_ENV),
        "--config_exp", str(config_exp),
        "--fname",      "All",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print("  CMD:", " ".join(cmd))
    import os
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(BASE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    return proc.returncode == 0


def run_preprocess(n_consumers: int, seed: int) -> bool:
    cmd = [
        sys.executable, "-u",
        str(BASE_DIR / "preprocess_sgcc_large.py"),
        "--n_consumers", str(n_consumers),
        "--seed", str(seed),
    ]
    print("  CMD:", " ".join(cmd))
    import os
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(BASE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    return proc.returncode == 0


def run_evaluate(agg: str = 'mean') -> bool:
    cmd = [
        sys.executable, "-u",
        str(BASE_DIR / "evaluate.py"),
        "--dataset", "SGCC_LARGE",
        "--agg", agg,
    ]
    print("  CMD:", " ".join(cmd))
    import os
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(BASE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    return proc.returncode == 0


# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline CARLA SGCC LARGE (<1 jam)")
    parser.add_argument('--skip_preprocess', action='store_true',
                        help="Lewati preprocess jika processed_large/ sudah ada")
    parser.add_argument('--n_consumers', type=int, default=2000,
                        help="Jumlah consumer yang disampling (default: 2000)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--agg', default='mean', choices=['mean', 'max', 'pct'],
                        help="Agregasi window->consumer di evaluate (default: mean)")
    args = parser.parse_args()

    dir_fname = _dir_fname(CONFIG_CLS)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir  = BASE_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_sgcc_large_{dir_fname}_{ts}.txt"

    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        _main(args, dir_fname, ts, log_path)
    finally:
        tee.close()


def _main(args, dir_fname, ts, log_path):
    t_pipeline_start = _now()

    print("=" * 65)
    print("  CARLA PIPELINE - SGCC LARGE")
    print("  window_size=365  stride=14  target <1 jam")
    print("=" * 65)
    print(f"  Run             : {ts}")
    print(f"  Log file        : {log_path}")
    print(f"  Checkpoint dir  : results/sgcc_large/{dir_fname}/")
    print(f"  n_consumers     : {args.n_consumers}")
    print(f"  skip_preprocess : {args.skip_preprocess}")
    print(f"  seed            : {args.seed}")

    timings = {}

    # --------------------------------------------------------------
    # [0] PREPROCESS
    # --------------------------------------------------------------
    if args.skip_preprocess and (PROCESSED_LARGE / 'train_list.txt').exists():
        print("\n" + "=" * 65)
        print("  [0/3] PREPROCESS - DILEWATI (processed_large/ sudah ada)")
        print("=" * 65)
        n_train = len(open(PROCESSED_LARGE / 'train_list.txt').read().strip().split('\n'))
        n_val   = len(open(PROCESSED_LARGE / 'val_list.txt').read().strip().split('\n'))
        n_test  = len(open(PROCESSED_LARGE / 'test_list.txt').read().strip().split('\n'))
        print(f"  Existing split: train={n_train}  val={n_val}  test={n_test}")
        timings['preprocess'] = 0.0
    else:
        print("\n" + "=" * 65)
        print("  [0/3] PREPROCESS")
        print("=" * 65)
        t0 = _now()
        ok = run_preprocess(n_consumers=args.n_consumers, seed=args.seed)
        t1 = _now()
        timings['preprocess'] = _log_timing("Preprocess", t0, t1)
        if not ok:
            print("[ERROR] Preprocess gagal.")
            sys.exit(1)

    # --------------------------------------------------------------
    # [1] PRETEXT
    # --------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  [1/3] PRETEXT  (carla_pretext.py)")
    print("=" * 65)
    t0 = _now()
    ok = run_stage("carla_pretext.py", CONFIG_PRE)
    t1 = _now()
    timings['pretext'] = _log_timing("Pretext", t0, t1)
    if not ok:
        print("[ERROR] Pretext stage gagal.")
        sys.exit(1)

    # --------------------------------------------------------------
    # [2] CLASSIFICATION
    # --------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  [2/3] CLASSIFICATION  (carla_classification.py)")
    print("=" * 65)
    t0 = _now()
    ok = run_stage("carla_classification.py", CONFIG_CLS)
    t1 = _now()
    timings['classification'] = _log_timing("Classification", t0, t1)
    if not ok:
        print("[ERROR] Classification stage gagal.")
        sys.exit(1)

    # --------------------------------------------------------------
    # [3] EVALUATE
    # --------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  [3/3] EVALUATE  (evaluate.py)")
    print("=" * 65)
    t0 = _now()
    ok = run_evaluate(agg=args.agg)
    t1 = _now()
    timings['evaluate'] = _log_timing("Evaluate", t0, t1)
    if not ok:
        print("[WARNING] evaluate.py selesai dengan error (cek output di atas).")

    # --------------------------------------------------------------
    # Ringkasan timing
    # --------------------------------------------------------------
    t_pipeline_end = _now()
    total_sec = (t_pipeline_end - t_pipeline_start).total_seconds()

    print("\n" + "=" * 65)
    print("  RINGKASAN TIMING PIPELINE")
    print("=" * 65)
    print(f"  {'Stage':<20} {'Waktu (s)':>10}  {'Menit':>8}")
    print(f"  {'-'*20}  {'-'*9}  {'-'*7}")
    for stage, sec in timings.items():
        print(f"  {stage:<20} {sec:>10.1f}  {sec/60:>7.2f}")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<20} {total_sec:>10.1f}  {total_sec/60:>7.2f}")
    print()
    status = "OK DALAM TARGET" if total_sec / 60 < 60 else "FAIL MELEBIHI 1 JAM"
    print(f"  Target <60 menit: {status}  ({total_sec/60:.1f} menit)")
    print(f"\n  Model tersimpan di : results/sgcc_large/{dir_fname}/")
    print(f"  Log tersimpan di   : {log_path}")
    print(f"\n  Untuk re-evaluasi  : python evaluate.py --dataset SGCC_LARGE")
    print("=" * 65)


if __name__ == "__main__":
    main()
