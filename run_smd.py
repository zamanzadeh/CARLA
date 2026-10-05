"""
run_smd.py — Jalankan pipeline CARLA pada dataset SMD

Machine list dibaca otomatis dari folder datasets/SMD/train/.

Cara pakai:
  python run_smd.py                                    # semua 28 machine
  python run_smd.py --machines machine-1-1.txt         # satu machine
  python run_smd.py --machines machine-1-1.txt machine-2-3.txt  # beberapa
"""

import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from utils.dataset_channels import get_smd_machines

CONFIG_ENV = BASE_DIR / "configs" / "env.yml"
CONFIG_PRE = BASE_DIR / "configs" / "pretext"        / "carla_pretext_smd.yml"
CONFIG_CLS = BASE_DIR / "configs" / "classification" / "carla_classification_smd.yml"


def run_stage(script: str, config_exp: Path, machine: str) -> bool:
    cmd = [
        sys.executable, str(BASE_DIR / script),
        "--config_env", str(CONFIG_ENV),
        "--config_exp", str(config_exp),
        "--fname",      machine,
    ]
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run CARLA on SMD dataset")
    parser.add_argument(
        "--machines", nargs="+", default=None, metavar="MACHINE",
        help="Nama file machine (misal: machine-1-1.txt). "
             "Default: semua machine dari folder train/.",
    )
    args = parser.parse_args()

    all_machines = get_smd_machines()

    if args.machines:
        invalid = [m for m in args.machines if m not in all_machines]
        if invalid:
            print(f"[WARNING] Machine tidak ditemukan di folder train/: {invalid}")
        machines = [m for m in args.machines if m in all_machines]
    else:
        machines = all_machines

    if not machines:
        print("[ERROR] Tidak ada machine valid untuk diproses.")
        sys.exit(1)

    print(f"\nDataset  : SMD")
    print(f"Machines : {len(machines)} dari {len(all_machines)} total")
    print(f"List     : {machines}\n")

    failed = []
    for i, machine in enumerate(machines, 1):
        print("=" * 60)
        print(f"  [{i}/{len(machines)}] Machine: {machine}")
        print("=" * 60)

        print("\n--- [1/2] Pretext stage ---")
        if not run_stage("carla_pretext.py", CONFIG_PRE, machine):
            print(f"[ERROR] Pretext gagal untuk {machine}. Dilewati.\n")
            failed.append((machine, "pretext")); continue

        print("\n--- [2/2] Classification stage ---")
        if not run_stage("carla_classification.py", CONFIG_CLS, machine):
            print(f"[ERROR] Classification gagal untuk {machine}.\n")
            failed.append((machine, "classification"))

    print("\n" + "=" * 60)
    print(f"  Selesai. {len(machines)-len(failed)}/{len(machines)} machine berhasil.")
    if failed:
        print("\n  Gagal:")
        for m, stage in failed:
            print(f"    - {m} (stage: {stage})")
    print(f"\n  Hasil: {BASE_DIR / 'results' / 'smd'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
