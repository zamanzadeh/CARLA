"""
run_lead.py — Jalankan pipeline CARLA pada dataset LEAD

Building list dibaca otomatis dari datasets/LEAD/processed/building_list.txt.

Cara pakai:
  python run_lead.py                          # semua building
  python run_lead.py --buildings 1.npy        # satu building
  python run_lead.py --buildings 1.npy 32.npy # beberapa building
"""

import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from utils.dataset_channels import get_lead_buildings

CONFIG_ENV = BASE_DIR / "configs" / "env.yml"
CONFIG_PRE = BASE_DIR / "configs" / "pretext"        / "carla_pretext_lead.yml"
CONFIG_CLS = BASE_DIR / "configs" / "classification" / "carla_classification_lead.yml"


def run_stage(script: str, config_exp: Path, building: str) -> bool:
    cmd = [
        sys.executable, str(BASE_DIR / script),
        "--config_env", str(CONFIG_ENV),
        "--config_exp", str(config_exp),
        "--fname",      building,
    ]
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run CARLA on LEAD dataset")
    parser.add_argument(
        "--buildings", nargs="+", default=None, metavar="BUILDING",
        help="Nama file building (misal: 1.npy). "
             "Default: semua building dari building_list.txt.",
    )
    args = parser.parse_args()

    all_buildings = get_lead_buildings()

    if args.buildings:
        invalid = [b for b in args.buildings if b not in all_buildings]
        if invalid:
            print(f"[WARNING] Building tidak ditemukan di building_list.txt: {invalid}")
        buildings = [b for b in args.buildings if b in all_buildings]
    else:
        buildings = all_buildings

    if not buildings:
        print("[ERROR] Tidak ada building valid untuk diproses.")
        sys.exit(1)

    print(f"\nDataset   : LEAD")
    print(f"Buildings : {len(buildings)} dari {len(all_buildings)} total")
    print(f"List      : {buildings[:10]}{'...' if len(buildings) > 10 else ''}\n")

    failed = []
    for i, building in enumerate(buildings, 1):
        print("=" * 60)
        print(f"  [{i}/{len(buildings)}] Building: {building}")
        print("=" * 60)

        print("\n--- [1/2] Pretext stage ---")
        if not run_stage("carla_pretext.py", CONFIG_PRE, building):
            print(f"[ERROR] Pretext gagal untuk {building}. Dilewati.\n")
            failed.append((building, "pretext")); continue

        print("\n--- [2/2] Classification stage ---")
        if not run_stage("carla_classification.py", CONFIG_CLS, building):
            print(f"[ERROR] Classification gagal untuk {building}.\n")
            failed.append((building, "classification"))

    print("\n" + "=" * 60)
    print(f"  Selesai. {len(buildings)-len(failed)}/{len(buildings)} building berhasil.")
    if failed:
        print("\n  Gagal:")
        for b, stage in failed:
            print(f"    - {b} (stage: {stage})")
    print("=" * 60)


if __name__ == "__main__":
    main()
