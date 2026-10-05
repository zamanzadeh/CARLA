"""
run_smap.py — Jalankan pipeline CARLA pada dataset SMAP

Channel list dibaca dari labeled_anomalies.csv (spacecraft='SMAP').

Cara pakai:
  python run_smap.py                        # semua channel SMAP
  python run_smap.py --channels A-1         # satu channel
  python run_smap.py --channels A-1 E-3    # beberapa channel
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from utils.dataset_channels import get_channels, get_spacecraft

CONFIG_ENV = BASE_DIR / "configs" / "env.yml"
CONFIG_PRE = BASE_DIR / "configs" / "pretext"        / "carla_pretext_smap.yml"
CONFIG_CLS = BASE_DIR / "configs" / "classification" / "carla_classification_smap.yml"
SPACECRAFT = "SMAP"


def run_stage(script: str, config_exp: Path, channel: str) -> bool:
    cmd = [
        sys.executable, str(BASE_DIR / script),
        "--config_env", str(CONFIG_ENV),
        "--config_exp", str(config_exp),
        "--fname",      channel,
    ]
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=f"Run CARLA on {SPACECRAFT} dataset")
    parser.add_argument(
        "--channels", nargs="+", default=None, metavar="CHANNEL",
        help="Channel yang akan diproses. Default: semua channel SMAP dari CSV.",
    )
    args = parser.parse_args()

    # -- Ambil daftar channel dari CSV --
    all_smap = get_channels(SPACECRAFT)

    if args.channels:
        invalid = []
        channels = []
        for c in args.channels:
            try:
                sc = get_spacecraft(c)
            except ValueError:
                invalid.append(c); continue
            if sc != SPACECRAFT:
                print(f"[WARNING] '{c}' adalah channel {sc}, bukan {SPACECRAFT}. Dilewati.")
            else:
                channels.append(c)
        if invalid:
            print(f"[WARNING] Channel tidak dikenal di CSV: {invalid}")
    else:
        channels = all_smap

    if not channels:
        print("[ERROR] Tidak ada channel valid untuk diproses.")
        sys.exit(1)

    print(f"\nDataset  : {SPACECRAFT}")
    print(f"Channels : {len(channels)} dari {len(all_smap)} total")
    print(f"List     : {channels}\n")

    failed = []
    for i, ch in enumerate(channels, 1):
        print("=" * 60)
        print(f"  [{i}/{len(channels)}] Channel: {ch}")
        print("=" * 60)

        print("\n--- [1/2] Pretext stage ---")
        if not run_stage("carla_pretext.py", CONFIG_PRE, ch):
            print(f"[ERROR] Pretext gagal untuk {ch}. Dilewati.\n")
            failed.append((ch, "pretext")); continue

        print("\n--- [2/2] Classification stage ---")
        if not run_stage("carla_classification.py", CONFIG_CLS, ch):
            print(f"[ERROR] Classification gagal untuk {ch}.\n")
            failed.append((ch, "classification"))

    print("\n" + "=" * 60)
    print(f"  Selesai. {len(channels)-len(failed)}/{len(channels)} channel berhasil.")
    if failed:
        print("\n  Gagal:")
        for ch, stage in failed:
            print(f"    - {ch} (stage: {stage})")
    print(f"\n  Hasil: {BASE_DIR / 'results' / SPACECRAFT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
