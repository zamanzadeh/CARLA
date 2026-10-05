"""
run_pln.py - Jalankan pipeline CARLA pada dataset PLN (label per FILE, bukan per timestamp)

Beda dengan run_lead.py/run_smd.py: PLN TIDAK melatih satu model per file. Karena label
hanya tersedia di level file, model harus belajar "normal" dari KUMPULAN file normal
(train_list.txt), lalu selama training memilih checkpoint terbaik berdasar val_list.txt,
dan akhirnya dievaluasi di level file pada kumpulan file held-out test_list.txt (TERPISAH
dari val_list.txt, supaya metrik akhir tidak bias optimis/data leakage). Jadi pipeline ini
hanya melatih SATU model bersama (fname='All').

Pastikan sebelum menjalankan ini:
  datasets/PLN/processed/
      data/{fname}.npy   -> (T, n_features) per file
      file_labels.csv    -> fname,label (0=normal, 1=anomali)
      train_list.txt     -> daftar file NORMAL untuk training (pooled)
      val_list.txt       -> daftar file held-out untuk memilih checkpoint terbaik
      test_list.txt      -> daftar file held-out (terpisah dari val) untuk evaluasi akhir

Cara pakai:
  python run_pln.py
"""

import sys
import subprocess
import yaml
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from utils.dataset_channels import get_pln_train_files, get_pln_val_files, get_pln_test_files

CONFIG_ENV = BASE_DIR / "configs" / "env.yml"
CONFIG_PRE = BASE_DIR / "configs" / "pretext"        / "carla_pretext_pln.yml"
CONFIG_CLS = BASE_DIR / "configs" / "classification" / "carla_classification_pln.yml"


class _Tee:
    """
    Redirect stdout supaya setiap print() tetap tampil di terminal SEKALIGUS
    ditulis ke file log. File log dibuat dengan nama UNIK (timestamp + tipe
    anomali) agar setiap run preprocessing/training punya jejak tertulis yang
    bisa dipakai untuk evaluasi/dokumentasi/perbandingan nanti (tidak tertimpa
    run lain, dan langsung terlihat run mana yang menghasilkan checkpoint mana).
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


def _dir_fname(config_exp: Path, fname: str = "All") -> str:
    """
    Samakan dgn logic suffix di utils/config.py::create_config(): kalau
    anomaly_kwargs.type diisi selain 'subseq'/kosong (mis. 'theft'), checkpoint
    disimpan di folder '{fname}_{type}', BUKAN '{fname}', supaya tidak menimpa
    checkpoint dari tipe anomali lain. Dipakai di sini hanya untuk menampilkan
    path yang BENAR ke user setelah training selesai.
    """
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
    # Stream output subprocess baris-per-baris lewat print() (bukan biarkan
    # subprocess inherit stdout langsung) supaya ikut tertangkap oleh _Tee saat
    # sys.stdout sedang diredirect ke file log - jadi log epoch/loss training
    # carla_pretext.py & carla_classification.py juga tersimpan, bukan hanya
    # pesan orkestrasi run_pln.py ini.
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
    # Nama file UNIK: timestamp + tag tipe anomali (sama dengan suffix folder
    # checkpoint, mis. 'All_theft'/'All') - jadi log run ini langsung tertaut
    # ke folder checkpoint mana yang dihasilkannya, untuk dipakai evaluasi.
    log_path = log_dir / f"run_pln_{dir_fname}_{ts}.txt"

    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        train_files = get_pln_train_files()
        val_files = get_pln_val_files()
        test_files = get_pln_test_files()

        print(f"Run             : {ts}")
        print(f"Checkpoint dir  : results/pln/{dir_fname}/")
        print(f"Log file        : {log_path}")
        print(f"\nDataset      : PLN")
        print(f"Train files  : {len(train_files)} (normal, dipool jadi 1 model)")
        print(f"Val files    : {len(val_files)} (held-out, untuk pilih checkpoint terbaik)")
        print(f"Test files   : {len(test_files)} (held-out TERPISAH dari val, untuk evaluasi akhir)\n")

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
        print(f"  Selesai. Model tersimpan di results/pln/{dir_fname}/")
        if dir_fname != "All":
            print(f"  (anomaly_kwargs.type != 'subseq' -> folder terpisah dari baseline 'All',")
            print(f"   jadi tidak menimpa checkpoint baseline kalau pernah dilatih sebelumnya.)")
        print("  Jalankan: python evaluate.py --dataset PLN")
        print(f"  Log run ini tersimpan di: {log_path}")
        print("=" * 60)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
