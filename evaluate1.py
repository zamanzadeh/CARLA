"""
evaluate.py - Evaluasi model CARLA pada dataset MSL, SMAP, SMD, LEAD, atau PLN
Output disimpan otomatis ke results/eval/eval_{dataset}_{ts}.txt dan .csv

PLN beda dari yang lain: labelnya per FILE (bukan per timestamp/window), jadi
evaluasinya juga di level file: skor anomali tiap window diagregasi jadi satu
skor per file, baru dibandingkan ke label file. Lihat evaluate_pln().
"""

import os
import sys
import csv
import argparse
import datetime
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))


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


from utils.config import create_config
from utils.common_config import get_model, get_val_transformations1, inject_sub_anomaly
from utils.collate import collate_custom
from utils.dataset_channels import (
    get_channels, get_spacecraft, get_smd_machines, get_lead_buildings,
    get_pln_train_files, get_pln_test_files, get_pln_file_labels,
    get_sgcc_train_files, get_sgcc_test_files,
)
from data.MSL import MSL
from data.SMD import SMD
from data.LEAD import LEAD
from data.PLN import PLN
from data.SGCC import SGCC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "MSL":  {"classification": BASE_DIR / "configs" / "classification" / "carla_classification_msl.yml"},
    "SMAP": {"classification": BASE_DIR / "configs" / "classification" / "carla_classification_smap.yml"},
    "SMD":  {"classification": BASE_DIR / "configs" / "classification" / "carla_classification_smd.yml"},
    "LEAD": {"classification": BASE_DIR / "configs" / "classification" / "carla_classification_lead.yml"},
    "PLN":  {"classification": BASE_DIR / "configs" / "classification" / "carla_classification_pln.yml"},
    "SGCC": {"classification": BASE_DIR / "configs" / "classification" / "carla_classification_sgcc.yml"},
}


def _dir_fname(config_exp, fname: str = "All", seed=None) -> str:
    """
    Samakan dgn suffix folder checkpoint di utils/config.py::create_config() dan
    di run_pln.py: kalau anomaly_kwargs.type diisi selain 'subseq'/kosong (mis.
    'theft'), tag jadi '{fname}_{type}' (mis. 'All_theft'), bukan cuma '{fname}'
    ('All'). Kalau seed tidak None, tambahkan suffix '_sN' (mis. 'All_s0').
    Dipakai supaya nama file output evaluate.py langsung menunjukkan model/
    checkpoint mana yang dievaluasi, termasuk seed-nya.
    """
    cfg = yaml.safe_load(open(config_exp))
    anomaly_type = cfg.get('anomaly_kwargs', {}).get('type', 'subseq')
    suffix = '' if anomaly_type in (None, 'subseq') else f'_{anomaly_type}'
    seed_suffix = '' if seed is None else f'_s{seed}'
    return fname + suffix + seed_suffix


@torch.no_grad()
def _get_logits(model, ts):
    if ts.ndim == 3:
        b, w, h = ts.shape
    else:
        b, w = ts.shape; h = 1
    out = model(ts.view(b, h, w))
    return out[0] if isinstance(out, (list, tuple)) else out


@torch.no_grad()
def find_majority_label(model, dataloader):
    model.eval()
    all_preds = []
    for batch in dataloader:
        ts = batch['ts_org'].to(device)
        preds = torch.argmax(_get_logits(model, ts), dim=1).cpu().numpy()
        all_preds.append(preds)
    all_preds = np.concatenate(all_preds)
    values, counts = np.unique(all_preds, return_counts=True)
    return int(values[np.argmax(counts)])


@torch.no_grad()
def run_inference(model, dataloader, normal_label):
    model.eval()
    all_scores, all_labels = [], []
    for batch in dataloader:
        ts     = batch['ts_org'].to(device)
        labels = batch['target'].cpu().numpy()
        probs  = F.softmax(_get_logits(model, ts), dim=1).cpu().numpy()
        scores = 1.0 - probs[:, normal_label]
        all_scores.append(scores)
        all_labels.append(labels)
    return np.concatenate(all_scores), np.concatenate(all_labels)


def compute_metrics(scores, labels, threshold=None):
    prec_curve, rec_curve, thr_pr = precision_recall_curve(labels, scores)
    aupr = average_precision_score(labels, scores)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_curve = np.where(
            (prec_curve + rec_curve) > 0,
            2 * prec_curve * rec_curve / (prec_curve + rec_curve),
            0.0,
        )
    if threshold is None:
        best_idx  = int(np.argmax(f1_curve[:-1]))
        threshold = float(thr_pr[best_idx])
    binary = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary, labels=[0, 1]).ravel()
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float('nan')
    f1_anomaly       = f1_score(labels, binary, average='binary', pos_label=1, zero_division=0)
    f1_normal        = f1_score(labels, binary, average='binary', pos_label=0, zero_division=0)
    f1_macro         = f1_score(labels, binary, average='macro',  zero_division=0)
    precision_binary = precision_score(labels, binary, zero_division=0)
    recall_binary    = recall_score(labels, binary, zero_division=0)
    prec_macro       = precision_score(labels, binary, average='macro', zero_division=0)
    rec_macro        = recall_score(labels, binary, average='macro',    zero_division=0)
    return {
        'threshold':        threshold,
        'precision':        precision_binary,
        'recall':           recall_binary,
        'precision_macro':  prec_macro,
        'recall_macro':     rec_macro,
        'f1_score':         f1_anomaly,   # alias: F1 anomaly (binary, pos_label=1)
        'f1_anomaly':       f1_anomaly,
        'f1_normal':        f1_normal,
        'f1_macro':         f1_macro,
        'aupr': aupr, 'auroc': auroc,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'prec_curve': prec_curve, 'rec_curve': rec_curve,
        'f1_curve': f1_curve, 'thr_pr': thr_pr,
    }


def print_metrics(dataset, channel, m):
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Dataset : {dataset}   Channel : {channel}")
    print(sep)
    print(f"  Threshold (auto)  : {m['threshold']:.4f}")
    print(f"  Precision         : {m['precision']:.4f}  (binary, pos_label=1)")
    print(f"  Recall            : {m['recall']:.4f}  (binary, pos_label=1)")
    print(f"  Precision Macro   : {m['precision_macro']:.4f}")
    print(f"  Recall Macro      : {m['recall_macro']:.4f}")
    print(f"  F1 Anomaly        : {m['f1_anomaly']:.4f}")
    print(f"  F1 Normal         : {m['f1_normal']:.4f}")
    print(f"  F1 Macro          : {m['f1_macro']:.4f}")
    print(f"  AUPR              : {m['aupr']:.4f}")
    print(f"  AUROC             : {m['auroc']:.4f}")
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    print(sep)


def _build_datasets(dataset, fname, p):
    val_transform = get_val_transformations1(p)
    sanomaly      = inject_sub_anomaly(p)
    if dataset in ('MSL', 'SMAP'):
        train_ds = MSL(fname, train=True,  transform=val_transform, sanomaly=sanomaly)
        test_ds  = MSL(fname, train=False, transform=val_transform, sanomaly=sanomaly,
                       mean_data=train_ds.mean, std_data=train_ds.std)
    elif dataset == 'SMD':
        train_ds = SMD(fname, train=True,  transform=val_transform, sanomaly=sanomaly)
        test_ds  = SMD(fname, train=False, transform=val_transform, sanomaly=sanomaly,
                       mean_data=train_ds.mean, std_data=train_ds.std)
    elif dataset == 'LEAD':
        train_ds = LEAD(fname, train=True,  transform=val_transform, sanomaly=sanomaly)
        test_ds  = LEAD(fname, train=False, transform=val_transform, sanomaly=sanomaly,
                        mean_data=train_ds.mean, std_data=train_ds.std)
    else:
        raise ValueError(f"Dataset tidak dikenal: {dataset}")
    return train_ds, test_ds


def evaluate_channel(dataset, fname, threshold=None):
    config_env = str(BASE_DIR / "configs" / "env.yml")
    config_cls = str(CONFIGS[dataset]["classification"])
    p = create_config(config_env, config_cls, fname)

    model_path = p['classification_model']
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model belum ada: {model_path}")
        return None

    model = get_model(p)
    model = torch.nn.DataParallel(model)
    ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
    model.module.load_state_dict(ckpt['model'])
    model.to(device).eval()

    train_ds, test_ds = _build_datasets(dataset, fname, p)

    if test_ds.targets.sum() == 0:
        print(f"  [SKIP] {fname} - tidak ada anomali di test set.")
        return None

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=p['batch_size'], shuffle=False,
        num_workers=p['num_workers'], collate_fn=collate_custom, drop_last=False)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=p['batch_size'], shuffle=False,
        num_workers=p['num_workers'], collate_fn=collate_custom, drop_last=False)

    normal_label = find_majority_label(model, train_dl)
    print(f"  [{fname}] Label normal (majority): {normal_label}")

    scores, labels = run_inference(model, test_dl, normal_label)
    m = compute_metrics(scores, labels, threshold=threshold)
    print_metrics(dataset, fname, m)
    return m


def aggregate_window_scores(window_scores, method='mean'):
    """
    Agregasi skor anomali window-level (1D array, semua window dari SATU file)
    jadi satu skor anomali untuk file itu.

      'mean' (default) : rata-rata skor semua window. Paling stabil/robust
                          terhadap window tunggal yang noisy - cocok kalau
                          jumlah file sedikit (varians evaluasi sudah tinggi).
      'max'             : skor window paling anomali. Sensitif ke anomali
                          singkat/lokal yang hanya muncul di sebagian window.
      'pct'             : persentase window dengan skor >= 0.5 (anomali).
                          Independen dari threshold file-level yang baru
                          ditentukan belakangan oleh compute_metrics().
    """
    if method == 'max':
        return float(np.max(window_scores))
    elif method == 'pct':
        return float(np.mean(window_scores >= 0.5))
    else:  # 'mean'
        return float(np.mean(window_scores))


def evaluate_pln(threshold=None, agg='mean', seed=None):
    """
    Evaluasi PLN di level FILE (bukan window). Satu model bersama (fname='All')
    dipakai untuk semua file. Untuk setiap file held-out (test_list.txt):
      1. Bangun dataset window-window file itu (self-normalized).
      2. Jalankan inferensi -> skor anomali per window.
      3. Agregasi window->file (lihat aggregate_window_scores).
    Skor+label semua file lalu dikumpulkan dan dihitung metrik SEKALI di akhir.

    seed: kalau tidak None (mis. 0,1,2,...), model dibaca dari dir checkpoint
          dengan suffix '_sN' (mis. results/pln/All_s0/) — dipakai oleh
          run_pln_multiseed.py untuk evaluasi per-seed tanpa saling menimpa.
    """
    config_env = str(BASE_DIR / "configs" / "env.yml")
    config_cls = str(CONFIGS["PLN"]["classification"])
    p = create_config(config_env, config_cls, "All", seed=seed)

    model_path = p['classification_model']
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model belum ada: {model_path}. Jalankan run_pln.py dahulu.")
        return None, []

    model = get_model(p)
    model = torch.nn.DataParallel(model)
    ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
    model.module.load_state_dict(ckpt['model'])
    model.to(device).eval()

    sanomaly = inject_sub_anomaly(p)
    val_transform = get_val_transformations1(p)

    # Normal label: pool ulang file training (normal) untuk voting mayoritas cluster.
    train_files = get_pln_train_files()
    train_ds = None
    for ii, fname in enumerate(train_files):
        ds = PLN(fname, train=True, transform=val_transform, sanomaly=sanomaly,
                 mean_data=None, std_data=None)
        if train_ds is None:
            train_ds = ds
        else:
            train_ds.concat_ds(ds)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=p['batch_size'], shuffle=False,
        num_workers=p['num_workers'], collate_fn=collate_custom, drop_last=False)
    normal_label = find_majority_label(model, train_dl)
    print(f"  Label normal (majority, dari {len(train_files)} file training): {normal_label}")

    # Skor per file (held-out)
    test_files = get_pln_test_files()
    file_scores, file_labels, rows = [], [], []
    for fname in test_files:
        test_ds = PLN(fname, train=False, transform=val_transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=p['batch_size'], shuffle=False,
            num_workers=p['num_workers'], collate_fn=collate_custom, drop_last=False)
        win_scores, win_labels = run_inference(model, test_dl, normal_label)
        file_score = aggregate_window_scores(win_scores, method=agg)
        true_label = int(win_labels[0])  # semua window di file ini punya label sama
        file_scores.append(file_score)
        file_labels.append(true_label)
        rows.append({'fname': fname, 'score': file_score, 'true_label': true_label,
                      'n_windows': len(win_scores)})
        print(f"  [{fname}] skor={file_score:.4f}  label_asli={true_label}  n_window={len(win_scores)}")

    if not file_scores:
        print("  [ERROR] Tidak ada file test untuk dievaluasi.")
        return None, []

    file_scores = np.array(file_scores)
    file_labels = np.array(file_labels)

    if file_labels.sum() == 0 or file_labels.sum() == len(file_labels):
        print("  [WARNING] Semua file test berlabel sama (semua normal atau semua anomali) - "
              "precision/recall/AUROC bisa tidak bermakna.")

    m = compute_metrics(file_scores, file_labels, threshold=threshold)
    print_metrics("PLN", f"ALL ({len(test_files)} file, agg={agg})", m)

    for row in rows:
        row['pred_label'] = int(row['score'] >= m['threshold'])

    return m, rows


def evaluate_sgcc(threshold=None, agg='mean', seed=None):
    """
    Evaluasi SGCC di level PELANGGAN/FILE (bukan window).
    Paradigma identik dengan evaluate_pln() tapi memakai data.SGCC dan konfigurasi SGCC.

    seed: kalau tidak None (mis. 0,1,2,...), model dibaca dari dir checkpoint
          dengan suffix '_sN' (mis. results/sgcc/All_s0/) — dipakai oleh
          run_sgcc_multiseed.py untuk evaluasi per-seed tanpa saling menimpa.
    """
    config_env = str(BASE_DIR / "configs" / "env.yml")
    config_cls = str(CONFIGS["SGCC"]["classification"])
    p = create_config(config_env, config_cls, "All", seed=seed)

    model_path = p['classification_model']
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model belum ada: {model_path}. Jalankan run_sgcc.py dahulu.")
        return None, []

    model = get_model(p)
    model = torch.nn.DataParallel(model)
    ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
    model.module.load_state_dict(ckpt['model'])
    model.to(device).eval()

    sanomaly = inject_sub_anomaly(p)
    val_transform = get_val_transformations1(p)

    # Normal label: pool ulang file training (normal) untuk voting mayoritas cluster.
    train_files = get_sgcc_train_files()
    train_ds = None
    for ii, fname in enumerate(train_files):
        ds = SGCC(fname, train=True, transform=val_transform, sanomaly=sanomaly,
                  mean_data=None, std_data=None)
        if train_ds is None:
            train_ds = ds
        else:
            train_ds.concat_ds(ds)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=p['batch_size'], shuffle=False,
        num_workers=p['num_workers'], collate_fn=collate_custom, drop_last=False)
    normal_label = find_majority_label(model, train_dl)
    print(f"  Label normal (majority, dari {len(train_files)} pelanggan training): {normal_label}")

    # Skor per pelanggan (held-out)
    test_files = get_sgcc_test_files()
    file_scores, file_labels, rows = [], [], []
    for fname in test_files:
        test_ds = SGCC(fname, train=False, transform=val_transform, sanomaly=sanomaly,
                       mean_data=None, std_data=None)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=p['batch_size'], shuffle=False,
            num_workers=p['num_workers'], collate_fn=collate_custom, drop_last=False)
        win_scores, win_labels = run_inference(model, test_dl, normal_label)
        file_score = aggregate_window_scores(win_scores, method=agg)
        true_label = int(win_labels[0])
        file_scores.append(file_score)
        file_labels.append(true_label)
        rows.append({'fname': fname, 'score': file_score, 'true_label': true_label,
                      'n_windows': len(win_scores)})
        print(f"  [{fname}] skor={file_score:.4f}  label_asli={true_label}  n_window={len(win_scores)}")

    if not file_scores:
        print("  [ERROR] Tidak ada file test untuk dievaluasi.")
        return None, []

    file_scores = np.array(file_scores)
    file_labels = np.array(file_labels)

    if file_labels.sum() == 0 or file_labels.sum() == len(file_labels):
        print("  [WARNING] Semua pelanggan test berlabel sama - precision/recall/AUROC bisa tidak bermakna.")

    m = compute_metrics(file_scores, file_labels, threshold=threshold)
    print_metrics("SGCC", f"ALL ({len(test_files)} pelanggan, agg={agg})", m)

    for row in rows:
        row['pred_label'] = int(row['score'] >= m['threshold'])

    return m, rows


def main():
    parser = argparse.ArgumentParser(description="Evaluasi CARLA (MSL/SMAP/SMD/LEAD/PLN/SGCC)")
    parser.add_argument('--dataset',   required=True, choices=['MSL', 'SMAP', 'SMD', 'LEAD', 'PLN', 'SGCC'])
    parser.add_argument('--channel',   nargs='+', default=None, metavar="NAME")
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--agg',         default='mean', choices=['mean', 'max', 'pct'],
                        help="Agregasi window->file untuk PLN (default: mean)")
    parser.add_argument('--output',      default=None, metavar="PATH")
    parser.add_argument('--seed',        type=int, default=None,
                        help='Seed training yang dievaluasi (mis. --seed 0 membaca '
                             'checkpoint dari results/pln/All_s0/). Default None: '
                             'baca dari results/pln/All/ (perilaku lama).')
    parser.add_argument('--output_json', default=None, metavar="PATH",
                        help='Simpan metrik agregat (F1, AUROC, dll.) ke file JSON '
                             'untuk dibaca run_pln_multiseed.py. Tidak ada output '
                             'JSON kalau tidak diisi.')
    args = parser.parse_args()

    dataset = args.dataset.upper()

    if dataset == 'PLN':
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Tag tipe anomali (mis. 'All_theft' vs 'All') dari anomaly_kwargs.type di
        # config classification PLN yang aktif - sama dengan suffix folder checkpoint
        # di utils/config.py & run_pln.py - supaya nama file hasil evaluasi langsung
        # menunjukkan model checkpoint mana yang sedang dievaluasi.
        dir_fname = _dir_fname(CONFIGS["PLN"]["classification"], seed=args.seed)
        if args.output:
            out_txt = Path(args.output)
            out_txt.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = BASE_DIR / "results" / "eval"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_txt = out_dir / f"eval_PLN_{dir_fname}_{args.agg}_{ts}.txt"
        out_csv = out_txt.with_suffix('.csv')

        tee = _Tee(out_txt)
        sys.stdout = tee
        try:
            print(f"Evaluation run  : {ts}")
            print(f"Seed            : {args.seed if args.seed is not None else '(default, no suffix)'}")
            print(f"Dataset         : PLN (evaluasi level FILE, agregasi={args.agg})")
            print(f"Checkpoint dir  : results/pln/{dir_fname}/")
            print(f"Output TXT      : {out_txt}")
            print(f"Output CSV      : {out_csv}")

            m, rows = evaluate_pln(threshold=args.threshold, agg=args.agg, seed=args.seed)

            if m is not None:
                with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
                    writer = csv.DictWriter(
                        cf, fieldnames=['fname', 'score', 'pred_label', 'true_label', 'n_windows'])
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

                # Output JSON untuk run_pln_multiseed.py (opsional)
                if args.output_json:
                    import json
                    json_metrics = {
                        k: float(v) for k, v in m.items()
                        if k not in ('prec_curve', 'rec_curve', 'f1_curve', 'thr_pr')
                    }
                    json_metrics['seed'] = args.seed
                    json_metrics['agg']  = args.agg
                    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
                    with open(args.output_json, 'w', encoding='utf-8') as jf:
                        json.dump(json_metrics, jf, indent=2)
                    print(f"  JSON : {args.output_json}")

                print(f"\nHasil disimpan ke:")
                print(f"  TXT : {out_txt}")
                print(f"  CSV : {out_csv}")
        finally:
            tee.close()
        return

    if dataset == 'SGCC':
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_fname = _dir_fname(CONFIGS["SGCC"]["classification"], seed=args.seed)
        if args.output:
            out_txt = Path(args.output)
            out_txt.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = BASE_DIR / "results" / "eval"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_txt = out_dir / f"eval_SGCC_{dir_fname}_{args.agg}_{ts}.txt"
        out_csv = out_txt.with_suffix('.csv')

        tee = _Tee(out_txt)
        sys.stdout = tee
        try:
            print(f"Evaluation run  : {ts}")
            print(f"Seed            : {args.seed if args.seed is not None else '(default, no suffix)'}")
            print(f"Dataset         : SGCC (evaluasi level PELANGGAN, agregasi={args.agg})")
            print(f"Checkpoint dir  : results/sgcc/{dir_fname}/")
            print(f"Output TXT      : {out_txt}")
            print(f"Output CSV      : {out_csv}")

            m, rows = evaluate_sgcc(threshold=args.threshold, agg=args.agg, seed=args.seed)

            if m is not None:
                with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
                    writer = csv.DictWriter(
                        cf, fieldnames=['fname', 'score', 'pred_label', 'true_label', 'n_windows'])
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

                if args.output_json:
                    import json
                    json_metrics = {
                        k: float(v) for k, v in m.items()
                        if k not in ('prec_curve', 'rec_curve', 'f1_curve', 'thr_pr')
                    }
                    json_metrics['seed'] = args.seed
                    json_metrics['agg']  = args.agg
                    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
                    with open(args.output_json, 'w', encoding='utf-8') as jf:
                        json.dump(json_metrics, jf, indent=2)
                    print(f"  JSON : {args.output_json}")

                print(f"\nHasil disimpan ke:")
                print(f"  TXT : {out_txt}")
                print(f"  CSV : {out_csv}")
        finally:
            tee.close()
        return

    if dataset == 'SMD':
        all_items  = get_smd_machines()
        label_type = "machine"
    elif dataset == 'LEAD':
        all_items  = get_lead_buildings(eval_only=True)
        label_type = "building"
    else:
        all_items  = get_channels(dataset)
        label_type = "channel"

    if args.channel:
        items = []
        for c in args.channel:
            if dataset in ('SMD', 'LEAD'):
                if c not in all_items:
                    print(f"[WARNING] '{c}' tidak ditemukan di {dataset} list. Dilewati.")
                else:
                    items.append(c)
            else:
                try:
                    sc = get_spacecraft(c)
                except ValueError:
                    print(f"[WARNING] '{c}' tidak ditemukan di CSV. Dilewati.")
                    continue
                if sc != dataset:
                    print(f"[WARNING] '{c}' bukan channel {dataset}. Dilewati.")
                else:
                    items.append(c)
    else:
        items = all_items

    if not items:
        print("[ERROR] Tidak ada item valid untuk dievaluasi.")
        sys.exit(1)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_txt = Path(args.output)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = BASE_DIR / "results" / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        channel_tag = "_".join(args.channel) if args.channel else "all"
        channel_tag = channel_tag.replace('.', '-').replace('/', '-')
        out_txt = out_dir / f"eval_{dataset}_{channel_tag}_{ts}.txt"

    out_csv = out_txt.with_suffix('.csv')
    tee = _Tee(out_txt)
    sys.stdout = tee

    try:
        print(f"Evaluation run  : {ts}")
        print(f"Dataset         : {dataset}")
        print(f"Output TXT      : {out_txt}")
        print(f"Output CSV      : {out_csv}")
        print(f"{label_type.capitalize()}s : {len(items)} dari {len(all_items)} total")

        results = {}
        for i, name in enumerate(items, 1):
            print(f"\n[{i}/{len(items)}] Evaluasi {label_type}: {name}")
            m = evaluate_channel(dataset, name, threshold=args.threshold)
            if m is not None:
                results[name] = m

        keys = ['precision', 'recall', 'precision_macro', 'recall_macro',
                'f1_anomaly', 'f1_normal', 'f1_macro', 'aupr', 'auroc']
        if len(results) > 1:
            avgs = {k: np.nanmean([v[k] for v in results.values()]) for k in keys}
            sep  = "=" * 52
            print(f"\n{sep}")
            print(f"  RATA-RATA {dataset} ({len(results)} {label_type})")
            print(sep)
            for k in keys:
                print(f"  {k:<18}: {avgs[k]:.4f}")
            print(sep)

        csv_fields = ['name', 'threshold', 'precision', 'recall',
                      'precision_macro', 'recall_macro',
                      'f1_anomaly', 'f1_normal', 'f1_macro',
                      'aupr', 'auroc', 'tp', 'tn', 'fp', 'fn']
        with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_fields)
            writer.writeheader()
            for name, m in results.items():
                writer.writerow({
                    'name':            name,
                    'threshold':       f"{m['threshold']:.6f}",
                    'precision':       f"{m['precision']:.6f}",
                    'recall':          f"{m['recall']:.6f}",
                    'precision_macro': f"{m['precision_macro']:.6f}",
                    'recall_macro':    f"{m['recall_macro']:.6f}",
                    'f1_anomaly':      f"{m['f1_anomaly']:.6f}",
                    'f1_normal':       f"{m['f1_normal']:.6f}",
                    'f1_macro':        f"{m['f1_macro']:.6f}",
                    'aupr':            f"{m['aupr']:.6f}",
                    'auroc':           f"{m['auroc']:.6f}",
                    'tp': m['tp'], 'tn': m['tn'], 'fp': m['fp'], 'fn': m['fn'],
                })
            if len(results) > 1:
                avgs = {k: np.nanmean([v[k] for v in results.values()]) for k in keys}
                writer.writerow({
                    'name':            f'__AVG__ ({len(results)} {label_type}s)',
                    'threshold':       '',
                    'precision':       f"{avgs['precision']:.6f}",
                    'recall':          f"{avgs['recall']:.6f}",
                    'precision_macro': f"{avgs['precision_macro']:.6f}",
                    'recall_macro':    f"{avgs['recall_macro']:.6f}",
                    'f1_anomaly':      f"{avgs['f1_anomaly']:.6f}",
                    'f1_normal':       f"{avgs['f1_normal']:.6f}",
                    'f1_macro':        f"{avgs['f1_macro']:.6f}",
                    'aupr':            f"{avgs['aupr']:.6f}",
                    'auroc':           f"{avgs['auroc']:.6f}",
                    'tp': '', 'tn': '', 'fp': '', 'fn': '',
                })

        print(f"\nHasil disimpan ke:")
        print(f"  TXT : {out_txt}")
        print(f"  CSV : {out_csv}")

    finally:
        tee.close()


if __name__ == '__main__':
    main()
