"""
generate_eda_notebooks.py
Buat 4 notebook EDA (MSL, SMAP, LEAD, PLN) untuk keperluan tesis.
Jalankan: python notebooks/generate_eda_notebooks.py
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ------------------------------------------------------------------ #
#  Helper: bangun struktur cell notebook
# ------------------------------------------------------------------ #

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

def save(notebook, filename):
    path = OUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"  Tersimpan: {path}")


# ================================================================== #
#  SETUP CELL (sama untuk semua notebook)
# ================================================================== #

SETUP = code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
import ast, warnings, os
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'lines.linewidth': 0.9,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

BASE = Path(r'D:\\CARLA')
COLORS = {'normal': '#2196F3', 'anomaly': '#FF5722', 'val': '#4CAF50'}

# Buat folder output jika belum ada
FIG_DIR = BASE / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f'Setup selesai. Output gambar: {FIG_DIR}')
""")


# ================================================================== #
#  1. EDA MSL
# ================================================================== #

def make_msl():
    cells = [
        md("# Exploratory Data Analysis — MSL (Mars Science Laboratory)\n"
           "Dataset MSL merupakan data telemetri dari Mars Science Laboratory (Curiosity rover) milik NASA. "
           "Setiap *channel* merepresentasikan satu sensor onboard. Anomali pada dataset ini bersifat "
           "**point** dan **contextual** yang telah dilabeli secara manual oleh para ahli NASA."),

        SETUP,

        md("## 1. Memuat Metadata"),
        code("""\
DATA_DIR = BASE / 'datasets' / 'MSL_SMAP'
df_labels = pd.read_csv(DATA_DIR / 'labeled_anomalies.csv')
msl = df_labels[df_labels['spacecraft'] == 'MSL'].copy()
msl = msl.reset_index(drop=True)

print(f"Jumlah channel MSL : {len(msl)}")
print(f"\\nKolom tersedia    : {list(msl.columns)}")
print()
display(msl[['chan_id','class','num_values','anomaly_sequences']].head(10))
"""),

        md("## 2. Distribusi Panjang Deret Waktu per Channel"),
        code("""\
lengths = msl['num_values'].values
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(msl['chan_id'], lengths, color=COLORS['normal'], alpha=0.8)
axes[0].set_xlabel('Channel ID')
axes[0].set_ylabel('Jumlah Timestep')
axes[0].set_title('Panjang Deret Waktu per Channel MSL')
axes[0].tick_params(axis='x', rotation=90)

axes[1].hist(lengths, bins=10, color=COLORS['normal'], edgecolor='white', alpha=0.85)
axes[1].set_xlabel('Jumlah Timestep')
axes[1].set_ylabel('Frekuensi')
axes[1].set_title('Distribusi Panjang Deret Waktu')
for ax in axes:
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_msl_lengths.png', bbox_inches='tight')
plt.show()
print(f"\\nStatistik panjang deret waktu:")
print(pd.Series(lengths).describe().round(1).to_string())
"""),

        md("## 3. Distribusi Tipe Anomali"),
        code("""\
import re
def parse_class(c):
    # class column mungkin berisi identifier tak-quoted: [point, contextual]
    return re.findall(r'[A-Za-z_][A-Za-z0-9_]*', str(c))

all_types = []
for _, row in msl.iterrows():
    all_types.extend(parse_class(row['class']))

from collections import Counter
type_counts = Counter(all_types)
print("Tipe anomali MSL:")
for k, v in type_counts.items():
    print(f"  {k:15s}: {v} segmen")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(type_counts.keys(), type_counts.values(),
       color=[COLORS['anomaly'], '#FF9800'], edgecolor='white', alpha=0.85)
ax.set_xlabel('Tipe Anomali')
ax.set_ylabel('Jumlah Segmen')
ax.set_title('Distribusi Tipe Anomali — MSL')
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_msl_anomaly_types.png', bbox_inches='tight')
plt.show()
"""),

        md("## 4. Visualisasi Sampel Channel (Normal vs Anomali)"),
        code("""\
def load_msl(chan_id, split='test'):
    path = DATA_DIR / 'data' / split / f'{chan_id}.npy'
    return np.load(path)

def parse_anomaly_seqs(seq_str):
    return ast.literal_eval(seq_str)

# Ambil 4 channel sampel
sample_chans = msl['chan_id'].iloc[:4].tolist()
fig, axes = plt.subplots(len(sample_chans), 1, figsize=(14, 3.5 * len(sample_chans)))
if len(sample_chans) == 1:
    axes = [axes]

for ax, chan_id in zip(axes, sample_chans):
    row = msl[msl['chan_id'] == chan_id].iloc[0]
    data = load_msl(chan_id, 'test')   # (T, 25)
    T = data.shape[0]
    t = np.arange(T)

    # Plot channel pertama sebagai representasi
    ax.plot(t, data[:, 0], color=COLORS['normal'], linewidth=0.8, label='Channel 0')

    # Highlight anomali
    seqs = parse_anomaly_seqs(row['anomaly_sequences'])
    for i, (s, e) in enumerate(seqs):
        ax.axvspan(s, e, color=COLORS['anomaly'], alpha=0.25,
                   label='Anomali' if i == 0 else '_')

    ax.set_title(f'Channel {chan_id}  |  T={T}  |  '
                 f"Tipe: {row['class']}  |  {len(seqs)} segmen anomali")
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Nilai (norm.)')
    ax.legend(fontsize=8, loc='upper right')

plt.suptitle('MSL — Contoh Deret Waktu dengan Anomali Tersorot', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_msl_timeseries.png', bbox_inches='tight')
plt.show()
"""),

        md("## 5. Rasio Anomali per Channel"),
        code("""\
anomaly_ratios = []
for _, row in msl.iterrows():
    T = row['num_values']
    seqs = parse_anomaly_seqs(row['anomaly_sequences'])
    n_anom = sum(e - s + 1 for s, e in seqs)
    anomaly_ratios.append(n_anom / T * 100)

msl['anomaly_ratio_%'] = anomaly_ratios

fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(msl['chan_id'], msl['anomaly_ratio_%'],
              color=COLORS['anomaly'], alpha=0.8)
ax.axhline(np.mean(anomaly_ratios), color='black', linestyle='--', linewidth=1,
           label=f'Rata-rata: {np.mean(anomaly_ratios):.2f}%')
ax.set_xlabel('Channel ID')
ax.set_ylabel('Rasio Anomali (%)')
ax.set_title('Rasio Anomali per Channel — MSL')
ax.tick_params(axis='x', rotation=90)
ax.legend()
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_msl_anomaly_ratio.png', bbox_inches='tight')
plt.show()

print(f"Rata-rata rasio anomali : {np.mean(anomaly_ratios):.2f}%")
print(f"Min: {np.min(anomaly_ratios):.2f}%  Max: {np.max(anomaly_ratios):.2f}%")
"""),

        md("## 6. Statistik Fitur (Channel P-1 sebagai Representasi)"),
        code("""\
chan = 'P-1'
train_data = load_msl(chan, 'train')  # (T_train, 25)
test_data  = load_msl(chan, 'test')   # (T_test,  25)

print(f"Channel {chan}:")
print(f"  Train shape : {train_data.shape}")
print(f"  Test shape  : {test_data.shape}")
print(f"  Jumlah fitur: {train_data.shape[1]}")

# Statistik deskriptif 25 fitur dari data train
df_stats = pd.DataFrame(train_data,
    columns=[f'F{i}' for i in range(train_data.shape[1])])
stats = df_stats.describe().T
stats['skewness'] = df_stats.skew()
stats['kurtosis'] = df_stats.kurt()
print(f"\\nStatistik deskriptif {train_data.shape[1]} fitur (data train):")
display(stats[['mean','std','min','max','skewness','kurtosis']].round(4).head(10))
"""),

        md("## 7. Heatmap Korelasi Antar Fitur"),
        code("""\
import matplotlib.colors as mcolors

chan = 'P-1'
train_data = load_msl(chan, 'train')
corr = np.corrcoef(train_data.T)  # (25, 25)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Koefisien Korelasi Pearson')
ax.set_title(f'Heatmap Korelasi Fitur — MSL Channel {chan}')
ax.set_xlabel('Indeks Fitur')
ax.set_ylabel('Indeks Fitur')
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_msl_correlation.png', bbox_inches='tight')
plt.show()

strong = np.sum(np.abs(corr) > 0.7) - corr.shape[0]
print(f"Pasangan fitur dengan |korelasi| > 0.7: {strong // 2}")
"""),

        md("## 8. Ringkasan Dataset MSL"),
        code("""\
print("=" * 50)
print("RINGKASAN DATASET MSL")
print("=" * 50)
print(f"Jumlah channel          : {len(msl)}")
print(f"Jumlah fitur per sample : 25")
print(f"Rata-rata panjang test  : {msl['num_values'].mean():.0f} timestep")
print(f"Total segmen anomali    : {sum(len(parse_anomaly_seqs(r)) for r in msl['anomaly_sequences'])}")
print(f"Rata-rata rasio anomali : {np.mean(anomaly_ratios):.2f}%")
print(f"Tipe anomali            : point, contextual")
"""),
    ]
    return nb(cells)


# ================================================================== #
#  2. EDA SMAP
# ================================================================== #

def make_smap():
    cells = [
        md("# Exploratory Data Analysis — SMAP (Soil Moisture Active Passive)\n"
           "Dataset SMAP berasal dari data telemetri satelit SMAP milik NASA. "
           "Memiliki format yang sama dengan MSL namun mencakup 55 channel dengan "
           "karakteristik sinyal yang berbeda."),

        SETUP,

        md("## 1. Memuat Metadata"),
        code("""\
DATA_DIR = BASE / 'datasets' / 'MSL_SMAP'
df_labels = pd.read_csv(DATA_DIR / 'labeled_anomalies.csv')
smap = df_labels[df_labels['spacecraft'] == 'SMAP'].copy().reset_index(drop=True)

print(f"Jumlah channel SMAP : {len(smap)}")
display(smap[['chan_id','class','num_values','anomaly_sequences']].head(10))
"""),

        md("## 2. Distribusi Panjang Deret Waktu"),
        code("""\
lengths = smap['num_values'].values
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(smap['chan_id'], lengths, color=COLORS['normal'], alpha=0.8)
axes[0].set_xlabel('Channel ID')
axes[0].set_ylabel('Jumlah Timestep')
axes[0].set_title('Panjang Deret Waktu per Channel SMAP')
axes[0].tick_params(axis='x', rotation=90)

axes[1].hist(lengths, bins=12, color=COLORS['normal'], edgecolor='white', alpha=0.85)
axes[1].set_xlabel('Jumlah Timestep')
axes[1].set_ylabel('Frekuensi')
axes[1].set_title('Distribusi Panjang Deret Waktu')

plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_smap_lengths.png', bbox_inches='tight')
plt.show()
print(pd.Series(lengths).describe().round(1).to_string())
"""),

        md("## 3. Distribusi Tipe Anomali"),
        code("""\
import ast
from collections import Counter

def parse_class(c):
    return ast.literal_eval(c)

def parse_seqs(s):
    return ast.literal_eval(s)

all_types = []
for _, row in smap.iterrows():
    all_types.extend(parse_class(row['class']))

type_counts = Counter(all_types)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(type_counts.keys(), type_counts.values(),
       color=[COLORS['anomaly'], '#FF9800'], edgecolor='white', alpha=0.85)
ax.set_xlabel('Tipe Anomali')
ax.set_ylabel('Jumlah Segmen')
ax.set_title('Distribusi Tipe Anomali — SMAP')
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_smap_anomaly_types.png', bbox_inches='tight')
plt.show()
for k, v in type_counts.items():
    print(f"  {k:15s}: {v} segmen")
"""),

        md("## 4. Visualisasi Sampel Channel"),
        code("""\
def load_smap(chan_id, split='test'):
    return np.load(DATA_DIR / 'data' / split / f'{chan_id}.npy')

sample_chans = smap['chan_id'].iloc[:4].tolist()
fig, axes = plt.subplots(len(sample_chans), 1, figsize=(14, 3.5 * len(sample_chans)))

for ax, chan_id in zip(axes, sample_chans):
    row = smap[smap['chan_id'] == chan_id].iloc[0]
    data = load_smap(chan_id, 'test')
    T = data.shape[0]
    ax.plot(np.arange(T), data[:, 0], color=COLORS['normal'], linewidth=0.8)
    seqs = parse_seqs(row['anomaly_sequences'])
    for i, (s, e) in enumerate(seqs):
        ax.axvspan(s, e, color=COLORS['anomaly'], alpha=0.25,
                   label='Anomali' if i == 0 else '_')
    ax.set_title(f'Channel {chan_id}  |  T={T}  |  {len(seqs)} segmen anomali')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Nilai')
    ax.legend(fontsize=8, loc='upper right')

plt.suptitle('SMAP — Contoh Deret Waktu dengan Anomali Tersorot', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_smap_timeseries.png', bbox_inches='tight')
plt.show()
"""),

        md("## 5. Rasio Anomali per Channel"),
        code("""\
anomaly_ratios = []
for _, row in smap.iterrows():
    T = row['num_values']
    seqs = parse_seqs(row['anomaly_sequences'])
    n_anom = sum(e - s + 1 for s, e in seqs)
    anomaly_ratios.append(n_anom / T * 100)
smap['anomaly_ratio_%'] = anomaly_ratios

fig, ax = plt.subplots(figsize=(16, 4))
ax.bar(smap['chan_id'], smap['anomaly_ratio_%'], color=COLORS['anomaly'], alpha=0.8)
ax.axhline(np.mean(anomaly_ratios), color='black', linestyle='--', linewidth=1,
           label=f'Rata-rata: {np.mean(anomaly_ratios):.2f}%')
ax.set_xlabel('Channel ID')
ax.set_ylabel('Rasio Anomali (%)')
ax.set_title('Rasio Anomali per Channel — SMAP')
ax.tick_params(axis='x', rotation=90)
ax.legend()
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_smap_anomaly_ratio.png', bbox_inches='tight')
plt.show()
print(f"Rata-rata rasio anomali : {np.mean(anomaly_ratios):.2f}%")
"""),

        md("## 6. Perbandingan Statistik Train vs Test (Channel P-1)"),
        code("""\
chan = 'P-1'
tr = load_smap(chan, 'train')
te = load_smap(chan, 'test')

print(f"Channel {chan}:")
print(f"  Train shape : {tr.shape}")
print(f"  Test shape  : {te.shape}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for i, (name, data, color) in enumerate([
        ('Train (Normal)', tr, COLORS['normal']),
        ('Test (Campuran)', te, COLORS['anomaly'])]):
    axes[i].plot(data[:, 0], color=color, linewidth=0.7, alpha=0.8)
    axes[i].set_title(f'{name} — Fitur 0')
    axes[i].set_xlabel('Timestep')
    axes[i].set_ylabel('Nilai')
plt.suptitle(f'Perbandingan Train vs Test — SMAP Channel {chan}')
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_smap_train_test.png', bbox_inches='tight')
plt.show()
"""),

        md("## 7. Ringkasan Dataset SMAP"),
        code("""\
print("=" * 50)
print("RINGKASAN DATASET SMAP")
print("=" * 50)
total_segs = sum(len(parse_seqs(r)) for r in smap['anomaly_sequences'])
print(f"Jumlah channel          : {len(smap)}")
print(f"Jumlah fitur per sample : {load_smap(smap.iloc[0]['chan_id'], 'test').shape[1]}")
print(f"Rata-rata panjang test  : {smap['num_values'].mean():.0f} timestep")
print(f"Total segmen anomali    : {total_segs}")
print(f"Rata-rata rasio anomali : {np.mean(anomaly_ratios):.2f}%")
print(f"Tipe anomali            : point, contextual")
"""),
    ]
    return nb(cells)


# ================================================================== #
#  3. EDA LEAD
# ================================================================== #

def make_lead():
    cells = [
        md("# Exploratory Data Analysis — LEAD (Large-scale Energy Anomaly Detection)\n"
           "Dataset LEAD merupakan data konsumsi energi bangunan komersial skala besar. "
           "Label anomali tersedia di level bangunan (building-level) mencakup berbagai "
           "pola konsumsi tidak normal seperti kebocoran energi, kegagalan peralatan, dan "
           "pemakaian tidak wajar."),

        SETUP,

        md("## 1. Memuat Daftar Bangunan"),
        code("""\
DATA_DIR = BASE / 'datasets' / 'LEAD' / 'processed'

def read_list(fname):
    path = DATA_DIR / fname
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]

train_buildings  = read_list('building_list.txt')
eval_buildings   = read_list('building_eval_list.txt')

print(f"Total bangunan (training)  : {len(train_buildings)}")
print(f"Bangunan dengan anomali    : {len(eval_buildings)}")
print(f"Contoh file: {train_buildings[:5]}")
"""),

        md("## 2. Statistik Panjang Deret Waktu"),
        code("""\
train_dir = DATA_DIR / 'train'
lengths, n_features = [], []

for fname in train_buildings[:50]:  # sampel 50 bangunan
    try:
        arr = np.load(train_dir / fname)
        lengths.append(arr.shape[0])
        n_features.append(arr.shape[1])
    except Exception:
        pass

print(f"Sampel  : {len(lengths)} bangunan")
print(f"Fitur   : {n_features[0]} per timestep (konsisten)")
print(f"\\nStatistik panjang deret waktu (timestep):")
print(pd.Series(lengths).describe().round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(lengths, bins=15, color=COLORS['normal'], edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Jumlah Timestep')
axes[0].set_ylabel('Frekuensi')
axes[0].set_title('Distribusi Panjang Deret Waktu — LEAD')

axes[1].boxplot(lengths, patch_artist=True,
                boxprops=dict(facecolor=COLORS['normal'], alpha=0.6))
axes[1].set_ylabel('Jumlah Timestep')
axes[1].set_title('Boxplot Panjang Deret Waktu')
axes[1].set_xticklabels(['Bangunan'])
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_lead_lengths.png', bbox_inches='tight')
plt.show()
"""),

        md("## 3. Visualisasi Contoh Deret Waktu Bangunan"),
        code("""\
# Ambil 4 bangunan acak dari daftar train
np.random.seed(42)
sample_files = np.random.choice(train_buildings, size=4, replace=False)

fig, axes = plt.subplots(4, 1, figsize=(14, 14))
for ax, fname in zip(axes, sample_files):
    arr = np.load(train_dir / fname)
    T = arr.shape[0]
    t = np.arange(T)
    # Plot 3 fitur pertama
    for ch in range(min(3, arr.shape[1])):
        ax.plot(t, arr[:, ch], linewidth=0.7, alpha=0.75, label=f'Fitur {ch}')
    ax.set_title(f'Bangunan: {fname}  |  T={T}  |  {arr.shape[1]} fitur')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Nilai Energi')
    ax.legend(fontsize=7, loc='upper right', ncol=3)

plt.suptitle('LEAD — Contoh Konsumsi Energi Bangunan (Fitur 0-2)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_lead_timeseries.png', bbox_inches='tight')
plt.show()
"""),

        md("## 4. Perbandingan Statistik: Bangunan Normal vs Anomali"),
        code("""\
# Load beberapa bangunan dari daftar eval (anomali) dan non-eval (normal)
non_eval = [b for b in train_buildings if b not in eval_buildings]

n_sample = 20
normal_means, anomaly_means = [], []
normal_stds,  anomaly_stds  = [], []

for fname in non_eval[:n_sample]:
    try:
        arr = np.load(train_dir / fname)
        normal_means.append(np.mean(arr))
        normal_stds.append(np.std(arr))
    except: pass

eval_dir = DATA_DIR / 'test'
for fname in eval_buildings[:n_sample]:
    try:
        arr = np.load(eval_dir / fname) if (eval_dir / fname).exists() \
              else np.load(train_dir / fname)
        anomaly_means.append(np.mean(arr))
        anomaly_stds.append(np.std(arr))
    except: pass

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

bp0 = axes[0].boxplot([normal_means, anomaly_means], patch_artist=True,
                      labels=['Normal', 'Anomali'])
bp0['boxes'][0].set_facecolor(COLORS['normal'])
bp0['boxes'][1].set_facecolor(COLORS['anomaly'])
for box in bp0['boxes']:
    box.set_alpha(0.7)
axes[0].set_ylabel('Rata-rata Konsumsi Energi')
axes[0].set_title('Distribusi Mean per Bangunan')

bp1 = axes[1].boxplot([normal_stds, anomaly_stds], patch_artist=True,
                      labels=['Normal', 'Anomali'])
bp1['boxes'][0].set_facecolor(COLORS['normal'])
bp1['boxes'][1].set_facecolor(COLORS['anomaly'])
for box in bp1['boxes']:
    box.set_alpha(0.7)
axes[1].set_ylabel('Standar Deviasi Konsumsi')
axes[1].set_title('Distribusi Std per Bangunan')

plt.suptitle('LEAD — Perbandingan Statistik Bangunan Normal vs Anomali', fontsize=11)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_lead_normal_vs_anomaly.png', bbox_inches='tight')
plt.show()

print(f"Mean energi normal  : {np.mean(normal_means):.4f} ± {np.std(normal_means):.4f}")
print(f"Mean energi anomali : {np.mean(anomaly_means):.4f} ± {np.std(anomaly_means):.4f}")
"""),

        md("## 5. Distribusi Nilai Fitur (Histogram)"),
        code("""\
# Ambil satu bangunan representatif
fname = train_buildings[0]
arr   = np.load(train_dir / fname)

n_feat_show = min(6, arr.shape[1])
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()

for i in range(n_feat_show):
    axes[i].hist(arr[:, i], bins=40, color=COLORS['normal'],
                 edgecolor='white', alpha=0.8)
    axes[i].set_title(f'Fitur {i}')
    axes[i].set_xlabel('Nilai')
    axes[i].set_ylabel('Frekuensi')

for j in range(n_feat_show, len(axes)):
    axes[j].set_visible(False)

plt.suptitle(f'LEAD — Distribusi Nilai 6 Fitur Pertama ({fname})', fontsize=11)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_lead_feature_dist.png', bbox_inches='tight')
plt.show()
"""),

        md("## 6. Heatmap Korelasi Fitur"),
        code("""\
fname = train_buildings[0]
arr   = np.load(train_dir / fname)
corr  = np.corrcoef(arr.T)   # (39, 39)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Koefisien Korelasi Pearson')
ax.set_title(f'Heatmap Korelasi 39 Fitur — LEAD ({fname})')
ax.set_xlabel('Indeks Fitur')
ax.set_ylabel('Indeks Fitur')
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_lead_correlation.png', bbox_inches='tight')
plt.show()

strong = (np.abs(corr) > 0.7).sum() - corr.shape[0]
print(f"Pasangan fitur dengan |korelasi| > 0.7: {strong // 2}")
"""),

        md("## 7. Ringkasan Dataset LEAD"),
        code("""\
print("=" * 55)
print("RINGKASAN DATASET LEAD")
print("=" * 55)
print(f"Total bangunan              : {len(train_buildings)}")
print(f"Bangunan dengan anomali     : {len(eval_buildings)}")
print(f"Jumlah fitur per timestep   : 39")
print(f"Rata-rata panjang TS        : {np.mean(lengths):.0f} timestep")
print(f"Label anomali               : level bangunan (building-level)")
print(f"Metode evaluasi CARLA       : agregasi window → file-level score")
"""),
    ]
    return nb(cells)


# ================================================================== #
#  4. EDA PLN
# ================================================================== #

def make_pln():
    cells = [
        md("# Exploratory Data Analysis — PLN (Profil Beban Listrik PLN)\n"
           "Dataset PLN berisi data load profile pelanggan listrik PT PLN. "
           "Setiap file merepresentasikan satu pelanggan dengan data deret waktu "
           "konsumsi daya dalam interval 30 menit (atau 1 jam). Label anomali bersifat "
           "**per file** (bukan per timestep), sehingga satu file dinyatakan normal (0) "
           "atau anomali (1) secara keseluruhan."),

        SETUP,

        md("## 1. Memuat Daftar File dan Label"),
        code("""\
DATA_DIR = BASE / 'datasets' / 'PLN' / 'processed'

file_labels = pd.read_csv(DATA_DIR / 'file_labels.csv')
print(f"Total file : {len(file_labels)}")
print(f"\\nDistribusi label:")
print(file_labels['label'].value_counts().rename({0: 'Normal (0)', 1: 'Anomali (1)'}).to_string())
print()
display(file_labels.head(10))
"""),

        md("## 2. Distribusi Label (Class Imbalance)"),
        code("""\
counts = file_labels['label'].value_counts().sort_index()
labels_name = ['Normal', 'Anomali']

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(labels_name, counts.values,
            color=[COLORS['normal'], COLORS['anomaly']],
            edgecolor='white', alpha=0.85)
axes[0].set_ylabel('Jumlah File')
axes[0].set_title('Distribusi Label File PLN')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 0.2, str(v), ha='center', fontsize=11, fontweight='bold')

axes[1].pie(counts.values, labels=labels_name,
            colors=[COLORS['normal'], COLORS['anomaly']],
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
axes[1].set_title('Proporsi Label')

plt.suptitle('PLN — Distribusi Kelas (Label per File)', fontsize=11)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_pln_class_dist.png', bbox_inches='tight')
plt.show()

ratio = counts[1] / len(file_labels) * 100
print(f"Rasio anomali : {ratio:.1f}%  (imbalance ratio 1:{counts[0]//max(1,counts[1])})")
"""),

        md("## 3. Statistik Panjang Deret Waktu per File"),
        code("""\
data_dir = DATA_DIR / 'data'
lengths_normal, lengths_anomaly = [], []

for _, row in file_labels.iterrows():
    try:
        arr = np.load(data_dir / row['fname'])
        T = arr.shape[0]
        if row['label'] == 0:
            lengths_normal.append(T)
        else:
            lengths_anomaly.append(T)
    except Exception:
        pass

all_lengths = lengths_normal + lengths_anomaly
print(f"Total file berhasil dimuat : {len(all_lengths)}")
print(f"\\nPanjang deret waktu (timestep):")
print(pd.DataFrame({
    'Normal'  : pd.Series(lengths_normal).describe(),
    'Anomali' : pd.Series(lengths_anomaly).describe(),
}).round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(lengths_normal,  bins=20, alpha=0.7, color=COLORS['normal'],
        label=f'Normal (n={len(lengths_normal)})', edgecolor='white')
ax.hist(lengths_anomaly, bins=20, alpha=0.7, color=COLORS['anomaly'],
        label=f'Anomali (n={len(lengths_anomaly)})', edgecolor='white')
ax.set_xlabel('Panjang Deret Waktu (Timestep)')
ax.set_ylabel('Jumlah File')
ax.set_title('Distribusi Panjang Deret Waktu — PLN')
ax.legend()
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_pln_lengths.png', bbox_inches='tight')
plt.show()
"""),

        md("## 4. Visualisasi Contoh File Normal vs Anomali"),
        code("""\
normal_files  = file_labels[file_labels['label'] == 0]['fname'].tolist()
anomaly_files = file_labels[file_labels['label'] == 1]['fname'].tolist()

n_show = min(3, len(normal_files), len(anomaly_files))
fig, axes = plt.subplots(n_show * 2, 1, figsize=(14, 4 * n_show * 2))

row_idx = 0
for category, files, color, label in [
    ('Normal', normal_files[:n_show], COLORS['normal'], 'Normal'),
    ('Anomali', anomaly_files[:n_show], COLORS['anomaly'], 'Anomali'),
]:
    for fname in files[:n_show]:
        try:
            arr = np.load(data_dir / fname)
        except: continue
        ax = axes[row_idx]
        T  = arr.shape[0]
        # Plot semua 8 fitur dengan transparansi
        for ch in range(min(8, arr.shape[1])):
            ax.plot(np.arange(T), arr[:, ch],
                    linewidth=0.7, alpha=0.5,
                    color=color, label=f'Fitur {ch}' if ch < 3 else '_')
        ax.set_title(f'[{label}] {fname}  |  T={T}  |  {arr.shape[1]} fitur')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Daya (kW)')
        if row_idx == 0:
            ax.legend(fontsize=7, loc='upper right', ncol=4)
        row_idx += 1

plt.suptitle('PLN — Profil Beban Listrik: Normal vs Anomali', fontsize=12, y=1.005)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_pln_timeseries.png', bbox_inches='tight')
plt.show()
"""),

        md("## 5. Statistik Deskriptif Fitur"),
        code("""\
# Agregat statistik dari semua file
means_normal,  stds_normal  = [], []
means_anomaly, stds_anomaly = [], []

for _, row in file_labels.iterrows():
    try:
        arr = np.load(data_dir / row['fname'])
        if row['label'] == 0:
            means_normal.append(np.mean(arr, axis=0))
            stds_normal.append(np.std(arr, axis=0))
        else:
            means_anomaly.append(np.mean(arr, axis=0))
            stds_anomaly.append(np.std(arr, axis=0))
    except: pass

means_normal  = np.array(means_normal)
means_anomaly = np.array(means_anomaly)
stds_normal   = np.array(stds_normal)
stds_anomaly  = np.array(stds_anomaly)

n_feat = means_normal.shape[1]
feat_labels = [f'F{i}' for i in range(n_feat)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(n_feat)
w = 0.35

axes[0].bar(x - w/2, means_normal.mean(axis=0),  w, label='Normal',
            color=COLORS['normal'],  alpha=0.8, edgecolor='white')
axes[0].bar(x + w/2, means_anomaly.mean(axis=0), w, label='Anomali',
            color=COLORS['anomaly'], alpha=0.8, edgecolor='white')
axes[0].set_xticks(x); axes[0].set_xticklabels(feat_labels)
axes[0].set_ylabel('Rata-rata Nilai')
axes[0].set_title('Rata-rata Fitur: Normal vs Anomali')
axes[0].legend()

axes[1].bar(x - w/2, stds_normal.mean(axis=0),  w, label='Normal',
            color=COLORS['normal'],  alpha=0.8, edgecolor='white')
axes[1].bar(x + w/2, stds_anomaly.mean(axis=0), w, label='Anomali',
            color=COLORS['anomaly'], alpha=0.8, edgecolor='white')
axes[1].set_xticks(x); axes[1].set_xticklabels(feat_labels)
axes[1].set_ylabel('Standar Deviasi')
axes[1].set_title('Std Fitur: Normal vs Anomali')
axes[1].legend()

plt.suptitle('PLN — Perbandingan Statistik Fitur Normal vs Anomali', fontsize=11)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_pln_feature_stats.png', bbox_inches='tight')
plt.show()
"""),

        md("## 6. Heatmap Korelasi Fitur (Satu File Representatif)"),
        code("""\
# Pilih file normal terpanjang sebagai representasi
best_fname = max(normal_files, key=lambda f: np.load(data_dir / f).shape[0])
arr  = np.load(data_dir / best_fname)
corr = np.corrcoef(arr.T)   # (8, 8)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Korelasi Pearson')
ax.set_xticks(range(n_feat)); ax.set_xticklabels(feat_labels)
ax.set_yticks(range(n_feat)); ax.set_yticklabels(feat_labels)
for i in range(n_feat):
    for j in range(n_feat):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                fontsize=8, color='black')
ax.set_title(f'Heatmap Korelasi Fitur PLN ({best_fname})')
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_pln_correlation.png', bbox_inches='tight')
plt.show()
"""),

        md("## 7. Analisis Train / Val / Test Split"),
        code("""\
def read_list(fname):
    p = DATA_DIR / fname
    return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

train_files = read_list('train_list.txt')
val_files   = read_list('val_list.txt')
test_files  = read_list('test_list.txt')

label_map = dict(zip(file_labels['fname'], file_labels['label']))

def count_labels(files):
    n = sum(1 for f in files if label_map.get(f, 0) == 0)
    a = sum(1 for f in files if label_map.get(f, 0) == 1)
    return n, a

tn, ta = count_labels(train_files)
vn, va = count_labels(val_files)
sn, sa = count_labels(test_files)

splits = ['Train', 'Validasi', 'Test']
normals = [tn, vn, sn]
anomalies = [ta, va, sa]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3)
w = 0.35
ax.bar(x - w/2, normals,   w, label='Normal',  color=COLORS['normal'],  alpha=0.85)
ax.bar(x + w/2, anomalies, w, label='Anomali', color=COLORS['anomaly'], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel('Jumlah File')
ax.set_title('Distribusi Split Train / Validasi / Test — PLN')
ax.legend()
for i, (n, a) in enumerate(zip(normals, anomalies)):
    ax.text(i - w/2, n + 0.1, str(n), ha='center', fontsize=10)
    ax.text(i + w/2, a + 0.1, str(a), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(BASE / 'results' / 'figures' / 'eda_pln_split.png', bbox_inches='tight')
plt.show()

print(f"{'Split':10s} {'Normal':>8s} {'Anomali':>8s} {'Total':>8s}")
print("-" * 38)
for s, n, a in zip(splits, normals, anomalies):
    print(f"{s:10s} {n:>8d} {a:>8d} {n+a:>8d}")
"""),

        md("## 8. Ringkasan Dataset PLN"),
        code("""\
print("=" * 55)
print("RINGKASAN DATASET PLN")
print("=" * 55)
total = len(file_labels)
n_normal  = (file_labels['label'] == 0).sum()
n_anomaly = (file_labels['label'] == 1).sum()
print(f"Total file pelanggan         : {total}")
print(f"Normal (label 0)             : {n_normal}  ({n_normal/total*100:.1f}%)")
print(f"Anomali (label 1)            : {n_anomaly}  ({n_anomaly/total*100:.1f}%)")
print(f"Jumlah fitur per timestep    : {arr.shape[1]}")
print(f"Rata-rata panjang TS normal  : {np.mean(lengths_normal):.0f} timestep")
print(f"Rata-rata panjang TS anomali : {np.mean(lengths_anomaly):.0f} timestep")
print(f"Label anomali                : per FILE (bukan per timestep)")
print(f"Evaluasi CARLA               : agregasi window → file-level score")
print(f"\\nSplit:")
print(f"  Train   : {len(train_files)} file ({tn} normal, {ta} anomali)")
print(f"  Val     : {len(val_files)} file ({vn} normal, {va} anomali)")
print(f"  Test    : {len(test_files)} file ({sn} normal, {sa} anomali)")
"""),
    ]
    return nb(cells)


# ================================================================== #
#  Main
# ================================================================== #

if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Buat folder output figures jika belum ada
    fig_dir = Path(r'D:\CARLA\results\figures')
    print(f"\nMembuat {OUT_DIR} ...")

    print("\nGenerating notebooks...")
    save(make_msl(),  'eda_msl.ipynb')
    save(make_smap(), 'eda_smap.ipynb')
    save(make_lead(), 'eda_lead.ipynb')
    save(make_pln(),  'eda_pln.ipynb')
    print("\nSelesai! 4 notebook EDA telah dibuat di folder notebooks/")
    print("Jalankan: jupyter notebook notebooks/")
