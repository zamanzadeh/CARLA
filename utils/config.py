
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing

def create_config(config_file_env, config_file_exp, fname):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    # NOVELTY-AWARE PATH SUFFIX (backward compatible):
    # Sebelumnya, direktori checkpoint dibentuk HANYA dari train_db_name + fname.
    # Akibatnya, melatih PLN dengan anomaly_kwargs.type='theft' lalu 'subseq' (atau
    # sebaliknya) menulis ke path checkpoint yang SAMA PERSIS -> run kedua menimpa
    # model run pertama, dan evaluate.py selalu membaca model TERAKHIR yang dilatih,
    # bukan model yang sesuai konfigurasi yang sedang dievaluasi. Ini membuat hasil
    # evaluasi 'theft' vs 'subseq' tampak identik (sebenarnya cuma 1 model yang ada).
    #
    # Fix: tambahkan suffix ke nama folder HANYA kalau anomaly_kwargs.type diisi
    # dengan nilai selain default 'subseq'. Kalau key 'type' tidak ada sama sekali
    # (semua config lama: SMD/MSL/SMAP/LEAD, atau config PLN sebelum novelty ini),
    # suffix kosong -> path checkpoint identik dengan sebelumnya, TIDAK ada breaking
    # change untuk eksperimen/checkpoint yang sudah ada.
    anomaly_type = cfg.get('anomaly_kwargs', {}).get('type', 'subseq')
    dir_suffix = '' if anomaly_type in (None, 'subseq') else '_{}'.format(anomaly_type)
    dir_fname = fname + dir_suffix

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, dir_fname+'/pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['fname'] = fname
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
    cfg['bottomk_neighbors_train_path'] = os.path.join(pretext_dir, 'bottomk-train-neighbors.npy')
    cfg['aug_train_dataset'] = os.path.join(pretext_dir, 'aug_train_dataset.pth')
    cfg['pretext_features_train_path'] = os.path.join(pretext_dir, 'pretext_features_train.npy')
    cfg['pretext_features_test_path'] = os.path.join(pretext_dir, 'pretext_features_test.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-test-neighbors.npy')
    cfg['bottomk_neighbors_val_path'] = os.path.join(pretext_dir, 'bottomk-test-neighbors.npy')
 #  cfg['bottomk_neighbors_val_path'] = os.path.join(pretext_dir, 'bottomk-test-neighbors.npy')
    cfg['contrastive_dataset'] = os.path.join(pretext_dir, 'con_train_dataset.pth')


    if cfg['setup'] in ['classification']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        classification_dir = os.path.join(base_dir, dir_fname+ '/classification')
        mkdir_if_missing(base_dir)
        mkdir_if_missing(classification_dir)
        cfg['classification_dir'] = classification_dir
        cfg['classification_checkpoint'] = os.path.join(classification_dir, 'checkpoint.pth.tar')
        cfg['classification_model'] = os.path.join(classification_dir, 'model.pth.tar')
        cfg['classification_trainfeatures'] = os.path.join(classification_dir, 'classification_trainfeatures.csv')
        cfg['classification_trainprobs'] = os.path.join(classification_dir, 'classification_trainprobs.csv')
        cfg['classification_testfeatures'] = os.path.join(classification_dir, 'classification_testfeatures.csv')
        cfg['classification_testprobs'] = os.path.join(classification_dir, 'classification_testprobs.csv')

    return cfg 
