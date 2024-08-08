
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

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, fname+'/pretext')
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
    cfg['bottomk_neighbors_val_path'] = os.path.join(pretext_dir, 'bottomk-test-neighbors.npy')
    cfg['contrastive_dataset'] = os.path.join(pretext_dir, 'con_train_dataset.pth')


    if cfg['setup'] in ['classification']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        classification_dir = os.path.join(base_dir, fname+ '/classification')
        mkdir_if_missing(base_dir)
        mkdir_if_missing(classification_dir)
        cfg['classification_dir'] = classification_dir
        cfg['classification_checkpoint'] = os.path.join(classification_dir, 'checkpoint.pth.tar')
        cfg['classification_model'] = os.path.join(classification_dir, 'model.pth.tar')
        cfg['classification_trainfeatures'] = os.path.join(classification_dir, 'classification_traintfeatures.csv')
        cfg['classification_trainprobs'] = os.path.join(classification_dir, 'classification_trainprobs.csv')
        cfg['classification_testfeatures'] = os.path.join(classification_dir, 'classification_testtfeatures.csv')
        cfg['classification_testprobs'] = os.path.join(classification_dir, 'classification_testprobs.csv')

    return cfg 
