import argparse
import os
import torch
import numpy as np
import pandas
from utils.mypath import MyPath

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_val_transformations1, get_optimizer,\
                                adjust_learning_rate, inject_point_anomaly, inject_sub_anomaly, inject_sub_anomaly2
from utils.evaluate_utils import contrastive_evaluate
from utils.repository import TSRepository
from utils.train_utils import pretext_train
from utils.utils import fill_ts_repository
from termcolor import colored
from statsmodels.tsa.stattools import adfuller

# Parser
parser = argparse.ArgumentParser(description='pretext')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--fname',
                    help='Config the file name of Dataset')
args = parser.parse_args()

def main():

    print(colored('CARLA Pretext stage --> ', 'yellow'))

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.fname)
    #print(colored(p, 'red'))
    
    # Model
    #print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    # model = model.cuda()
   
    # CUDNN
    #print(colored('Set CuDNN benchmark', 'blue'))
    # torch.backends.cudnn.benchmark = True
    
    # Dataset
    #print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)

    panomaly = inject_point_anomaly(p)
    sanomaly = inject_sub_anomaly(p)
    sanomaly2 = inject_sub_anomaly2(p)
    val_transforms = get_val_transformations1(p)


    if p['train_db_name'] == 'MSL' or p['train_db_name'] == 'SMAP':
        if p['fname'] == 'All':
            with open(os.path.join(MyPath.db_root_dir('msl'), 'labeled_anomalies.csv'), 'r') as file:
                csv_reader = pandas.read_csv(file, delimiter=',')
            data_info = csv_reader[csv_reader['spacecraft'] == p['train_db_name']]
            ii = 0
            for file_name in data_info['chan_id']:
                p['fname'] = file_name
                if ii == 0 :
                    train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True,
                                                  split='train+unlabeled')
                    val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, train_dataset.mean,
                                              train_dataset.std)
                    base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True,
                                                     split='train')
                else:
                    new_train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True,
                                                  split='train+unlabeled')
                    new_val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, new_train_dataset.mean,
                                                  new_train_dataset.std)

                    train_dataset.concat_ds(new_train_dataset)
                    val_dataset.concat_ds(new_val_dataset)
                    base_dataset.concat_ds(new_train_dataset)

                ii += 1
        else:
            train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True,
                                              split='train+unlabeled')
            val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, train_dataset.mean,
                                          train_dataset.std)
            base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True,
                                             split='train') # Dataset w/o augs for knn eval

    elif p['train_db_name'] == 'yahoo':
        filename = os.path.join('datasets', 'A1Benchmark/', p['fname'])
        dataset = []

        print(filename)
        df = pandas.read_csv(filename)
        dataset.append({
            'value': df['value'].tolist(),
            'label': df['is_anomaly'].tolist()
        })

        ts = dataset[0]
        data = np.array(ts['value'])
        labels = np.array(ts['label'])
        l = len(data) // 2

        n = 0
        while adfuller(data[:l], 1)[1] > 0.05 or adfuller(data[:l])[1] > 0.05:
            data = np.diff(data)
            labels = labels[1:]
            n += 1
        l -= n

        all_train_data = data[:l]
        all_test_data = data[l:]
        all_train_labels = labels[:l]
        all_test_labels= labels[l:]

        #mean, std = all_train_data.mean(), all_train_data.std()
        #all_train_data = (all_train_data - mean) / std    normalized after augmentation
        #all_test_data = (all_test_data - mean) / std

        TRAIN_TS = all_train_data
        TEST_TS = all_test_data
        train_label = all_train_labels
        test_label = all_test_labels

        print(">>>", "train/test w. shapes of {}/{}".format(np.shape(TRAIN_TS), np.shape(TEST_TS)))

        train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2,
                                          to_augmented_dataset=True, data=TRAIN_TS, label=train_label)
        val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, train_dataset.mean,
                                          train_dataset.std, TEST_TS, test_label)
        base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2,
                                          to_augmented_dataset=True, data=TRAIN_TS, label=train_label)

    elif p['train_db_name'] == 'smd':
        train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)
        val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, train_dataset.mean,
                                      train_dataset.std)
        base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)

    elif p['train_db_name'] == 'swat':
        train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)
        val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, train_dataset.mean,
                                      train_dataset.std)
        base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)

    elif p['train_db_name'] == 'wadi':
        train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)
        val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, train_dataset.mean,
                                      train_dataset.std)
        base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)

    elif p['train_db_name'] == 'kpi':
        train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)
        val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, train_dataset.mean,
                                      train_dataset.std)
        base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)

    elif p['train_db_name'] == 'Power':
        train_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)
        val_dataset = get_val_dataset(p, val_transforms, panomaly, sanomaly, sanomaly2, False, train_dataset.mean,
                                      train_dataset.std)
        base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, sanomaly2, to_augmented_dataset=True)


    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    base_dataloader = get_val_dataloader(p, base_dataset)

    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    
    # TS Repository
   # base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, to_augmented_dataset=True, split='train')

    ts_repository_base = TSRepository(len(base_dataset),
                                      p['model_kwargs']['features_dim'],
                                      p['num_classes'], p['criterion_kwargs']['temperature'])
    # ts_repository_base.cuda()
    ts_repository_val = TSRepository(len(val_dataset),
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature'])
    # ts_repository_val.cuda()

    # Criterion
    #print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    # criterion = criterion.cuda()

    # Optimizer and scheduler
    #print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        # model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        # model = model.cuda()
    
    # Training
    #print(colored('Starting main loop', 'blue'))
    pretext_best_loss = np.inf
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')
        tmp_loss = pretext_train(train_dataloader, model, criterion, optimizer, epoch)

        # Fill ts repository
        # print('Fill TS Repository for kNN/kFN ...')
        # fill_ts_repository(p, base_dataloader, model, ts_repository_base, real_aug =False, ts_repository_aug=None)
        
        # Checkpoint
        if tmp_loss <= pretext_best_loss:
            pretext_best_loss = tmp_loss
            print('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1}, p['pretext_checkpoint'])
            torch.save(model.state_dict(), p['pretext_model'])

    # Save final model
    checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': p['epochs']}, p['pretext_checkpoint'])
    #torch.save(model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the classification loss.
    print(colored('Fill TS Repository for mining the nearest/furthest neighbors (train) ...', 'blue'))
    ts_repository_aug = TSRepository(len(base_dataset) * 2,
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature']) #need size of repository == 1+num_of_anomalies
    fill_ts_repository(p, base_dataloader, model, ts_repository_base, real_aug = True, ts_repository_aug = ts_repository_aug)
    out_pre = np.column_stack((ts_repository_base.features, ts_repository_base.targets))
    np.save(p['pretext_features_train_path'], out_pre)
    topk = 10
    print('Mine the nearest neighbors (Top-%d)' %(topk))
    kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk)
    #print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], knearest)
    np.save(p['bottomk_neighbors_train_path'], kfurtherst)

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill TS Repository for mining the nearest/furthest neighbors (val) ...', 'blue'))

    fill_ts_repository(p, val_dataloader, model, ts_repository_val, real_aug =False, ts_repository_aug= None)
    out_pre = np.column_stack((ts_repository_val.features, ts_repository_val.targets))
    np.save(p['pretext_features_test_path'], out_pre)
    topk = 10
    print('Mine the nearest neighbors (Top-%d)' %(topk))
    kfurtherst, knearest = ts_repository_val.furthest_nearest_neighbors(topk)
    #print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], knearest)
    np.save(p['bottomk_neighbors_val_path'], kfurtherst)

 
if __name__ == '__main__':
    main()
