import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from data.augment import NoiseTransformation, SubAnomaly
from utils.collate import collate_custom


def get_criterion(p):
    if p['criterion'] == 'pretext':
        from losses.losses import PretextLoss
        criterion = PretextLoss(p['batch_size'], **p['criterion_kwargs'])

    elif p['criterion'] == 'classification':
        from losses.losses import ClassificationLoss
        criterion = ClassificationLoss(**p['criterion_kwargs'])

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 8

    elif p['backbone'] == 'resnet_ts':
        return 8

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'resnet_ts':
        from models.resent_time import resnet_ts
        backbone = resnet_ts(**p['res_kwargs'])

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    if p['setup'] in ['pretext']:
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    elif p['setup'] in ['classification']:
        from models.models import ClusteringModel
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])

    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')

        if p['setup'] == 'classification':  # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state, strict=False)
            assert (set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias',
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                    or set(missing[1]) == {
                        'contrastive_head.weight', 'contrastive_head.bias'})

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model


def get_train_dataset(p, transform, sanomaly, to_augmented_dataset=False,
                      to_neighbors_dataset=False, split=None, data=None, label=None):
    # Base dataset
    mean, std = 0, 0
    if p['train_db_name'] == 'MSL' or p['train_db_name'] == 'SMAP':
        from data.MSL import MSL
        dataset = MSL(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None)
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'yahoo':
        from data.Yahoo import Yahoo
        dataset = Yahoo(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                        data=data, label=label)
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'kpi':
        from data.KPI import KPI
        dataset = KPI(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                       mean_data=None, std_data=None)
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'smd':
        from data.SMD import SMD
        dataset = SMD(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None)
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'swat':
        from data.SWAT import SWAT
        dataset = SWAT(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None)
        mean, std = dataset.get_info()
    elif p['train_db_name'] == 'wadi':
        from data.WADI import WADI
        dataset = WADI(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None)
        mean, std = dataset.get_info()

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset:  # Dataset returns a ts and an augmentation of that.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset:  # Dataset returns ts and its nearest and furthest neighbors.
        from data.custom_dataset import NeighborsDataset
        nindices = np.load(p['topk_neighbors_train_path'])
        findices = np.load(p['bottomk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, None, nindices, findices, p)

    dataset.mean = mean
    dataset.std = std
    return dataset


def get_aug_train_dataset(p, transform, to_neighbors_dataset=False):
    dataloader = torch.load(p['contrastive_dataset'])
    if to_neighbors_dataset:  # Dataset returns a ts and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        N_indices = np.load(p['topk_neighbors_train_path'])
        F_indices = np.load(p['bottomk_neighbors_train_path'])
        dataset = NeighborsDataset(dataloader.dataset, transform, N_indices, F_indices, p)

    return dataset


def get_val_dataset(p, transform=None, sanomaly=None, to_neighbors_dataset=False,
                    mean_data=None, std_data=None, data=None, label=None):
    # Base dataset
    if p['val_db_name'] == 'MSL' or p['val_db_name'] == 'SMAP':
        from data.MSL import MSL
        dataset = MSL(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data)

    elif p['train_db_name'] == 'yahoo':
        from data.Yahoo import Yahoo
        dataset = Yahoo(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                        mean_data=mean_data, std_data=std_data, data=data, label=label)

    elif p['train_db_name'] == 'kpi':
        from data.KPI import KPI
        dataset = KPI(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data)

    elif p['val_db_name'] == 'smd':
        from data.SMD import SMD
        dataset = SMD(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data)

    elif p['val_db_name'] == 'swat':
        from data.SWAT import SWAT
        dataset = SWAT(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data)

    elif p['val_db_name'] == 'wadi':
        from data.WADI import WADI
        dataset = WADI(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data)

    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))

    # Wrap into other dataset (__getitem__ changes) 
    if to_neighbors_dataset:  # Dataset returns a ts and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        N_indices = np.load(p['topk_neighbors_val_path'])
        F_indices = np.load(p['bottomk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, transform, N_indices, F_indices, 5)  # Only use 5

    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                       drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                       drop_last=False, shuffle=False)


def inject_sub_anomaly(p):
    return SubAnomaly(p['anomaly_kwargs']['portion'])


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'ts':
        return transforms.Compose([
            NoiseTransformation(p['transformation_kwargs']['noise_sigma']),
            # Crop(p['transformation_kwargs']['crop_size'])
        ])

    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    return transforms.Compose([
        transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(**p['transformation_kwargs']['normalize'])])


def get_val_transformations1(p):
    return transforms.Compose([
        NoiseTransformation(p['transformation_kwargs']['noise_sigma'])
    ])


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only:  # Only weights in the cluster head will be updated
        for name, param in model.named_parameters():
            if 'cluster_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert (len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
