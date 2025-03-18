
import os
import torch
import numpy as np
import errno
from data.ra_dataset import SaveAugmentedDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_ts_repository(p, loader, model, ts_repository, real_aug=False, ts_repository_aug=None):
    model.eval()
    ts_repository.reset()
    if ts_repository_aug != None: ts_repository_aug.reset()
    if real_aug:
        ts_repository.resize(3)

    con_data = torch.tensor([]).to(device)
    con_target = torch.tensor([]).to(device)
    for i, batch in enumerate(loader): 
        ts_org = batch['ts_org'].to(device, non_blocking=True) #cuda
        targets = batch['target'].to(device, non_blocking=True)
        if ts_org.ndim == 3:
            b, w, h = ts_org.shape
        else:
            b, w = ts_org.shape
            h = 1

        # ts_org = torch.from_numpy(ts_org).float(). #cuda
        output = model(ts_org.reshape(b, h, w))
        ts_repository.update(output, targets)
        if ts_repository_aug != None: ts_repository_aug.update(output, targets)
        if i % 100 == 0:
            print('Fill TS Repository [%d/%d]' %(i, len(loader)))

        if real_aug:
            con_data = torch.cat((con_data, ts_org), dim=0)
            # con_target = torch.cat((con_target, torch.from_numpy(targets).float()), dim=0)
            con_target = torch.cat((con_target, targets), dim=0) #cuda


            ts_w_augment = batch['ts_w_augment'].to(device, non_blocking=True) #cuda
            targets = torch.LongTensor([2]*ts_w_augment.shape[0]).to(device, non_blocking=True)
            # ts_w_augment = torch.from_numpy(ts_w_augment).float() #cuda
            output = model(ts_w_augment.reshape(b, h, w))
            ts_repository.update(output, targets)
            # ts_repository_aug.update(output, targets)


            ts_ss_augment = batch['ts_ss_augment'].to(device, non_blocking=True) #cuda
            targets = torch.LongTensor([4]*ts_ss_augment.shape[0]).to(device, non_blocking=True)
            # ts_ss_augment = torch.from_numpy(ts_ss_augment).float() #cuda
            con_data = torch.cat((con_data, ts_ss_augment), dim=0)
            con_target = torch.cat((con_target, targets), dim=0)
            output = model(ts_ss_augment.reshape(b, h, w))
            ts_repository.update(output, targets)
            ts_repository_aug.update(output, targets)


    if real_aug:
        con_dataset = SaveAugmentedDataset(con_data, con_target)
        con_loader = torch.utils.data.DataLoader(con_dataset, num_workers=p['num_workers'],
                                                 batch_size=p['batch_size'], pin_memory=True,
                                                 drop_last=False, shuffle=False)
        torch.save(con_loader, p['contrastive_dataset'])
