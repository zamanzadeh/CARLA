
import torch
import numpy as np
from torch import Tensor

from utils.utils import AverageMeter, ProgressMeter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pretext_train(train_loader, model, criterion, optimizer, epoch, prev_loss, device='cuda'):

    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch+1))

    model.to(device)
    model.train()

    for i, batch in enumerate(train_loader):
        ts_org = batch['ts_org'].float().to(device, non_blocking=True)
        ts_w_augmented = batch['ts_w_augment'].float().to(device, non_blocking=True)
        ts_ss_augmented = batch['ts_ss_augment'].float().to(device, non_blocking=True)

        if ts_org.ndim == 3:
            b, w, h = ts_org.shape
        else:
            b, w = ts_org.shape
            h =1

        input_: Tensor = torch.cat([ts_org, ts_w_augmented, ts_ss_augmented], dim=0).view(b * 3, h, w)

        output = model(input_)
        
        if prev_loss is not None:
            loss = criterion(output, prev_loss)
        else:
            loss = criterion(output)

        losses.update(loss.item())
        prev_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            progress.display(i)

    return loss


def self_sup_classification_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """ 
    Train w/ classification-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    inconsistency_losses = AverageMeter('Inconsistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, inconsistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch+1))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].to(device) #.cuda(non_blocking=True)
        nneighbors = batch['NNeighbor'].to(device) #.cuda(non_blocking=True)
        fneighbors = batch['FNeighbor'].to(device)  #.cuda(non_blocking=True)

        if anchors.ndim == 3:
            b, w, h = anchors.shape
        else:
            b, w = anchors.shape
            h =1

        anchors = anchors.reshape(b, h, w)
        nneighbors = nneighbors.reshape(b, h, w)
        fneighbors = fneighbors.reshape(b, h, w)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                nneighbors_features = model(nneighbors, forward_pass='backbone')
                fneighbors_features = model(fneighbors, forward_pass='backbone')

            anchors_output = model(anchors_features, forward_pass='head')
            nneighbors_output = model(nneighbors_features, forward_pass='head')
            fneighbors_output = model(fneighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            nneighbors_output = model(nneighbors)
            fneighbors_output = model(fneighbors)

            # Ensure model output is iterable
        if not isinstance(anchors_output, (list, tuple)):
            anchors_output = [anchors_output]
            nneighbors_output = [nneighbors_output]
            fneighbors_output = [fneighbors_output]

        # Loss for every head
        total_loss, consistency_loss, inconsistency_loss, entropy_loss = [], [], [], []
        for anchors_output_subhead, nneighbors_output_subhead, fneighbors_output_subhead in zip(anchors_output, nneighbors_output, fneighbors_output):
            total_loss_, consistency_loss_, inconsistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         nneighbors_output_subhead, fneighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            inconsistency_loss.append(inconsistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Aggregate losses and check for NaN
        total_loss_values = [v.item() for v in total_loss if not torch.isnan(v)]
        consistency_loss_values = [v.item() for v in consistency_loss if not torch.isnan(v)]
        inconsistency_loss_values = [v.item() for v in inconsistency_loss if not torch.isnan(v)]
        entropy_loss_values = [v.item() for v in entropy_loss if not torch.isnan(v)]

        if total_loss_values:
            total_losses.update(np.mean(total_loss_values))
        if consistency_loss_values:
            consistency_losses.update(np.mean(consistency_loss_values))
        if inconsistency_loss_values:
            inconsistency_losses.update(np.mean(inconsistency_loss_values))
        if entropy_loss_values:
            entropy_losses.update(np.mean(entropy_loss_values))

        total_loss_final = torch.sum(torch.stack(total_loss, dim=0))
        assert total_loss_final.requires_grad, "Total loss does not require grad!"

        optimizer.zero_grad()
        total_loss_final.backward()
        optimizer.step()

        if i % 100 == 0:
            progress.display(i)
