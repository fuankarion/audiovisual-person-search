import os
import torch
import random
import numpy as np
from scipy.special import softmax

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from core.metrics import eval_triplet_loss


def batch_to_eval_data(device, a, p, n):
    distance_positive = torch.sqrt(torch.sum(torch.pow(a - p, 2), 1))
    distance_negative = torch.sqrt(torch.sum(torch.pow(a - n, 2), 1))

    ones = torch.ones(distance_positive.size(0)).to(device)
    ones_pred = torch.stack([ones, distance_positive], dim=1)

    zeros = torch.zeros(distance_positive.size(0)).to(device)
    zeros_pred = torch.stack([zeros, distance_negative], dim=1)

    return torch.cat([zeros_pred, ones_pred], dim=0)


def optimize_tripplet_loss(positive_prob, model, dataloader_train,
                           data_loader_val, device, criterion, optimizer,
                           scheduler, num_epochs, models_out=None, log=None):

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        loss_train, ap_train = _train_model_tripplet(positive_prob, model, dataloader_train, optimizer,
                                           criterion, scheduler, device)
        loss_val, ap_val = _test_model_tripplet(positive_prob, model,
                                                data_loader_val, criterion,
                                                device)

        if models_out is not None:
            model_target = os.path.join(models_out, str(epoch+1)+'.pth')
            print('save model to ', model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.writeDataLog([epoch+1, loss_train, ap_train, loss_val, ap_val])

    return model


def _train_model_tripplet(positive_prob, model, dataloader, optimizer,
                          criterion, scheduler, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    pred_lst = []
    label_lst = []
    softmax_layer = torch.nn.Softmax(dim=1)

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Train iter ', idx, '/', len(dataloader), end='\r')

        anchor_data, positive_data, negative_data = dl
        anchor_data = anchor_data.to(device)
        positive_data = positive_data.to(device)
        negative_data = negative_data.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            anchor, positive, negative = model(anchor_data, positive_data, negative_data)
            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()

        # eval
        with torch.set_grad_enabled(False):
            acc_tripplet = eval_triplet_loss(anchor, positive, negative)
            diff_negative = torch.squeeze((anchor-negative).pow(2).sum(1))
            diff_positive = torch.squeeze((anchor-positive).pow(2).sum(1))

            diff_out = torch.stack((diff_negative, diff_positive), dim=1)
            diff_out = (1-softmax_layer(diff_out)).cpu().numpy()

            for idx_out, a in enumerate(diff_out):
                if random.uniform(0, 1) < positive_prob:
                    label_lst.append(1)
                    pred_lst.append(diff_out[idx_out][1])
                else:
                    label_lst.append(0)
                    pred_lst.append(diff_out[idx_out][0])

        # statistics
        running_loss += loss.item()
        running_acc += acc_tripplet.item()

    scheduler.step()
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_auc = roc_auc_score(label_lst, pred_lst)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    print('')
    print('Loss: {:.4f}, Acc: {:.4f}, AUC: {:.4f}, Map: {:.4f}'.format(epoch_loss, epoch_acc, epoch_auc, epoch_ap))
    return epoch_loss, epoch_ap


def _test_model_tripplet(positive_prob, model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    gts_list = []
    preds_list = []

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Test iter ', idx, '/', len(dataloader), end='\r')

        anchor_data, positive_data, negative_data = dl

        anchor_data = anchor_data.to(device)
        positive_data = positive_data.to(device)
        negative_data = negative_data.to(device)

        with torch.set_grad_enabled(False):
            anchor, positive, negative = model(anchor_data, positive_data, negative_data)
            loss = criterion(anchor, positive, negative)

            anchor = torch.squeeze(anchor)
            positive = torch.squeeze(positive)
            negative = torch.squeeze(negative)
            eval_data = batch_to_eval_data(device, anchor, positive, negative)
            gts_list.extend(eval_data[:, 0].cpu().detach().numpy().tolist())
            preds_list.extend(eval_data[:, 1].cpu().detach().numpy().tolist())

        # statistics
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_ap = average_precision_score(gts_list, 1-softmax(np.asarray(preds_list)))
    print('')
    print('Loss: {:.4f}, Map: {:.4f}'.format(epoch_loss, epoch_ap))
    return epoch_loss, epoch_ap
