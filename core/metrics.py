import torch


def eval_triplet_loss(anchor, positive, negative):
    diff_positive = torch.squeeze((anchor-positive).pow(2).sum(1))
    diff_negative = torch.squeeze((anchor-negative).pow(2).sum(1))

    gt = diff_negative > diff_positive
    return (gt.double().sum())/len(gt)
