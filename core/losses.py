import torch
import torch.nn as nn
import torch.nn.functional as F


# From https://github.com/adambielski/siamese-triplet/blob/master/losses.py
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = torch.squeeze(anchor)
        positive = torch.squeeze(positive)
        negative = torch.squeeze(negative)

        distance_positive = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), 1))
        distance_negative = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), 1))

        losses = F.relu((distance_positive - distance_negative) + self.margin)
        return torch.mean(losses)
