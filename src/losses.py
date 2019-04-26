#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import multi_dot
from sklearn.feature_selection import SelectKBest, chi2
import torch
import torch.nn as nn
from torch.autograd import Variable


# Defines the GAN loss which uses either LSGAN or the regular GAN. When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        # Get soft and noisy labels
        if target_is_real:
            target_tensor = 0.7 + 0.3 * torch.rand(input.size(0))
        else:
            target_tensor = 0.3 * torch.rand(input.size(0))
        if input.is_cuda:
            target_tensor = target_tensor.cuda()
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input.squeeze(), target_tensor)