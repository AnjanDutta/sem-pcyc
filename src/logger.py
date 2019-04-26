#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Log Metric.
"""

import torch
from tensorboardX import SummaryWriter
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, log_dir, force=False):
        # clean previous logged data under the same directory name
        self._remove(log_dir, force)

        # create the summary writer object
        self._writer = SummaryWriter(log_dir)

        self.global_step = 0

    def __del__(self):
        self._writer.close()

    def add_scalar(self, name, scalar_value):
        assert isinstance(scalar_value, float), type(scalar_value)
        self._writer.add_scalar(name, scalar_value, self.global_step)

    def add_image(self, name, img_tensor):
        assert isinstance(img_tensor, torch.Tensor), type(img_tensor)
        self._writer.add_image(name, img_tensor, self.global_step)

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path, force):
        """ param <path> could either be relative or absolute. """
        if not os.path.exists(path):
            return
        elif os.path.isfile(path) and force:
            os.remove(path)  # remove the file
        elif os.path.isdir(path) and force:
            import shutil
            shutil.rmtree(path)  # remove dir and all contains
        else:
            print('Logdir contains data. Please, set `force` flag to overwrite it.')
            import sys
            sys.exit(0)