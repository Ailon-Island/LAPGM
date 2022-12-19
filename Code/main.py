# coding: utf-8
"""
==========================================================
Matching Image Keypoints by Neural Network Methods
==========================================================

This code contains data loading, training, and testing for Neural Network methods (especially for PCA-GM) with the aid of pygmtools and Jittor.
"""

# Author: Wei Jiang <ailon_jw@sjtu.edu.cn>
#
# Adopted from example in document of pygmtools by Runzhong Wang.


# Import Packages and Basic Options
# ---------------------------------
import os

from PIL import Image
import numpy as np

import jittor
from jittor.optim import AdamW, Adam, SGD
from jittor.lr_scheduler import StepLR, CosineAnnealingLR
import pygmtools as pygm

from dataset import GMDataset
from options import Options
from model import LAPGM
from trainer import Trainer

from utils import DummyScheduler


def main():
    # Create Benchmarks
    # -----------------
    # Now we create `benchmark` objects to get dataset and interact with the model.
    train_benchmark = pygm.benchmark.Benchmark(name=opt.dataset, sets='train',
                                               obj_resize=opt.obj_resize) \
        if opt.train else None
    test_benchmark = pygm.benchmark.Benchmark(name=opt.dataset, sets='test',
                                              obj_resize=opt.obj_resize)

    # Create Datasets
    # ---------------
    train_data = GMDataset(opt, train_benchmark) if opt.train else None
    test_data = GMDataset(opt, test_benchmark)

    # Create Model
    # ------------
    # We create a model object and load the pre-trained model if needed.
    model = LAPGM(opt)

    # Create Optimizer and Scheduler
    # ------------------------------
    optimizer, scheduler = None, None
    if opt.train:
        if opt.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'sgd':
            optimizer = SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        if opt.lr_scheduler == 'none':
            scheduler = DummyScheduler()
        elif opt.lr_scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=opt.lr_schedule_iter, gamma=opt.lr_schedule_gamma)
        elif opt.lr_scheduler == 'cosine':
            total_iters = min(opt.epochs * len(train_data), opt.max_iters)
            scheduler = CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=0.)
        else:
            raise NotImplementedError(f'Unknown learning rate scheduler: {opt.lr_scheduler}')

    # Train Model
    # -----------
    # We create a `trainer` object and train the model.
    trainer = Trainer(opt, model, optimizer, scheduler, test_data, test_benchmark, train_data, train_benchmark)
    if opt.train:
        trainer.train()
    else:
        for _ in range(10):
            trainer.test()


if __name__ == '__main__':
    pygm.BACKEND = 'jittor'  # set Jittor as the default backend for pygmtools
    jittor.flags.use_cuda = jittor.has_cuda
    opt = Options().parse()  # parse options
    jittor.set_global_seed(opt.seed)  # set global random seed

    main()
