#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import utils
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='SEM-PCYC for Zero-Shot Sketch-based Image Retrieval.')
        # Optional argument
        parser.add_argument('--dataset', required=True, default='Sketchy', help='Name of the dataset')
        # Different training test sets
        parser.add_argument('--split-eccv-2018', action='store_true', default=False,
                            help='Whether to use the splits of ECCV 2018 paper')
        parser.add_argument('--gzs-sbir', action='store_true', default=False,
                            help='Generalized zero-shot sketch based image retrieval')
        parser.add_argument('--filter-sketch', action='store_true', default=False, help='Allows only one sketch per '
                                                                                        'image (only for Sketchy)')
        # Semantic models
        parser.add_argument('--semantic-models', nargs='+', default=['word2vec-google-news', 'hieremb-path'],
                            type=str, help='Semantic model')
        # Weight (on loss) parameters
        parser.add_argument('--lambda-se', default=10.0, type=float, help='Weight on the semantic model')
        parser.add_argument('--lambda-im', default=10.0, type=float, help='Weight on the image model')
        parser.add_argument('--lambda-sk', default=10.0, type=float, help='Weight on the sketch model')
        parser.add_argument('--lambda-gen-cyc', default=1.0, type=float, help='Weight on cycle consistency loss (gen)')
        parser.add_argument('--lambda-gen-adv', default=1.0, type=float, help='Weight on adversarial loss (gen)')
        parser.add_argument('--lambda-gen-cls', default=1.0, type=float, help='Weight on classification loss (gen)')
        parser.add_argument('--lambda-gen-reg', default=0.1, type=float, help='Weight on regression loss (gen)')
        parser.add_argument('--lambda-disc-se', default=0.25, type=float, help='Weight on semantic loss (disc)')
        parser.add_argument('--lambda-disc-sk', default=0.5, type=float, help='Weight on sketch loss (disc)')
        parser.add_argument('--lambda-disc-im', default=0.5, type=float, help='Weight on image loss (disc)')
        parser.add_argument('--lambda-regular', default=0.001, type=float, help='Weight on regularizer')
        # Size parameters
        parser.add_argument('--im-sz', default=224, type=int, help='Image size')
        parser.add_argument('--sk-sz', default=224, type=int, help='Sketch size')
        parser.add_argument('--dim-out', default=128, type=int, help='Output dimension of sketch and image')
        # Model parameters
        parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
        parser.add_argument('--epoch-size', default=100, type=int, help='Epoch size')
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of workers in data loader')
        # Checkpoint parameters
        parser.add_argument('--test', action='store_true', default=False, help='Test only flag')
        parser.add_argument('--early-stop', type=int, default=20, help='Early stopping epochs.')
        # Optimization parameters
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='Number of epochs to train (default: 100)')
        parser.add_argument('--lr', type=lambda x: utils.restricted_float(x, [1e-5, 0.5]), default=0.0001, metavar='LR',
                            help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--milestones', type=int, nargs='+', default=[], help='Milestones for scheduler')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule steps.')
        # I/O parameters
        parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                            help='How many batches to wait before logging training status')
        parser.add_argument('--save-image-results', action='store_true', default=False, help='Whether to save image '
                                                                                             'results')
        parser.add_argument('--number-qualit-results', type=int, default=200, help='Number of qualitative results to be'
                                                                                   ' saved')
        parser.add_argument('--save-best-results', action='store_true', default=False, help='Whether to save the best '
                                                                                            'results')
        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
