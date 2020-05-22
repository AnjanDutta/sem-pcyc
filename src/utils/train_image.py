#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import numpy as np
import argparse
import os
import time
import configparser as cp
import socket

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

# user defined
import utils
from data import DataGeneratorImage
from models import VGGNetFeats, ResNet50Feats, SEResNet50Feats
from logger import Logger, AverageMeter


def main():

    global args

    # Parse options for processing
    parser = argparse.ArgumentParser(description='Image Model.')

    # Optional argument
    parser.add_argument('--dataset', default='Sketchy', help='Name of the dataset')
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume from latest checkpoint')
    parser.add_argument('--model', default='SEResNet50', help='Type of model')
    parser.add_argument('--im-sz', default=224, type=int, help='image size')

    # Optimization Options
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
    parser.add_argument('--multi-gpu', action='store_true', default=False, help='Enables Multiple GPU')
    parser.add_argument('--early-stop', type=int, default=10, help='Early stopping epochs.')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=lambda x: utils.restricted_float(x, [1e-5, 0.5]), default=0.01, metavar='LR', help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
    parser.add_argument('--lr-decay', type=lambda x: utils.restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY', help='Learning rate decay factor [.01, 1] (default: 0.6)')
    parser.add_argument('--dropout', action='store_true', default=False, help='Introduce dropout in the layers')
    parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S', help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

    # I/O
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='How many batches to wait before logging training status')

    # Parse the arguments
    args = parser.parse_args()
    print('Parameters:\t' + str(args))

    assert (args.model in ['VGGNet', 'ResNet50', 'SEResNet50'])

    # Manual seed
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Read the config file and
    config = utils.read_config()
    path_dataset = config['path_dataset']
    path_aux = config['path_aux']

    # modify the log and check point paths
    root_path = os.path.join(path_dataset, args.dataset)
    path_cp = os.path.join(path_aux, 'CheckPoints', args.dataset, 'image', args.model)
    path_log = os.path.join(path_aux, 'LogFiles', args.dataset, 'image', args.model)

    # Parameters for transforming the images
    im_mean = [0.485, 0.456, 0.406]  # RGB channel mean for Imagenet
    im_std = [0.229, 0.224, 0.225]  # RGB channel std for Imagenet
    transform = transforms.Compose([transforms.Resize((args.im_sz, args.im_sz)), transforms.ToTensor(), transforms.Normalize(im_mean, im_std)])

    # Load the dataset
    print('Loading data...', end='')

    if args.dataset == 'Sketchy':
        photo_dir = 'extended_photo'
        if photo_dir == 'extended_photo':
            photo_sd = ''
        else:
            photo_sd = 'tx_000000000000'
        splits = utils.load_files_sketchy(root_path=root_path, photo_dir=photo_dir, photo_sd=photo_sd)
    elif args.dataset == 'TU-Berlin':
        photo_dir = 'images'
        photo_sd = ''
        splits = utils.load_files_tuberlin(root_path=root_path, photo_dir=photo_dir, photo_sd=photo_sd)
    elif args.dataset == 'QuickDraw':
        raise NotImplementedError
    else:
        print('Wrong dataset.')
        exit()

    # class dictionary
    dict_clss = utils.create_dict_texts(splits['tr_clss_im'] + splits['va_clss_im'] + splits['te_clss_im'])
    tr_clss = utils.numeric_classes(splits['tr_clss_im'], dict_clss)
    va_clss = utils.numeric_classes(splits['va_clss_im'], dict_clss)
    te_clss = utils.numeric_classes(splits['te_clss_im'], dict_clss)

    # generators
    data_train = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['tr_fls_im'], tr_clss, transforms=transform)
    data_valid = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['va_fls_im'], va_clss, transforms=transform)
    data_test = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['te_fls_im'], te_clss, transforms=transform)
    print('Done')

    # compute class weights
    weights = torch.DoubleTensor(utils.class_weights(tr_clss))
    sampler = WeightedRandomSampler(weights, num_samples=len(data_train))

    # PyTorch train loader
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=4, pin_memory=True)
    # PyTorch valid loader
    valid_loader = DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # PyTorch test loader
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    print('Initializing model...', end='')
    if args.model == 'ResNet50':
        model = ResNet50Feats(num_classes=len(dict_clss), pretrained=True, finetune_feats=False, finetune_class=True)
    elif args.model == 'SEResNet50':
        model = SEResNet50Feats(num_classes=len(dict_clss), pretrained=True, finetune_feats=False, finetune_class=True)
    elif args.model == 'VGGNet':
        model = VGGNetFeats(num_classes=len(dict_clss), pretrained=True, finetune_feats=False, finetune_class=True)
    else:
        raise NotImplementedError
    print('Done')

    # Optimizer
    print('Defining optimizer...', end='')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print('Done')

    # Loss function
    print('Defining loss function...', end='')
    criterion = nn.CrossEntropyLoss()
    print('Done')

    # Logger
    print('Setting logger...', end='')
    logger = Logger(path_log, force=True)
    print('Done')

    cudnn.benchmark = True

    # Argument to variable
    lr = args.lr

    # Decay lr
    lr_step = (args.lr - args.lr * args.lr_decay) / (args.epochs * args.schedule[1] - args.epochs * args.schedule[0])

    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if args.ngpu > 0 & torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        model = model.cuda()
    print('Done')

    best_acc = 0
    early_stop_counter = 0

    # Epoch for loop
    for epoch in range(args.epochs):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train on training set
        train(train_loader, model, criterion, optimizer, epoch, logger)

        # evaluate on validation set
        with torch.no_grad():
            acc = validate(valid_loader, model, criterion, epoch, logger)

        print('Accuracy on validation set after {0} epochs: {1:.4f}'.format(epoch + 1, acc))

        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
            utils.save_checkpoint({'epoch': epoch + 1, 'state_dict_image': model.state_dict(), 'best_acc': best_acc,
                                   'optimizer': optimizer.state_dict()}, directory=path_cp)
        else:
            if args.early_stop == early_stop_counter:
                break
            early_stop_counter += 1

        # Logger step
        logger.add_scalar('learning_rate', args.lr)
        logger.add_scalar('accuracy', acc)
        logger.step()

        lr -= lr_step

    # load the best model yet
    best_model_file = os.path.join(path_cp, 'model_best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict_image'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded best model '{0}' (epoch {1}; accuracy {2:.4f})".format(best_model_file, epoch, best_acc))
        # evaluate on test set
        with torch.no_grad():
            acc = validate(test_loader, model, criterion, epoch, logger)
        print('Accuracy on test set: {0:.4f}'.format(acc))
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()


def train(train_loader, model, criterion, optimizer, epoch, logger=None):

    # Switch to train mode
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # Start counting time
    end = time.time()

    for i, (im, cl) in enumerate(train_loader):
        if torch.cuda.is_available():
            im, cl = im.cuda(), cl.cuda()
        op = model(im)
        loss = criterion(op, cl)
        acc = utils.accuracy(op.data, cl.data, topk=(1,))
        losses.update(loss.item(), cl.size(0))
        accuracies.update(acc[0].item(), cl.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.log_interval == 0:
            print('[Train] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                  .format(epoch + 1, i + 1, len(train_loader), batch_time=batch_time, loss=losses, acc=accuracies))


def validate(valid_loader, model, criterion, epoch, logger=None):

    # Switch to train mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # Start counting time
    end = time.time()

    for i, (im, cl) in enumerate(valid_loader):

        if torch.cuda.is_available():
            im, cl = im.cuda(), cl.cuda()

        # compute output
        op = model(im)
        loss = criterion(op, cl)
        acc = utils.accuracy(op.data, cl.data, topk=(1,))
        losses.update(loss.item(), cl.size(0))
        accuracies.update(acc[0].item(), cl.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.log_interval == 0:
            print('[Validation] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                  .format(epoch + 1, i + 1, len(valid_loader), batch_time=batch_time, loss=losses, acc=accuracies))

    return accuracies.avg


if __name__ == '__main__':
    main()
