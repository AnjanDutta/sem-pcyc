#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import time
import numpy as np

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

# user defined
import utils
from models import SEM_PCYC
from logger import Logger, AverageMeter
from losses import GANLoss
from options import Options
from test import validate
from data import DataGeneratorPaired, DataGeneratorSketch, DataGeneratorImage

np.random.seed(0)


def main():

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    if args.fine_grained:
        assert args.dataset == 'Sketchy'

    # Read the config file and
    config = utils.read_config()
    path_dataset = config['path_dataset']
    path_aux = config['path_aux']

    # modify the log and check point paths
    if '_' in args.dataset:
        token = args.dataset.split('_')
        args.dataset = token[0]
        ds_var = token[1]
    else:
        ds_var = None
    args.semantic_models = sorted(args.semantic_models)
    model_name = '+'.join(args.semantic_models)
    root_path = os.path.join(path_dataset, args.dataset)
    str_aux = ''
    if args.split_eccv_2018:
        str_aux = 'split_eccv_2018'
    path_sketch_model = os.path.join(path_aux, 'CheckPoints', 'sketch', args.dataset, 'vanilla')
    path_image_model = os.path.join(path_aux, 'CheckPoints', 'image', args.dataset, 'vanilla')
    path_cp = os.path.join(path_aux, 'CheckPoints', args.dataset, str_aux, model_name, str(args.dim_out))
    path_log = os.path.join(path_aux, 'LogFiles', args.dataset, str_aux, model_name, str(args.dim_out))
    path_qualitative_results = os.path.join(path_aux, 'Results', args.dataset, str_aux, model_name, str(args.dim_out))
    files_semantic_labels = []
    sem_dim = 0
    for f in args.semantic_models:
        fi = os.path.join(path_aux, 'Semantic', args.dataset, f + '.npy')
        files_semantic_labels.append(fi)
        sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]

    print('Checkpoint path: {}'.format(path_cp))
    print('Logger path: {}'.format(path_log))
    print('Result path: {}'.format(path_qualitative_results))

    # Parameters for transforming the images
    transform_image = transforms.Compose([transforms.Resize((args.im_sz, args.im_sz)), transforms.ToTensor()])
    transform_sketch = transforms.Compose([transforms.Resize((args.sk_sz, args.sk_sz)), transforms.ToTensor()])

    # Load the dataset
    print('Loading data...', end='')

    if args.dataset == 'Sketchy':
        if ds_var == 'extended':
            photo_dir = 'extended_photo'  # photo or extended_photo
            photo_sd = ''
        else:
            photo_dir = 'photo'
            photo_sd = 'tx_000000000000'
        sketch_dir = 'sketch'
        sketch_sd = 'tx_000000000000'

        if args.fine_grained:
            tr_fls_sk, tr_cfs_sk, tr_clss_sk, tr_fls_im, tr_cfs_im, tr_clss_im, \
            va_fls_sk, va_clss_sk, _, va_fls_im, va_clss_im, _, \
            te_fls_sk, te_clss_sk, _, te_fls_im, te_clss_im, _ = \
                utils.load_files_sketchy_finegrained_zeroshot(root_path=root_path, split_eccv_2018=args.split_eccv_2018,
                                                              photo_dir=photo_dir, sketch_dir=sketch_dir,
                                                              photo_sd=photo_sd, sketch_sd=sketch_sd)
        else:
            tr_fls_sk, tr_clss_sk, tr_fls_im, tr_clss_im, \
            va_fls_sk, va_clss_sk, va_fls_im, va_clss_im, \
            te_fls_sk, te_clss_sk, te_fls_im, te_clss_im = utils.load_files_sketchy_zeroshot(root_path=root_path,
                                                                                             split_eccv_2018=
                                                                                             args.split_eccv_2018,
                                                                                             photo_dir=photo_dir,
                                                                                             sketch_dir=sketch_dir,
                                                                                             photo_sd=photo_sd,
                                                                                             sketch_sd=sketch_sd)
    elif args.dataset == 'TU-Berlin':
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        tr_fls_sk, tr_clss_sk, tr_fls_im, tr_clss_im, \
            va_fls_sk, va_clss_sk, va_fls_im, va_clss_im, \
            te_fls_sk, te_clss_sk, te_fls_im, te_clss_im = utils.load_files_tuberlin_zeroshot(root_path=root_path,
                                                                                              photo_dir=photo_dir,
                                                                                              sketch_dir=sketch_dir,
                                                                                              photo_sd=photo_sd,
                                                                                              sketch_sd=sketch_sd)
    elif args.dataset == 'QuickDraw':
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        tr_fls_sk, tr_clss_sk, tr_fls_im, tr_clss_im, \
        va_fls_sk, va_clss_sk, va_fls_im, va_clss_im, \
        te_fls_sk, te_clss_sk, te_fls_im, te_clss_im = utils.load_files_quickdraw_zeroshot(root_path=root_path,
                                                                                           photo_dir=photo_dir,
                                                                                           sketch_dir=sketch_dir,
                                                                                           photo_sd=photo_sd,
                                                                                           sketch_sd=sketch_sd)
    else:
        print('Wrong dataset.')
        exit()

    # Combine the valid and test set into test set
    te_fls_sk = va_fls_sk + te_fls_sk
    te_clss_sk = va_clss_sk + te_clss_sk
    te_fls_im = va_fls_im + te_fls_im
    te_clss_im = va_clss_im + te_clss_im

    if args.gzs_sbir:
        _, idx_sk = np.unique([f.split('-')[0] for f in tr_fls_sk], return_index=True)
        _, idx_im = np.unique(tr_fls_im, return_index=True)
        min_idx = min(idx_sk.shape[0], idx_im.shape[0])
        idx_sk = np.random.choice(idx_sk, min_idx, replace=False)
        idx_im = np.random.choice(idx_im, min_idx, replace=False)
        te_fls_sk = [tr_fls_sk[i] for i in idx_sk] + te_fls_sk
        te_clss_sk = [tr_clss_sk[i] for i in idx_sk] + te_clss_sk
        te_fls_im = [tr_fls_im[i] for i in idx_im] + te_fls_im
        te_clss_im = [tr_clss_im[i] for i in idx_im] + te_clss_im

    # class dictionary
    dict_clss = utils.create_dict_texts(tr_clss_im)

    data_train = DataGeneratorPaired(args.dataset, root_path, photo_dir, sketch_dir, photo_sd, sketch_sd, tr_fls_sk,
                                     tr_fls_im, tr_clss_im, transforms_sketch=transform_sketch,
                                     transforms_image=transform_image)
    data_valid_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, va_fls_sk, va_clss_sk,
                                            transforms=transform_sketch)
    data_valid_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, va_fls_im, va_clss_im,
                                          transforms=transform_image)
    data_test_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, te_fls_sk, te_clss_sk,
                                           transforms=transform_sketch)
    data_test_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, te_fls_im, te_clss_im,
                                         transforms=transform_image)
    print('Done')

    train_sampler = WeightedRandomSampler(data_train.get_weights(), num_samples=args.epoch_size * args.batch_size,
                                          replacement=True)

    # PyTorch train loader
    train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    # PyTorch valid loader for sketch
    valid_loader_sketch = DataLoader(dataset=data_valid_sketch, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
    # PyTorch valid loader for image
    valid_loader_image = DataLoader(dataset=data_valid_image, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    # PyTorch test loader for sketch
    test_loader_sketch = DataLoader(dataset=data_test_sketch, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
    # PyTorch test loader for image
    test_loader_image = DataLoader(dataset=data_test_image, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

    # Model parameters
    params_model = dict()
    # Paths to pre-trained sketch and image models
    params_model['path_sketch_model'] = path_sketch_model
    params_model['path_image_model'] = path_image_model
    # Dimensions
    params_model['dim_out'] = args.dim_out
    params_model['sem_dim'] = sem_dim
    # Number of classes
    params_model['num_clss'] = len(dict_clss)
    # Weight (on losses) parameters
    params_model['lambda_se'] = args.lambda_se
    params_model['lambda_im'] = args.lambda_im
    params_model['lambda_sk'] = args.lambda_sk
    params_model['lambda_gen_cyc'] = args.lambda_gen_cyc
    params_model['lambda_gen_adv'] = args.lambda_gen_adv
    params_model['lambda_gen_cls'] = args.lambda_gen_cls
    params_model['lambda_gen_reg'] = args.lambda_gen_reg
    params_model['lambda_disc_se'] = args.lambda_disc_se
    params_model['lambda_disc_sk'] = args.lambda_disc_sk
    params_model['lambda_disc_im'] = args.lambda_disc_im
    params_model['lambda_regular'] = args.lambda_regular
    # Optimizers' parameters
    params_model['lr'] = args.lr
    params_model['momentum'] = args.momentum
    params_model['milestones'] = args.milestones
    params_model['gamma'] = args.gamma
    # Files with semantic labels
    params_model['files_semantic_labels'] = files_semantic_labels
    # Class dictionary
    params_model['dict_clss'] = dict_clss

    # Model
    sem_pcyc_model = SEM_PCYC(params_model)

    cudnn.benchmark = True

    # Logger
    print('Setting logger...', end='')
    logger = Logger(path_log, force=True)
    print('Done')

    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if args.ngpu > 0 & torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        sem_pcyc_model = sem_pcyc_model.cuda()
    print('Done')

    best_map = 0
    early_stop_counter = 0

    # Epoch for loop
    if not args.test:
        print('***Train***')
        for epoch in range(args.epochs):

            sem_pcyc_model.scheduler_gen.step()
            sem_pcyc_model.scheduler_disc.step()
            sem_pcyc_model.scheduler_ae.step()

            # train on training set
            losses_aut_enc, losses_gen_adv, losses_gen_cyc, losses_gen_cls, losses_gen, losses_disc_se, losses_disc_sk,\
            losses_disc_im, losses_disc = train(train_loader, sem_pcyc_model, epoch, args)

            # evaluate on validation set, map_ since map is already there
            print('***Validation***')
            metric, _, _, _, _ = validate(valid_loader_sketch, valid_loader_image, sem_pcyc_model, epoch, args)
            map_ = np.mean(metric['aps@all'])

            print('mAP@all on validation set after {0} epochs: {1:.4f} (real), {2:.4f} (binary)'
                .format(epoch + 1, map_, np.mean(metric['aps@all_bin'])))

            del metric

            if map_ > best_map:
                best_map = map_
                early_stop_counter = 0
                utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': sem_pcyc_model.state_dict(),
                                       'best_map': best_map}, directory=path_cp)
            else:
                if args.early_stop == early_stop_counter:
                    break
                early_stop_counter += 1

            # Logger step
            logger.add_scalar('semantic autoencoder loss', losses_aut_enc.avg)
            logger.add_scalar('generator adversarial loss', losses_gen_adv.avg)
            logger.add_scalar('generator cycle consistency loss', losses_gen_cyc.avg)
            logger.add_scalar('generator classification loss', losses_gen_cls.avg)
            logger.add_scalar('generator loss', losses_gen.avg)
            logger.add_scalar('semantic discriminator loss', losses_disc_se.avg)
            logger.add_scalar('sketch discriminator loss', losses_disc_sk.avg)
            logger.add_scalar('image discriminator loss', losses_disc_im.avg)
            logger.add_scalar('discriminator loss', losses_disc.avg)
            logger.add_scalar('mean average precision', map_)
            logger.step()

    # load the best model yet
    best_model_file = os.path.join(path_cp, 'model_best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        sem_pcyc_model.load_state_dict(checkpoint['state_dict'])
        print("Loaded best model '{0}' (epoch {1}; mAP@all {2:.4f})".format(best_model_file, epoch, best_map))
        print('***Test***')
        metric, sim, str_sim, sk_ind, im_ind = validate(test_loader_sketch, test_loader_image, sem_pcyc_model, epoch,
                                                        args)
        print('Results on test set: Prec@100 = {0:.4f}, mAP@all = {1:.4f}, Prec@200 = {2:.4f}, mAP@200 = {3:.4f}, '
              'Time = {4:.6f} || Prec@100 (binary) = {5:.4f}, mAP@all (binary) = {6:.4f}, Prec@200 (binary) = {7:.4f}, '
              'mAP@200 (binary) = {8:.4f}, Time (binary) = {9:.6f} '
            .format(metric['prec@100'], np.mean(metric['aps@all']), metric['prec@200'], np.mean(metric['aps@200']),
                    metric['time'], metric['prec@100_bin'], np.mean(metric['aps@all_bin']), metric['prec@200_bin'],
                    np.mean(metric['aps@200_bin']), metric['time_bin']))
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()


def train(train_loader, sem_pcyc_model, epoch, args):

    # Switch to train mode
    sem_pcyc_model.train()

    batch_time = AverageMeter()
    losses_gen_adv = AverageMeter()
    losses_gen_cyc = AverageMeter()
    losses_gen_cls = AverageMeter()
    losses_gen = AverageMeter()
    losses_disc_se = AverageMeter()
    losses_disc_sk = AverageMeter()
    losses_disc_im = AverageMeter()
    losses_disc = AverageMeter()
    losses_aut_enc = AverageMeter()

    # Start counting time
    time_start = time.time()

    for i, (sk, im, cl) in enumerate(train_loader):

        # Transfer sk and im to cuda
        if torch.cuda.is_available():
            sk, im = sk.cuda(), im.cuda()

        # Optimize parameters
        loss_aut_enc, loss_gen_adv, loss_gen_cyc, loss_gen_cls, loss_gen, loss_disc_se, loss_disc_sk, loss_disc_im,\
        loss_disc = sem_pcyc_model.optimize_params(sk, im, cl)

        # Store losses for visualization
        losses_aut_enc.update(loss_aut_enc.item(), sk.size(0))
        losses_gen_adv.update(loss_gen_adv.item(), sk.size(0))
        losses_gen_cyc.update(loss_gen_cyc.item(), sk.size(0))
        losses_gen_cls.update(loss_gen_cls.item(), sk.size(0))
        losses_gen.update(loss_gen.item(), sk.size(0))
        losses_disc_se.update(loss_disc_se.item(), sk.size(0))
        losses_disc_sk.update(loss_disc_sk.item(), sk.size(0))
        losses_disc_im.update(loss_disc_im.item(), sk.size(0))
        losses_disc.update(loss_disc.item(), sk.size(0))

        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % args.log_interval == 0:
            print('[Train] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Gen. Loss {loss_gen.val:.4f} ({loss_gen.avg:.4f})\t'
                  'Disc. Loss {loss_disc.val:.4f} ({loss_disc.avg:.4f})\t'
                  .format(epoch + 1, i + 1, len(train_loader), batch_time=batch_time, loss_gen=losses_gen,
                          loss_disc=losses_disc))

    return losses_aut_enc, losses_gen_adv, losses_gen_cyc, losses_gen_cls, losses_gen, losses_disc_se, losses_disc_sk, \
           losses_disc_im, losses_disc


if __name__ == '__main__':
    main()