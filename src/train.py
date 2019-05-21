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
from options import Options
from test import validate
from data import DataGeneratorPaired, DataGeneratorSketch, DataGeneratorImage

np.random.seed(0)


def main():

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    if args.filter_sketch:
        assert args.dataset == 'Sketchy'
    if args.split_eccv_2018:
        assert args.dataset == 'Sketchy_extended' or args.dataset == 'Sketchy'
    if args.gzs_sbir:
        args.test = True

    # Read the config file and
    config = utils.read_config()
    path_dataset = config['path_dataset']
    path_aux = config['path_aux']

    # modify the log and check point paths
    ds_var = None
    if '_' in args.dataset:
        token = args.dataset.split('_')
        args.dataset = token[0]
        ds_var = token[1]

    str_aux = ''
    if args.split_eccv_2018:
        str_aux = 'split_eccv_2018'
    if args.gzs_sbir:
        str_aux = os.path.join(str_aux, 'generalized')
    args.semantic_models = sorted(args.semantic_models)
    model_name = '+'.join(args.semantic_models)
    root_path = os.path.join(path_dataset, args.dataset)
    path_sketch_model = os.path.join(path_aux, 'CheckPoints', args.dataset, 'sketch')
    path_image_model = os.path.join(path_aux, 'CheckPoints', args.dataset, 'image')
    path_cp = os.path.join(path_aux, 'CheckPoints', args.dataset, str_aux, model_name, str(args.dim_out))
    path_log = os.path.join(path_aux, 'LogFiles', args.dataset, str_aux, model_name, str(args.dim_out))
    path_results = os.path.join(path_aux, 'Results', args.dataset, str_aux, model_name, str(args.dim_out))
    files_semantic_labels = []
    sem_dim = 0
    for f in args.semantic_models:
        fi = os.path.join(path_aux, 'Semantic', args.dataset, f + '.npy')
        files_semantic_labels.append(fi)
        sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]

    print('Checkpoint path: {}'.format(path_cp))
    print('Logger path: {}'.format(path_log))
    print('Result path: {}'.format(path_results))

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
        splits = utils.load_files_sketchy_zeroshot(root_path=root_path, split_eccv_2018=args.split_eccv_2018,
                                                   photo_dir=photo_dir, sketch_dir=sketch_dir, photo_sd=photo_sd,
                                                   sketch_sd=sketch_sd)
    elif args.dataset == 'TU-Berlin':
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        splits = utils.load_files_tuberlin_zeroshot(root_path=root_path, photo_dir=photo_dir, sketch_dir=sketch_dir,
                                                    photo_sd=photo_sd, sketch_sd=sketch_sd)
    else:
        raise Exception('Wrong dataset.')

    # Combine the valid and test set into test set
    splits['te_fls_sk'] = np.concatenate((splits['va_fls_sk'], splits['te_fls_sk']), axis=0)
    splits['te_clss_sk'] = np.concatenate((splits['va_clss_sk'], splits['te_clss_sk']), axis=0)
    splits['te_fls_im'] = np.concatenate((splits['va_fls_im'], splits['te_fls_im']), axis=0)
    splits['te_clss_im'] = np.concatenate((splits['va_clss_im'], splits['te_clss_im']), axis=0)

    if args.gzs_sbir:
        perc = 0.2
        _, idx_sk = np.unique(splits['tr_fls_sk'], return_index=True)
        tr_fls_sk_ = splits['tr_fls_sk'][idx_sk]
        tr_clss_sk_ = splits['tr_clss_sk'][idx_sk]
        _, idx_im = np.unique(splits['tr_fls_im'], return_index=True)
        tr_fls_im_ = splits['tr_fls_im'][idx_im]
        tr_clss_im_ = splits['tr_clss_im'][idx_im]
        if args.dataset == 'Sketchy' and args.filter_sketch:
            _, idx_sk = np.unique([f.split('-')[0] for f in tr_fls_sk_], return_index=True)
            tr_fls_sk_ = tr_fls_sk_[idx_sk]
            tr_clss_sk_ = tr_clss_sk_[idx_sk]
        idx_sk = np.sort(np.random.choice(tr_fls_sk_.shape[0], int(perc * splits['te_fls_sk'].shape[0]), replace=False))
        idx_im = np.sort(np.random.choice(tr_fls_im_.shape[0], int(perc * splits['te_fls_im'].shape[0]), replace=False))
        splits['te_fls_sk'] = np.concatenate((tr_fls_sk_[idx_sk], splits['te_fls_sk']), axis=0)
        splits['te_clss_sk'] = np.concatenate((tr_clss_sk_[idx_sk], splits['te_clss_sk']), axis=0)
        splits['te_fls_im'] = np.concatenate((tr_fls_im_[idx_im], splits['te_fls_im']), axis=0)
        splits['te_clss_im'] = np.concatenate((tr_clss_im_[idx_im], splits['te_clss_im']), axis=0)

    # class dictionary
    dict_clss = utils.create_dict_texts(splits['tr_clss_im'])

    data_train = DataGeneratorPaired(args.dataset, root_path, photo_dir, sketch_dir, photo_sd, sketch_sd,
                                     splits['tr_fls_sk'], splits['tr_fls_im'], splits['tr_clss_im'],
                                     transforms_sketch=transform_sketch, transforms_image=transform_image)
    data_valid_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, splits['va_fls_sk'],
                                            splits['va_clss_sk'], transforms=transform_sketch)
    data_valid_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['va_fls_im'],
                                          splits['va_clss_im'], transforms=transform_image)
    data_test_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, splits['te_fls_sk'],
                                           splits['te_clss_sk'], transforms=transform_sketch)
    data_test_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['te_fls_im'],
                                         splits['te_clss_im'], transforms=transform_image)
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
            losses = train(train_loader, sem_pcyc_model, epoch, args)

            # evaluate on validation set, map_ since map is already there
            print('***Validation***')
            valid_data = validate(valid_loader_sketch, valid_loader_image, sem_pcyc_model, epoch, args)
            map_ = np.mean(valid_data['aps@all'])

            print('mAP@all on validation set after {0} epochs: {1:.4f} (real), {2:.4f} (binary)'
                .format(epoch + 1, map_, np.mean(valid_data['aps@all_bin'])))

            del valid_data

            if map_ > best_map:
                best_map = map_
                early_stop_counter = 0
                utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': sem_pcyc_model.state_dict(), 'best_map':
                    best_map}, directory=path_cp)
            else:
                if args.early_stop == early_stop_counter:
                    break
                early_stop_counter += 1

            # Logger step
            logger.add_scalar('semantic autoencoder loss', losses['aut_enc'].avg)
            logger.add_scalar('generator adversarial loss', losses['gen_adv'].avg)
            logger.add_scalar('generator cycle consistency loss', losses['gen_cyc'].avg)
            logger.add_scalar('generator classification loss', losses['gen_cls'].avg)
            logger.add_scalar('generator regression loss', losses['gen_reg'].avg)
            logger.add_scalar('generator loss', losses['gen'].avg)
            logger.add_scalar('semantic discriminator loss', losses['disc_se'].avg)
            logger.add_scalar('sketch discriminator loss', losses['disc_sk'].avg)
            logger.add_scalar('image discriminator loss', losses['disc_im'].avg)
            logger.add_scalar('discriminator loss', losses['disc'].avg)
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
        valid_data = validate(test_loader_sketch, test_loader_image, sem_pcyc_model, epoch, args)
        print('Results on test set: mAP@all = {1:.4f}, Prec@100 = {0:.4f}, mAP@200 = {3:.4f}, Prec@200 = {2:.4f}, '
              'Time = {4:.6f} || mAP@all (binary) = {6:.4f}, Prec@100 (binary) = {5:.4f}, mAP@200 (binary) = {8:.4f}, '
              'Prec@200 (binary) = {7:.4f}, Time (binary) = {9:.6f} '
              .format(valid_data['prec@100'], np.mean(valid_data['aps@all']), valid_data['prec@200'],
                      np.mean(valid_data['aps@200']), valid_data['time_euc'], valid_data['prec@100_bin'],
                      np.mean(valid_data['aps@all_bin']), valid_data['prec@200_bin'], np.mean(valid_data['aps@200_bin'])
                      , valid_data['time_bin']))
        print('Saving qualitative results...', end='')
        path_qualitative_results = os.path.join(path_results, 'qualitative_results')
        utils.save_qualitative_results(root_path, sketch_dir, sketch_sd, photo_dir, photo_sd, splits['te_fls_sk'],
                                       splits['te_fls_im'], path_qualitative_results, valid_data['aps@all'],
                                       valid_data['sim_euc'], valid_data['str_sim'], save_image=args.save_image_results,
                                       nq=args.number_qualit_results, best=args.save_best_results)
        print('Done')
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
    losses_gen_reg = AverageMeter()
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
        loss = sem_pcyc_model.optimize_params(sk, im, cl)

        # Store losses for visualization
        losses_aut_enc.update(loss['aut_enc'].item(), sk.size(0))
        losses_gen_adv.update(loss['gen_adv'].item(), sk.size(0))
        losses_gen_cyc.update(loss['gen_cyc'].item(), sk.size(0))
        losses_gen_cls.update(loss['gen_cls'].item(), sk.size(0))
        losses_gen_reg.update(loss['gen_reg'].item(), sk.size(0))
        losses_gen.update(loss['gen'].item(), sk.size(0))
        losses_disc_se.update(loss['disc_se'].item(), sk.size(0))
        losses_disc_sk.update(loss['disc_sk'].item(), sk.size(0))
        losses_disc_im.update(loss['disc_im'].item(), sk.size(0))
        losses_disc.update(loss['disc'].item(), sk.size(0))

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

    losses = {'aut_enc': losses_aut_enc, 'gen_adv': losses_gen_adv, 'gen_cyc': losses_gen_cyc, 'gen_cls':
        losses_gen_cls, 'gen_reg': losses_gen_reg, 'gen': losses_gen, 'disc_se': losses_disc_se, 'disc_sk':
        losses_disc_sk, 'disc_im': losses_disc_im, 'disc': losses_disc}

    return losses


if __name__ == '__main__':
    main()