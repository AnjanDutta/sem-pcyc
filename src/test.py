#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import time
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist

# pytorch, torch vision
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# user defined
import itq
import utils
from options import Options
from logger import Logger, AverageMeter
from models import VGGNetFeats, Generator
from data import DataGeneratorSketch, DataGeneratorImage

np.random.seed(0)


def main():

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

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
    path_cp = os.path.join(path_aux, 'CheckPoints', args.dataset, str_aux, model_name, str(args.dim_out))
    path_log = os.path.join(path_aux, 'LogFiles', args.dataset, str_aux, model_name, str(args.dim_out))
    path_qualitative_results = os.path.join(path_aux, 'Results', args.dataset, str_aux, model_name, str(args.dim_out))

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

        tr_fls_sk, tr_clss_sk, tr_fls_im, tr_clss_im, \
            va_fls_sk, va_clss_sk, va_fls_im, va_clss_im, \
            te_fls_sk, te_clss_sk, te_fls_im, te_clss_im = load_files_sketchy_zeroshot(root_path=root_path,
                                                                                       split_eccv_2018=
                                                                                       args.split_eccv_2018,
                                                                                       photo_dir=photo_dir,
                                                                                       sketch_dir=sketch_dir,
                                                                                       photo_sd=photo_sd,
                                                                                       sketch_sd=sketch_sd,
                                                                                       repeat=1)
    elif args.dataset == 'TU-Berlin':
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        tr_fls_sk, tr_clss_sk, tr_fls_im, tr_clss_im, \
            va_fls_sk, va_clss_sk, va_fls_im, va_clss_im, \
            te_fls_sk, te_clss_sk, te_fls_im, te_clss_im = load_files_tuberlin_zeroshot(root_path=root_path,
                                                                                        photo_dir=photo_dir,
                                                                                        sketch_dir=sketch_dir,
                                                                                        photo_sd=photo_sd,
                                                                                        sketch_sd=sketch_sd,
                                                                                        repeat=1)
    elif args.dataset == 'QuickDraw':
        NotImplementedError
    else:
        print('Wrong dataset.')
        exit()

    if args.combine_valid_test:
        va_fls_sk = va_fls_sk + te_fls_sk
        va_clss_sk = va_clss_sk + te_clss_sk
        va_fls_im = va_fls_im + te_fls_im
        va_clss_im = va_clss_im + te_clss_im

    if args.gzs_sbir > 0:
        tot_len = len(tr_fls_im)
        idx = np.random.choice(np.arange(tot_len), int(args.gzs_sbir * tot_len), replace=False).tolist()
        va_fls_sk = [tr_fls_sk[id] for id in idx] + va_fls_sk + te_fls_sk
        va_clss_sk = [tr_clss_sk[id] for id in idx] + va_clss_sk + te_clss_sk
        va_fls_im = [tr_fls_im[id] for id in idx] + va_fls_im + te_fls_im
        va_clss_im = [tr_clss_im[id] for id in idx] + va_clss_im + te_clss_im

    data_valid_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, va_fls_sk, va_clss_sk,
                                            transforms=transform_sketch)
    data_valid_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, va_fls_im, va_clss_im,
                                          transforms=transform_image)
    print('Done')

    # PyTorch valid loader for query
    valid_loader_sketch = DataLoader(dataset=data_valid_sketch, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                     pin_memory=True)
    # PyTorch valid loader for query
    valid_loader_image = DataLoader(dataset=data_valid_image, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                    pin_memory=True)

    # Model
    sem_pcyc_model = SEM_PCYC(params_model)

    cudnn.benchmark = True

    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if args.ngpu > 0 & torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        sem_pcyc_model = sem_pcyc_model.cuda()
    print('Done')

    # load the best model yet
    best_model_file = os.path.join(path_cp, 'model_best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file, map_location=lambda storage, loc: storage)
        epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        sem_pcyc_model.load_state_dict(checkpoint['state_dict_gen_sk2se'])
        print("Loaded best model '{0}' (epoch {1}; mAP@all {2:.4f})".format(best_model_file, epoch, best_map))
        # evaluate on test set
        metric, sim, str_sim, sk_ind, im_ind = validate(valid_loader_sketch, valid_loader_image, models, gens, epoch,
                                                        args)
        print('Results on test set: Prec@100 = {0:.4f}, mAP@all = {1:.4f}'
              .format(metric['prec@100_bin'], np.mean(metric['aps@all_bin'])))
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()


def validate(valid_loader_sketch, valid_loader_image, sem_pcyc_model, epoch, args, logger=None):

    # Switch to test mode
    sem_pcyc_model.eval()

    batch_time = AverageMeter()

    # Start counting time
    time_start = time.time()

    for i, (ind, sk, cls_sk) in enumerate(valid_loader_sketch):

        if torch.cuda.is_available():
            sk = sk.cuda()

        # Sketch embedding into a semantic space
        sk_em = sem_pcyc_model.get_sketch_embeddings(sk)

        # Accumulate sketch embedding
        if i == 0:
            acc_sk_ind = ind
            acc_sk_em = sk_em.cpu().data.numpy()
            acc_cls_sk = cls_sk
        else:
            acc_sk_ind = np.concatenate((acc_sk_ind, ind), axis=0)
            acc_sk_em = np.concatenate((acc_sk_em, sk_em.cpu().data.numpy()), axis=0)
            acc_cls_sk = np.concatenate((acc_cls_sk, cls_sk), axis=0)

        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % args.log_interval == 0:
            print('[Test][Sketch] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  .format(epoch + 1, i + 1, len(valid_loader_sketch), batch_time=batch_time))

    for i, (ind, im, cls_im) in enumerate(valid_loader_image):

        if torch.cuda.is_available():
            im = im.cuda()

        # Image embedding into a semantic space
        im_em = sem_pcyc_model.get_image_embeddings(im)

        # Accumulate sketch embedding
        if i == 0:
            acc_im_ind = ind
            acc_im_em = im_em.cpu().data.numpy()
            acc_cls_im = cls_im
        else:
            acc_im_ind = np.concatenate((acc_im_ind, ind), axis=0)
            acc_im_em = np.concatenate((acc_im_em, im_em.cpu().data.numpy()), axis=0)
            acc_cls_im = np.concatenate((acc_cls_im, cls_im), axis=0)

        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % args.log_interval == 0:
            print('[Test][Image] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  .format(epoch + 1, i + 1, len(valid_loader_image), batch_time=batch_time))

    # Compute mAP
    print('Computing evaluation metrics...', end='')

    # Compute similarity
    t = time.time()
    sim_euc = np.exp(-cdist(acc_sk_em, acc_im_em, metric='euclidean'))
    time_euc = (time.time() - t) / acc_cls_sk.shape[0]

    # binary encoding with ITQ
    acc_sk_em_bin, acc_im_em_bin = itq.compressITQ(acc_sk_em, acc_im_em)
    t = time.time()
    sim_ham = np.exp(-cdist(acc_sk_em_bin, acc_im_em_bin, metric='hamming'))
    time_bin = (time.time() - t) / acc_cls_sk.shape[0]

    # similarity of classes or ground truths
    # Multiplied by 1 for boolean to integer conversion
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1

    apsall = utils.apsak(sim_euc, str_sim)
    aps200 = utils.apsak(sim_euc, str_sim, k=200)
    prec100, rec100 = utils.precak(sim_euc, str_sim, k=100)
    prec200, rec200 = utils.precak(sim_euc, str_sim, k=200)

    apsall_bin = utils.apsak(sim_ham, str_sim)
    aps200_bin = utils.apsak(sim_ham, str_sim, k=200)
    prec100_bin, rec100_bin = utils.precak(sim_ham, str_sim, k=100)
    prec200_bin, rec200_bin = utils.precak(sim_ham, str_sim, k=200)

    metric = dict()
    metric['aps@all'] = apsall
    metric['aps@200'] = aps200
    metric['prec@100'] = prec100
    metric['rec@100'] = rec100
    metric['prec@200'] = prec200
    metric['rec@200'] = rec200
    metric['time'] = time_euc
    metric['aps@all_bin'] = apsall_bin
    metric['aps@200_bin'] = aps200_bin
    metric['prec@100_bin'] = prec100_bin
    metric['rec@100_bin'] = rec100_bin
    metric['prec@200_bin'] = prec200_bin
    metric['rec@200_bin'] = rec200_bin
    metric['time_bin'] = time_bin

    print('Done')

    return metric, sim_ham, str_sim, acc_sk_ind, acc_im_ind


if __name__ == '__main__':
    main()
