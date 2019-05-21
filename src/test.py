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
from models import SEM_PCYC
from data import DataGeneratorSketch, DataGeneratorImage

np.random.seed(0)


def main():

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    args.test = True
    if args.filter_sketch:
        assert args.dataset == 'Sketchy'
    if args.split_eccv_2018:
        assert args.dataset == 'Sketchy_extended' or args.dataset == 'Sketchy'

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
    path_results = os.path.join(path_aux, 'Results', args.dataset, str_aux, model_name, str(args.dim_out))
    files_semantic_labels = []
    sem_dim = 0
    for f in args.semantic_models:
        fi = os.path.join(path_aux, 'Semantic', args.dataset, f + '.npy')
        files_semantic_labels.append(fi)
        sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]

    print('Checkpoint path: {}'.format(path_cp))
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

    if args.gzs_sbir > 0:
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

    data_test_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, splits['te_fls_sk'],
                                           splits['te_clss_sk'], transforms=transform_sketch)
    data_test_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['te_fls_im'],
                                         splits['te_clss_im'], transforms=transform_image)
    print('Done')

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


def validate(valid_loader_sketch, valid_loader_image, sem_pcyc_model, epoch, args):

    # Switch to test mode
    sem_pcyc_model.eval()

    batch_time = AverageMeter()

    # Start counting time
    time_start = time.time()

    for i, (sk, cls_sk) in enumerate(valid_loader_sketch):

        if torch.cuda.is_available():
            sk = sk.cuda()

        # Sketch embedding into a semantic space
        sk_em = sem_pcyc_model.get_sketch_embeddings(sk)

        # Accumulate sketch embedding
        if i == 0:
            acc_sk_em = sk_em.cpu().data.numpy()
            acc_cls_sk = cls_sk
        else:
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

    for i, (im, cls_im) in enumerate(valid_loader_image):

        if torch.cuda.is_available():
            im = im.cuda()

        # Image embedding into a semantic space
        im_em = sem_pcyc_model.get_image_embeddings(im)

        # Accumulate sketch embedding
        if i == 0:
            acc_im_em = im_em.cpu().data.numpy()
            acc_cls_im = cls_im
        else:
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
    sim_bin = np.exp(-cdist(acc_sk_em_bin, acc_im_em_bin, metric='hamming'))
    time_bin = (time.time() - t) / acc_cls_sk.shape[0]

    # similarity of classes or ground truths
    # Multiplied by 1 for boolean to integer conversion
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1

    apsall = utils.apsak(sim_euc, str_sim)
    aps200 = utils.apsak(sim_euc, str_sim, k=200)
    prec100, _ = utils.precak(sim_euc, str_sim, k=100)
    prec200, _ = utils.precak(sim_euc, str_sim, k=200)

    apsall_bin = utils.apsak(sim_bin, str_sim)
    aps200_bin = utils.apsak(sim_bin, str_sim, k=200)
    prec100_bin, _ = utils.precak(sim_bin, str_sim, k=100)
    prec200_bin, _ = utils.precak(sim_bin, str_sim, k=200)

    valid_data = {'aps@all': apsall, 'aps@200': aps200, 'prec@100': prec100, 'prec@200': prec200, 'sim_euc': sim_euc,
                  'time_euc': time_euc, 'aps@all_bin': apsall_bin, 'aps@200_bin': aps200_bin, 'prec@100_bin':
                      prec100_bin, 'prec@200_bin': prec200_bin, 'sim_bin': sim_bin, 'time_bin': time_bin, 'str_sim':
                      str_sim}

    print('Done')

    return valid_data


if __name__ == '__main__':
    main()
