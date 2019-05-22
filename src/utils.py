#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import socket
import random
import itertools
import numpy as np
import multiprocessing
import configparser as cp
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

import torch

np.random.seed(0)


def numeric_classes(tags_classes, dict_tags):
    num_classes = np.array([dict_tags.get(t) for t in tags_classes])
    return num_classes


def create_dict_texts(texts):
    texts = sorted(list(set(texts)))
    d = {l: i for i, l in enumerate(texts)}
    return d


def read_config():
    config = cp.ConfigParser()
    cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config.read(os.path.join(cur_path, 'config.ini'))
    host = socket.gethostname()
    return config[host]


def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x


def save_checkpoint(state, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)


def prec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if k is not None:
        pr = len(act_set & pred_set) / min(k, len(pred_set))
    else:
        pr = len(act_set & pred_set) / max(len(pred_set), 1)
    return pr


def rec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / max(len(act_set), 1)
    return re


def precak(sim, str_sim, k=None):
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    pred_lists = np.argsort(-sim, axis=1)
    num_cores = min(multiprocessing.cpu_count(), 32)
    nq = len(act_lists)
    preck = Parallel(n_jobs=num_cores)(delayed(prec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    reck = Parallel(n_jobs=num_cores)(delayed(rec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    return np.mean(preck), np.mean(reck)


def aps(sim, str_sim):
    nq = str_sim.shape[0]
    num_cores = min(multiprocessing.cpu_count(), 32)
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
    return aps


def apsak(sim, str_sim, k=None):
    idx = (-sim).argsort()[:, :k]
    sim_k = np.array([sim[i, id] for i, id in enumerate(idx)])
    str_sim_k = np.array([str_sim[i, id] for i, id in enumerate(idx)])
    idx_nz = np.where(str_sim_k.sum(axis=1) != 0)[0]
    sim_k = sim_k[idx_nz]
    str_sim_k = str_sim_k[idx_nz]
    aps_ = np.zeros((sim.shape[0]), dtype=np.float)
    aps_[idx_nz] = aps(sim_k, str_sim_k)
    return aps_


def get_coarse_grained_samples(classes, fls_im, fls_sk, set_type='train', filter_sketch=True):

    idx_im_ret = np.array([], dtype=np.int)
    idx_sk_ret = np.array([], dtype=np.int)
    clss_im = np.array([f.split('/')[-2] for f in fls_im])
    clss_sk = np.array([f.split('/')[-2] for f in fls_sk])
    names_sk = np.array([f.split('-')[0] for f in fls_sk])
    for i, c in enumerate(classes):
        idx1 = np.where(clss_im == c)[0]
        idx2 = np.where(clss_sk == c)[0]
        if set_type == 'train':
            idx_cp = list(itertools.product(idx1, idx2))
            if len(idx_cp) > 100000:
                random.seed(i)
                idx_cp = random.sample(idx_cp, 100000)
            idx1, idx2 = zip(*idx_cp)
        else:
            # remove duplicate sketches
            if filter_sketch:
                names_sk_tmp = names_sk[idx2]
                idx_tmp = np.unique(names_sk_tmp, return_index=True)[1]
                idx2 = idx2[idx_tmp]
        idx_im_ret = np.concatenate((idx_im_ret, idx1), axis=0)
        idx_sk_ret = np.concatenate((idx_sk_ret, idx2), axis=0)

    return idx_im_ret, idx_sk_ret


def load_files_sketchy_zeroshot(root_path, split_eccv_2018=False, filter_sketch=False, photo_dir='photo',
                                sketch_dir='sketch', photo_sd='tx_000000000000', sketch_sd='tx_000000000000'):
    # paths of sketch and image
    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # all the image and sketch files together with classes and core names
    fls_sk = np.array(['/'.join(f.split('/')[-2:]) for f in glob.glob(os.path.join(path_sk, '*/*.png'))])
    fls_im = np.array(['/'.join(f.split('/')[-2:]) for f in glob.glob(os.path.join(path_im, '*/*.jpg'))])

    # classes for image and sketch
    clss_sk = np.array([f.split('/')[0] for f in fls_sk])
    clss_im = np.array([f.split('/')[0] for f in fls_im])

    # all the unique classes
    classes = sorted(os.listdir(path_sk))

    # divide the classes
    if split_eccv_2018:
        # According to Yelamarthi et al., "A Zero-Shot Framework for Sketch Based Image Retrieval", ECCV 2018.
        cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(cur_path, "test_classes_eccv_2018.txt")) as fp:
            te_classes = fp.read().splitlines()
            va_classes = te_classes
            tr_classes = np.setdiff1d(classes, np.union1d(te_classes, va_classes))
    else:
        # According to Shen et al., "Zero-Shot Sketch-Image Hashing", CVPR 2018.
        np.random.seed(0)
        tr_classes = np.random.choice(classes, int(0.8 * len(classes)), replace=False)
        va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.1 * len(classes)), replace=False)
        te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))

    idx_tr_im, idx_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='train',
                                                      filter_sketch=filter_sketch)
    idx_va_im, idx_va_sk = get_coarse_grained_samples(va_classes, fls_im, fls_sk, set_type='valid',
                                                      filter_sketch=filter_sketch)
    idx_te_im, idx_te_sk = get_coarse_grained_samples(te_classes, fls_im, fls_sk, set_type='test',
                                                      filter_sketch=filter_sketch)

    splits = dict()

    splits['tr_fls_sk'] = fls_sk[idx_tr_sk]
    splits['va_fls_sk'] = fls_sk[idx_va_sk]
    splits['te_fls_sk'] = fls_sk[idx_te_sk]

    splits['tr_clss_sk'] = clss_sk[idx_tr_sk]
    splits['va_clss_sk'] = clss_sk[idx_va_sk]
    splits['te_clss_sk'] = clss_sk[idx_te_sk]

    splits['tr_fls_im'] = fls_im[idx_tr_im]
    splits['va_fls_im'] = fls_im[idx_va_im]
    splits['te_fls_im'] = fls_im[idx_te_im]

    splits['tr_clss_im'] = clss_im[idx_tr_im]
    splits['va_clss_im'] = clss_im[idx_va_im]
    splits['te_clss_im'] = clss_im[idx_te_im]

    return splits


def load_files_tuberlin_zeroshot(root_path, photo_dir='images', sketch_dir='sketches', photo_sd='', sketch_sd=''):

    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # image files and classes
    fls_im = glob.glob(os.path.join(path_im, '*', '*.jpg'))
    fls_im = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im])
    clss_im = np.array([f.split('/')[-2] for f in fls_im])

    # sketch files and classes
    fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
    fls_sk = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk])
    clss_sk = np.array([f.split('/')[-2] for f in fls_sk])

    # all the unique classes
    classes = np.unique(clss_im)

    # divide the classes, done according to the "Zero-Shot Sketch-Image Hashing" paper
    np.random.seed(0)
    tr_classes = np.random.choice(classes, int(0.88 * len(classes)), replace=False)
    va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.06 * len(classes)), replace=False)
    te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))

    idx_tr_im, idx_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='train')
    idx_va_im, idx_va_sk = get_coarse_grained_samples(va_classes, fls_im, fls_sk, set_type='valid')
    idx_te_im, idx_te_sk = get_coarse_grained_samples(te_classes, fls_im, fls_sk, set_type='test')

    splits = dict()

    splits['tr_fls_sk'] = fls_sk[idx_tr_sk]
    splits['va_fls_sk'] = fls_sk[idx_va_sk]
    splits['te_fls_sk'] = fls_sk[idx_te_sk]

    splits['tr_clss_sk'] = clss_sk[idx_tr_sk]
    splits['va_clss_sk'] = clss_sk[idx_va_sk]
    splits['te_clss_sk'] = clss_sk[idx_te_sk]

    splits['tr_fls_im'] = fls_im[idx_tr_im]
    splits['va_fls_im'] = fls_im[idx_va_im]
    splits['te_fls_im'] = fls_im[idx_te_im]

    splits['tr_clss_im'] = clss_im[idx_tr_im]
    splits['va_clss_im'] = clss_im[idx_va_im]
    splits['te_clss_im'] = clss_im[idx_te_im]

    return splits


def save_qualitative_results(root, sketch_dir, sketch_sd, photo_dir, photo_sd, fls_sk, fls_im, dir_op, aps, sim,
                             str_sim, nq=50, nim=20, im_sz=(256, 256), best=False, save_image=False):

    # Set directories according to dataset
    dir_sk = os.path.join(root, sketch_dir, sketch_sd)
    dir_im = os.path.join(root, photo_dir, photo_sd)

    if not os.path.isdir(dir_op):
        os.makedirs(dir_op)
    else:
        clean_folder(dir_op)

    if best:
        ind_sk = np.argsort(-aps)[:nq]
    else:
        np.random.seed(0)
        ind_sk = np.random.choice(len(aps), nq, replace=False)

    # create a text file for results
    fp = open(os.path.join(dir_op, "Results.txt"), "w")

    for i, isk in enumerate(ind_sk):
        fp.write("{0}, ".format(fls_sk[isk]))
        if save_image:
            sdir_op = os.path.join(dir_op, str(i + 1))
            if not os.path.isdir(sdir_op):
                os.makedirs(sdir_op)
            sk = Image.open(os.path.join(dir_sk, fls_sk[isk])).convert(mode='RGB').resize(im_sz)
            sk.save(os.path.join(sdir_op, fls_sk[isk].split('/')[0] + '.png'))
        ind_im = np.argsort(-sim[isk])[:nim]
        for j, iim in enumerate(ind_im):
            if j < len(ind_im)-1:
                fp.write("{0} {1}, ".format(fls_im[iim], str_sim[isk][iim]))
            else:
                fp.write("{0} {1}".format(fls_im[iim], str_sim[isk][iim]))
            if save_image:
                im = Image.open(os.path.join(dir_im, fls_im[iim])).convert(mode='RGB').resize(im_sz)
                im.save(os.path.join(sdir_op, str(j + 1) + '_' + str(str_sim[isk][iim]) + '.png'))
        fp.write("\n")
    fp.close()


def clean_folder(folder):

    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        try:
            if os.path.isfile(p):
                os.unlink(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except Exception as e:
            print(e)
