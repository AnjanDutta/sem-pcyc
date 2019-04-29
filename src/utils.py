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


def get_classwise_samples(classes, fls_im, fls_sk, set_type='train', filter_sketch=True):

    idx_im = []
    idx_sk = []
    clss_im = [f.split('/')[-2] for f in fls_im]
    clss_sk = [f.split('/')[-2] for f in fls_sk]
    for c in classes:
        idx1 = [i for i, l in enumerate(clss_im) if l == c]
        if set_type == 'train':
            idx2 = [i for i, l in enumerate(clss_sk) if l == c]
            idx_cp = list(itertools.product(idx1, idx2))
            if len(idx_cp) > int(7e4):
                idx_cp = random.sample(idx_cp, int(7e4))
            idx1, idx2 = zip(*idx_cp)
        else:
            idx2 = [i for i, l in enumerate(clss_sk) if l == c]
            # remove duplicate sketches
            if filter_sketch:
                fls_sk_tmp = [fls_sk[i] for i in idx2]
                fls_sk_tmp = [f.split('-')[0] for f in fls_sk_tmp]
                idx_tmp = np.unique(fls_sk_tmp, return_index=True)[1]
                idx2 = [idx2[i] for i in idx_tmp]
        idx_im += idx1
        idx_sk += idx2

    return idx_im, idx_sk


def get_finegrained_samples(classes, fls_im, fls_sk, set_type='train', filter_sketch=True):

    idx_im = []
    idx_sk = []
    clss_im = [f.split('/')[-2] for f in fls_im]
    clss_sk = [f.split('/')[-2] for f in fls_sk]
    core_fls_im = [f.split('.')[-2] for f in fls_im]
    core_fls_sk = [f.split('-')[-2] for f in fls_sk]
    if set_type == 'train':
        for i, cfi in enumerate(core_fls_im):
            if cfi.split('/')[0] in classes:
                idx2 = [j for j, cfs in enumerate(core_fls_sk) if cfs == cfi]
                idx_cp = list(itertools.product([i], idx2))
                idx1, idx2 = zip(*idx_cp)
                idx_im += idx1
                idx_sk += idx2
    else:
        for c in classes:
            idx1 = [i for i, l in enumerate(clss_im) if l == c]
            idx2 = [i for i, l in enumerate(clss_sk) if l == c]
            if filter_sketch:
                fls_sk_tmp = [fls_sk[i] for i in idx2]
                fls_sk_tmp = [f.split('-')[0] for f in fls_sk_tmp]
                idx_tmp = np.unique(fls_sk_tmp, return_index=True)[1]
                idx2 = [idx2[i] for i in idx_tmp]
            idx_im += idx1
            idx_sk += idx2

    return idx_im, idx_sk


def load_files_sketchy_zeroshot(root_path, split_eccv_2018=False, photo_dir='photo', sketch_dir='sketch', photo_sd='',
                                sketch_sd='tx_000000000000'):

    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # image files and classes
    fls_im = glob.glob(os.path.join(path_im, '*', '*.jpg'))
    fls_im = [os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im]
    clss_im = [f.split('/')[-2] for f in fls_im]

    # all the unique classes
    classes = np.unique(clss_im)

    # divide the classes
    if split_eccv_2018:
        cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(cur_path, "test_classes_eccv_2018.txt")) as fp:
            te_classes = fp.read().splitlines()
            va_classes = te_classes
            tr_classes = [x for x in classes if x not in te_classes and x not in va_classes]
    else:
        tr_classes = np.random.choice(classes, int(0.8 * len(classes)), replace=False).tolist()
        va_classes = np.random.choice([x for x in classes if x not in tr_classes], int(0.1 * len(classes)),
                                      replace=False).tolist()
        te_classes = [x for x in classes if x not in tr_classes and x not in va_classes]

    # sketch files and classes
    fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
    fls_sk = [os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk]
    clss_sk = [f.split('/')[-2] for f in fls_sk]

    tr_idx_im, tr_idx_sk = get_classwise_samples(tr_classes, fls_im, fls_sk, set_type='train')
    va_idx_im, va_idx_sk = get_classwise_samples(va_classes, fls_im, fls_sk, set_type='valid')
    te_idx_im, te_idx_sk = get_classwise_samples(te_classes, fls_im, fls_sk, set_type='test')

    tr_fls_sk = [fls_sk[i] for i in tr_idx_sk]
    va_fls_sk = [fls_sk[i] for i in va_idx_sk]
    te_fls_sk = [fls_sk[i] for i in te_idx_sk]

    tr_clss_sk = [clss_sk[i] for i in tr_idx_sk]
    va_clss_sk = [clss_sk[i] for i in va_idx_sk]
    te_clss_sk = [clss_sk[i] for i in te_idx_sk]

    tr_fls_im = [fls_im[i] for i in tr_idx_im]
    va_fls_im = [fls_im[i] for i in va_idx_im]
    te_fls_im = [fls_im[i] for i in te_idx_im]

    tr_clss_im = [clss_im[i] for i in tr_idx_im]
    va_clss_im = [clss_im[i] for i in va_idx_im]
    te_clss_im = [clss_im[i] for i in te_idx_im]

    return tr_fls_sk, tr_clss_sk, tr_fls_im, tr_clss_im,\
        va_fls_sk, va_clss_sk, va_fls_im, va_clss_im,\
        te_fls_sk, te_clss_sk, te_fls_im, te_clss_im


def load_files_sketchy_finegrained_zeroshot(root_path, split_eccv_2018=False, photo_dir='photo', sketch_dir='sketch',
                                            photo_sd='', sketch_sd='tx_000000000000'):

    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # image files and classes
    fls_im = glob.glob(os.path.join(path_im, '*', '*.jpg'))
    fls_im = [os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im]
    clss_im = [f.split('/')[-2] for f in fls_im]

    # all the unique classes
    classes = np.unique(clss_im)

    # divide the classes
    if split_eccv_2018:
        cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(cur_path, "test_classes_eccv_2018.txt")) as fp:
            te_classes = fp.read().splitlines()
            va_classes = te_classes
            tr_classes = [x for x in classes if x not in te_classes and x not in va_classes]
    else:
        tr_classes = np.random.choice(classes, int(0.8 * len(classes)), replace=False).tolist()
        va_classes = np.random.choice([x for x in classes if x not in tr_classes], int(0.1 * len(classes)),
                                      replace=False).tolist()
        te_classes = [x for x in classes if x not in tr_classes and x not in va_classes]

    # sketch files and classes
    fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
    fls_sk = [os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk]
    clss_sk = [f.split('/')[-2] for f in fls_sk]

    tr_idx_im, tr_idx_sk = get_finegrained_samples(tr_classes, fls_im, fls_sk, set_type='train')
    va_idx_im, va_idx_sk = get_finegrained_samples(va_classes, fls_im, fls_sk, set_type='valid')
    te_idx_im, te_idx_sk = get_finegrained_samples(te_classes, fls_im, fls_sk, set_type='test')

    tr_fls_sk = [fls_sk[i] for i in tr_idx_sk]
    va_fls_sk = [fls_sk[i] for i in va_idx_sk]
    te_fls_sk = [fls_sk[i] for i in te_idx_sk]

    tr_cfs_sk = [fls_sk[i].split('-')[0] for i in tr_idx_sk]
    va_cfs_sk = [fls_sk[i].split('-')[0] for i in va_idx_sk]
    te_cfs_sk = [fls_sk[i].split('-')[0] for i in te_idx_sk]

    tr_clss_sk = [clss_sk[i] for i in tr_idx_sk]
    va_clss_sk = [clss_sk[i] for i in va_idx_sk]
    te_clss_sk = [clss_sk[i] for i in te_idx_sk]

    tr_fls_im = [fls_im[i] for i in tr_idx_im]
    va_fls_im = [fls_im[i] for i in va_idx_im]
    te_fls_im = [fls_im[i] for i in te_idx_im]

    tr_cfs_im = [fls_im[i].split('.')[0] for i in tr_idx_im]
    va_cfs_im = [fls_im[i].split('.')[0] for i in va_idx_im]
    te_cfs_im = [fls_im[i].split('.')[0] for i in te_idx_im]

    tr_clss_im = [clss_im[i] for i in tr_idx_im]
    va_clss_im = [clss_im[i] for i in va_idx_im]
    te_clss_im = [clss_im[i] for i in te_idx_im]

    return tr_fls_sk, tr_cfs_sk, tr_clss_sk, tr_fls_im, tr_cfs_im, tr_clss_im,\
        va_fls_sk, va_cfs_sk, va_clss_sk, va_fls_im, va_cfs_im, va_clss_im,\
        te_fls_sk, te_cfs_sk, te_clss_sk, te_fls_im, te_cfs_im, te_clss_im


def load_files_tuberlin_zeroshot(root_path, photo_dir='images', sketch_dir='sketches', photo_sd='', sketch_sd=''):

    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # image files and classes
    fls_im = glob.glob(os.path.join(path_im, '*', '*.jpg'))
    fls_im = [os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im]
    clss_im = [f.split('/')[-2] for f in fls_im]

    # all the unique classes
    classes = np.unique(clss_im)

    # divide the classes, done according to the "Zero-Shot Sketch-Image Hashing" paper
    tr_classes = np.random.choice(classes, int(0.88 * len(classes)), replace=False).tolist()
    va_classes = np.random.choice([x for x in classes if x not in tr_classes], int(0.06 * len(classes)),
                                  replace=False).tolist()
    te_classes = [x for x in classes if x not in tr_classes and x not in va_classes]

    # sketch files and classes
    fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
    fls_sk = [os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk]
    clss_sk = [f.split('/')[-2] for f in fls_sk]

    tr_idx_im, tr_idx_sk = get_classwise_samples(tr_classes, fls_im, fls_sk, set_type='train')
    va_idx_im, va_idx_sk = get_classwise_samples(va_classes, fls_im, fls_sk, set_type='valid')
    te_idx_im, te_idx_sk = get_classwise_samples(te_classes, fls_im, fls_sk, set_type='test')

    tr_fls_sk = [fls_sk[i] for i in tr_idx_sk]
    va_fls_sk = [fls_sk[i] for i in va_idx_sk]
    te_fls_sk = [fls_sk[i] for i in te_idx_sk]

    tr_clss_sk = [clss_sk[i] for i in tr_idx_sk]
    va_clss_sk = [clss_sk[i] for i in va_idx_sk]
    te_clss_sk = [clss_sk[i] for i in te_idx_sk]

    tr_fls_im = [fls_im[i] for i in tr_idx_im]
    va_fls_im = [fls_im[i] for i in va_idx_im]
    te_fls_im = [fls_im[i] for i in te_idx_im]

    tr_clss_im = [clss_im[i] for i in tr_idx_im]
    va_clss_im = [clss_im[i] for i in va_idx_im]
    te_clss_im = [clss_im[i] for i in te_idx_im]

    return tr_fls_sk, tr_clss_sk, tr_fls_im, tr_clss_im,\
        va_fls_sk, va_clss_sk, va_fls_im, va_clss_im,\
        te_fls_sk, te_clss_sk, te_fls_im, te_clss_im


def load_files_quickdraw_zeroshot(root_path, photo_dir='images', sketch_dir='sketches', photo_sd='', sketch_sd=''):
    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # image files and classes
    fls_im = glob.glob(os.path.join(path_im, '*', '*.jpg'))
    fls_im = [os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im]
    clss_im = [f.split('/')[-2] for f in fls_im]

    # all the unique classes
    classes = np.unique(clss_im)

    # divide the classes, done according to the "Zero-Shot Sketch-Image Hashing" paper
    tr_classes = np.random.choice(classes, int(0.73 * len(classes)), replace=False).tolist()
    va_classes = np.random.choice([x for x in classes if x not in tr_classes], int(0.14 * len(classes)),
                                  replace=False).tolist()
    te_classes = [x for x in classes if x not in tr_classes and x not in va_classes]

    # sketch files and classes
    fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
    fls_sk = [os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk]
    clss_sk = [f.split('/')[-2] for f in fls_sk]

    tr_idx_im, tr_idx_sk = get_classwise_samples(tr_classes, fls_im, fls_sk, set_type='train')
    va_idx_im, va_idx_sk = get_classwise_samples(va_classes, fls_im, fls_sk, set_type='valid')
    te_idx_im, te_idx_sk = get_classwise_samples(te_classes, fls_im, fls_sk, set_type='test')

    # Truncate the dataset for quicker retrieval
    va_idx_im = random.sample(va_idx_im, 15000)
    va_idx_sk = random.sample(va_idx_sk, 7500)
    te_idx_im = random.sample(te_idx_im, 15000)
    te_idx_sk = random.sample(te_idx_sk, 7500)

    tr_fls_sk = [fls_sk[i] for i in tr_idx_sk]
    va_fls_sk = [fls_sk[i] for i in va_idx_sk]
    te_fls_sk = [fls_sk[i] for i in te_idx_sk]

    tr_clss_sk = [clss_sk[i] for i in tr_idx_sk]
    va_clss_sk = [clss_sk[i] for i in va_idx_sk]
    te_clss_sk = [clss_sk[i] for i in te_idx_sk]

    tr_fls_im = [fls_im[i] for i in tr_idx_im]
    va_fls_im = [fls_im[i] for i in va_idx_im]
    te_fls_im = [fls_im[i] for i in te_idx_im]

    tr_clss_im = [clss_im[i] for i in tr_idx_im]
    va_clss_im = [clss_im[i] for i in va_idx_im]
    te_clss_im = [clss_im[i] for i in te_idx_im]

    return tr_fls_sk, tr_clss_sk, tr_fls_im, tr_clss_im, \
           va_fls_sk, va_clss_sk, va_fls_im, va_clss_im, \
           te_fls_sk, te_clss_sk, te_fls_im, te_clss_im


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

    idx = np.where(str_sim_k.sum(axis=1) == 0)
    sim_k = np.delete(sim_k, idx, axis=0)
    str_sim_k = np.delete(str_sim_k, idx, axis=0)

    return aps(sim_k, str_sim_k)
