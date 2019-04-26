#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import re
import os
import glob
import gensim.downloader as api
import utils


def create_wordemb(root_path, type='word2vec-google-news-300'):

    clss = glob.glob(os.path.join(root_path, '*'))
    clss = [c.split('/')[-1] for c in clss if os.path.isdir(c)]
    available_type = ['word2vec-google-news-300', 'fasttext-wiki-news-subwords-300', 'glove-wiki-gigaword-300']
    if type not in available_type:
        print("Type specified does not exist.")
        exit()
    model = api.load(type).wv

    lv = len(model['airplane'])  # for knowing the length of vector

    wordemb = dict()
    # synonym dictionary
    syn_dict = utils.get_synonym()
    for cls in clss:
        cls_tmp = re.sub('[)(]', '', cls)
        if cls_tmp in model.vocab:
            ws = [cls_tmp]
        elif cls_tmp in syn_dict.keys():
            ws = syn_dict[cls_tmp].split('_')
        else:
            ws = cls_tmp.split('_')
        v = np.zeros((len(ws), lv))
        for i, w in enumerate(ws):
            if w in model.vocab:
                v[i, :] = model[w]
            else:
                print(w)
        wordemb.update({cls: np.mean(v, axis=0)})
    return wordemb


def main():

    # Parse options for processing
    parser = argparse.ArgumentParser(description='Generate word2vec representation.')

    parser.add_argument('--dataset', default='Sketchy', help='Name of the dataset')
    parser.add_argument('--type', default='word2vec-google-news-300', help='Type of word2vec model')
    parser.add_argument('--save-dir', default='/home/adutta/Workspace/SemCycGenZ2SIR/Semantic', help='')

    # Parse the arguments
    args = parser.parse_args()

    # Read the config file and
    config = utils.read_config()
    path_dataset = config['path_dataset']

    if args.dataset == 'Sketchy':
        sketch_dir = 'sketch'
        sketch_sd = 'tx_000000000000'
    elif args.dataset == 'TU-Berlin':
        sketch_dir = 'sketches'
        sketch_sd = ''
    elif args.dataset == 'QuickDraw':
        sketch_dir = 'sketches'
        sketch_sd = ''
    else:
        sketch_dir = ''
        sketch_sd = ''
    root_path = os.path.join(path_dataset, args.dataset, sketch_dir, sketch_sd)
    save_path = os.path.join(args.save_dir, args.dataset)

    wordemb = create_wordemb(root_path, args.type)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, '-'.join(args.type.split('-')[:-1]) + '.npy'), wordemb)


if __name__ == '__main__':

    main()
