#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import argparse
import numpy as np
import re
import os
import glob
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

# user defined
import utils


def create_hier_emb(root_path, sim='path'):

    clss = glob.glob(os.path.join(root_path, '*'))
    clss = [c.split('/')[-1] for c in clss if os.path.isdir(c)]

    class_nodes, graph_nodes = create_node_set(clss)

    print('#class nodes: {0}, #graph nodes: {1}'.format(len(class_nodes), len(graph_nodes)))

    print('Creating hierarchical embedding...', end='')

    hier_emb = dict()

    brown_ic = wordnet_ic.ic('ic-brown.dat')
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')

    for i, cn in enumerate(class_nodes):
        if sim == 'path':
            sims = [wn.path_similarity(cn, gn) for gn in graph_nodes]
        elif sim == 'lch':
            sims = [wn.lch_similarity(cn, gn) for gn in graph_nodes]
        elif sim == 'wup':
            sims = [wn.wup_similarity(cn, gn) for gn in graph_nodes]
        elif sim == 'jcn':
            sims = [min(wn.jcn_similarity(cn, gn, brown_ic), 1) for gn in graph_nodes]
        elif sim == 'lin':
            sims = [wn.lin_similarity(cn, gn, semcor_ic) for gn in graph_nodes]
        else:
            sims = np.zeros(1, len(graph_nodes))
            NotImplemented
        hier_emb.update({clss[i]: np.array(sims)})

    print('Done')

    return hier_emb, len(graph_nodes)


def create_node_set(clss):

    class_nodes = list()
    graph_nodes = set()
    syn_dict = utils.synsets_wordnet()

    def recurse(s):

        """ Recursively move up semantic hierarchy and add nodes"""

        if s not in graph_nodes:
            graph_nodes.add(s)
            hypernyms = s.hypernyms()
            for s1 in hypernyms:
                recurse(s1)

    for cls in clss:
        cls_tmp = re.sub('[)(]', '', cls)
        if cls_tmp in syn_dict.keys():
            s = wn.synset(syn_dict[cls_tmp])                                # create synset
        else:
            s = wn.synsets(cls_tmp, pos='n')[0]
        class_nodes.append(s)
        recurse(s)

    return class_nodes, graph_nodes


def main():

    # Parse options for processing
    parser = argparse.ArgumentParser(description='Generate hierarchical embedding.')

    parser.add_argument('--dataset', default='Sketchy', help='Name of the dataset')
    parser.add_argument('--similarity', default='jcn', help='Type of similarity')
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

    hier_emb, dim = create_hier_emb(root_path, sim=args.similarity)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'hieremb-' + args.similarity + '.npy'), hier_emb)


if __name__ == '__main__':

    main()

    # For reading the classes of the awa dataset
    # fp = open('/home/adutta/Downloads/Animals_with_Attributes/classes.txt', 'r')
    # lines = fp.readlines()
    # fp.close()
    # clss = []
    # for line in lines:
    #     cls = line.split('\t')[1].replace('\n', '').replace('+', '_')
    #     clss.append(cls)
