# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import logging
import time
import os
import pickle
import argparse
import cv2

from verification import verification
from plot import draw_chart

import sys
sys.path.insert(0, '../facealign')
sys.path.insert(0, '../util') 

from models import model_factory

from distance import get_distance

def parse_line(line):
    splits = line.split()
    # skip line
    if len(splits) < 2:
        return None

    if len(splits) == 2:
        return int(splits[0]), int(splits[1])

    # name id1 id2
    if len(splits) == 3:
        return True, splits[0], splits[1], splits[0], splits[2]
    # name1 id1 name2 id2
    return False, splits[0], splits[1], splits[2], splits[3]

def load_image_list(pair_list):    
    img_list = []
    for pair in pair_list:
        # skip invalid pairs
        if not os.path.exists(pair[0]) or not os.path.exists(pair[1]):
            continue
        img1 = cv2.imread(pair[0])
        #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.imread(pair[1])
        #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        #print(img1.shape)
        img_list.append([img1, img2, pair[0], pair[1]])
    return img_list
    
def load_ytf_pairs(path, prefix):
    pos_list_ = []
    neg_list_ = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            flag, a, b = line.split(',')
            flag = int(flag)
            a = os.path.join(prefix, a)
            b = os.path.join(prefix, b)
            if flag == 1:
                pos_list_.append([a, b])
            else:
                neg_list_.append([a, b])
                
    pos_img = load_image_list(pos_list_)
    neg_img = load_image_list(neg_list_)
    return pos_img, neg_img
    
    
def load_image_pairs(pair_path, prefix):
    pair_list = []
    num_of_folds = -1 
    pairs_of_each_fold = -1
    # parse pairs 
    with open(pair_path, 'r') as f:
        for line in f.readlines():
            pair = parse_line(line)
            if len(pair) == 2:
                num_of_folds = pair[0]
                pairs_of_each_fold = pair[1]
                continue

            if pair is not None:
                pair_list.append(pair)
                # print(pair)

    # protect error pairs file
    assert num_of_folds > 0

    #print('#pairs:%d' % len(pair_list))
    # compute feature
    pos_img = []
    neg_img = []
    count = 0
    for pair in pair_list:
        count += 1
        
        img_path1 = '%s/%s/%s_%04d.jpg' % (prefix, pair[1], pair[1], int(pair[2]))
        img_path2 = '%s/%s/%s_%04d.jpg' % (prefix, pair[3], pair[3], int(pair[4]))
        rel_path1 = '%s/%s_%04d.jpg' % (pair[1], pair[1], int(pair[2]))
        rel_path2 = '%s/%s_%04d.jpg' % (pair[3], pair[3], int(pair[4]))
        # skip invalid pairs
        if not os.path.exists(img_path1) or not os.path.exists(img_path2):
            continue
        img1 = cv2.imread(img_path1)
        #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.imread(img_path2)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        #print(img1.shape)
        if pair[0]:
            pos_img.append([img1, img2, rel_path1, rel_path2])
        else:
            neg_img.append([img1, img2, rel_path1, rel_path2])
    return pos_img, neg_img, num_of_folds
        
        
def extract_feature(extractor, img_list):
    feat_list = []
    n = len(img_list)
    idx = 1
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        feat1 = extractor.extract_feature(img1)
        feat2 = extractor.extract_feature(img2)
        feat_list.append([feat1, feat2])
        if idx > 1:
            print('{}'.format('\b'*10))
        print('{}/{}'.format(idx, n), end='')
        idx += 1
    return feat_list

def crop_image_list(img_list, imsize):
    out_list = []
    h, w, c = img_list[0][0].shape
    x1 = (w - imsize[0])//2
    y1 = (h - imsize[1])//2
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        img1 = img1[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        img2 = img2[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        out_list.append([img1, img2])
    #print(img1.shape)
    return out_list
    
    
def load_bin(path, image_size):
    import mxnet as mx
    bins, issame_list = pickle.load(open(path, 'rb'))
    pos_img = []
    neg_img = []
    for i in xrange(len(issame_list)):
        _bin = bins[i*2]
        img1 = mx.image.imdecode(_bin)
        _bin = bins[i*2+1]
        img2 = mx.image.imdecode(_bin)
        if issame_list[i]:
            pos_img.append([img1, img2])
        else:
            neg_img.append([img2, img2])
            
    return pos_img, neg_img  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", help="lfw | ytf")
    parser.add_argument("--data",   help="lfw.np or pair.txt")
    parser.add_argument("--prefix", help="data prefix")
    parser.add_argument("--model_name", help= 'specify which model to test \n'
                                              ' centerface\n'
                                              ' sphereface\n'
                                              ' AMSoftmax\n'
                                              ' arcface\n'
                                              ' yours \n')

    parser.add_argument("--dist_type", default='cosine', help="distance measure ['cosine', 'L2', 'SSD']")
    parser.add_argument("--do_mirror", default=False, help="mirror image and concatinate features")

    args = parser.parse_args()
    output_dir = '.'
    
    # parse args   
    model_name = args.model_name
    test_set = args.test_set
    dist_type = args.dist_type
    do_mirror = args.do_mirror
    print('Dataset  \t: %s (%s,%s)' % (args.test_set, args.data, args.prefix))
    print('Testing  \t: %s' % model_name)
    print('Distance \t: %s' % dist_type)
    print('Do mirror\t: {}'.format(do_mirror))
    
    # model
    extractor, image_size = model_factory(model_name, '../models', do_mirror)
    print('Testing model\t: %s' % (extractor.weight))
    print('Image size\t: {}'.format(image_size))
    
    # load images
    if args.data.find('.np') > 0:
        pos_img, neg_img = pickle.load(open(args.data, 'rb'))
        #pos_img, neg_img = pickle.load(open(lfw_data, 'rb'), encoding='iso-8859-1')
    else:
        if args.test_set == 'lfw':
            pos_img, neg_img, num_of_fold = load_image_pairs(args.data, args.prefix)
        else:
            pos_img, neg_img = load_ytf_pairs(args.data, args.prefix)
        
    # crop image
    pos_img = crop_image_list(pos_img, image_size)
    neg_img = crop_image_list(neg_img, image_size)
    #print(type(pos_img[0][0]))
    #exit()  
    # compute feature
    print('Extracting features ...')
    pos_list = extract_feature(extractor, pos_img)
    print('  Done positive pairs')
    neg_list = extract_feature(extractor, neg_img)
    print('  Done negative pairs')

    # evaluate
    print('Evaluating ...')
    accuracy, std, threshold, pos, neg = verification(pos_list, neg_list, num_of_fold, dist_type = dist_type)    
    title = model_name
    #draw_chart(title, output_dir, {'pos': pos, 'neg': neg}, precision, threshold)
    print('------------------------------------------------------------')
    print('Accuracy on %s : %1.5f+-%1.5f \nBest threshold   : %f' % (args.test_set, accuracy, std, threshold))
   
   
