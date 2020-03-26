# -*- coding:utf-8 -*-
import os,sys
os.environ['GLOG_minloglevel'] = '2'  # Hide caffe debug info.

import cv2 as cv
import numpy as np

try:
    import caffe
except:
    caffe = None

class MobileID(object):
    def __init__(self, model, weight,do_mirror=False, featLayer='ip3', gpu_id = 0):

        self.featLayer = featLayer
        self.model = model
        self.weight = weight
        self.do_mirror = do_mirror
        self.net = caffe.Net(model, weight, caffe.TEST)

    @staticmethod
    def norm_image(_im):
        
        im = cv.cvtColor(_im, cv.COLOR_BGR2RGB)
        
        img32 = cv.resize(im, (47,55)).astype('float32')
        
        img = (img32 - 127.5) / 128
        img = np.transpose(img32, [2, 0, 1])
        return img

    def extract_image(self, im):
        # im = caffe.io.load_image(img_path)
        # self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
        img = self.norm_image(im)
        #print(img)
        self.net.blobs['data'].data[...] = img
        self.net.forward()
        feat1 = self.net.blobs[self.featLayer].data[0].flatten()
        return feat1
    
    def extract_feature(self, im):
        feat1 = self.extract_image(im)
        if self.do_mirror == False:
            return feat1
            
        flip = cv.flip(im, 1)
        feat2 = self.extract_image(flip)
        
        return np.concatenate([feat1, feat2])
        
        
    def extract(self, img_path):
        im = cv.imread(img_path)
        return self.extract_feature(im)
    
    