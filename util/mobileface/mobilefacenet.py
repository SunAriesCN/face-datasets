# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import numpy as np

class MobileFaceNet(object):
    def __init__(self, model, weight, do_mirror=False, featLayer='fc5'):
        self.featLayer = featLayer
        self.model = model
        self.weight = weight
        self.do_mirror = do_mirror
        self.net = cv2.dnn.readNetFromCaffe(model, weight)

    @staticmethod
    def norm_image(_im):
        
        img32 = np.float32(_im)
        img = (img32 - 127.5) / 128
        return img

    def extract_image(self, im):

        img = self.norm_image(im)
        
        inputBlob = cv2.dnn.blobFromImage(img)
        self.net.setInput(inputBlob, "data")
        feature = self.net.forward(self.featLayer)
        feat1 = feature.flatten()
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


