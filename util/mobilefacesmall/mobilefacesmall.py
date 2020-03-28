# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import cv2
from cv2 import dnn

from sklearn.preprocessing import normalize
import numpy as np


class MobileFaceNetSmall(object):
    def __init__(self, model, weight=None, do_mirror=False, feat_layer='fc1'):
        self.model = model
        self.weight = weight
        self.do_mirror = do_mirror
        self.feature_layer = feat_layer
        
        self.net = dnn.readNetFromONNX(model)
        
        
    def norm_image(self, image):
        
        if image.shape[0] != 112 or image.shape[1] != 112:
            image = cv2.resize(image, (112,112))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
#         image = (imgage - 127.5) / 128
        return image

 
    def extract_image(self, image):

        image = self.norm_image(image)
        
        inputBlob = cv2.dnn.blobFromImage(image)
        self.net.setInput(inputBlob, "data")
        embedding = self.net.forward(self.feature_layer)
        embedding = normalize(embedding).flatten()
        return embedding
    
    def extract_feature(self, im):
        feat1 = self.extract_image(im)
        if self.do_mirror == False:
            return feat1
            
        flip = cv.flip(im, 1)
        feat2 = self.extract_image(flip)
        
        return np.concatenate([feat1, feat2])
    
    def extract(self, img_path):
        im = cv2.imread(img_path)
        return self.extract_feature(im)
    

def unitest():
    image_path = "./normed_croped.jpg"
    extractor = MobileFaceNetSmall("../../models/model-y1-test2/model-y1-test2.onnx")
    embedding = extractor.extract(image_path)
    print(embedding[:10])
    
if __name__ == '__main__':
    unitest()
