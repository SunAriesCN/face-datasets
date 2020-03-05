# -*- coding:utf-8 -*-
import os, sys

import cv2
import numpy as np
import tensorflow.lite as tflite

class TFLiteExtractor:
    def __init__(self, model, weight=None, do_mirror=False):
        # TFLite have only model and no weights,
        # weights is included in model
        self.model = model
        self.weight = weight
        self.do_mirror = do_mirror

        self.interpreter = tflite.Interpreter(model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']


    def norm_image(self,image):

        input_size = self.input_shape[1]

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        image_resized = cv2.resize(image_bgr, (input_size, input_size))
        image_float32 = np.float32(image_resized)
        image_reshaped = image_float32.reshape(self.input_shape)

        image_normed = (image_reshaped - 127.5) / 127.5

        return image_normed

    def extract_image(self, image):
        input_data = self.norm_image(image)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        feature = self.interpreter.get_tensor(self.output_details[0]['index']).flatten()

        return feature

    def extract_feature(self, image):
        feature = self.extract_image(image)
        if self.do_mirror == False:
            return feature
        
        flip = cv2.flip(image, 1)
        feature_flipped = self.extract_image(flip)

        return np.concatenate([feature, feature_flipped])

    def extract(self, image_path):
        image = cv2.imread(image_path)
        return self.extract_feature(image)


