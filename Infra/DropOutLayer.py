import theano.tensor as T
import theano
import numpy as np

class DropOutLayer(object):
    def __init__(self, input, srng, image_shape, is_training,p):
        self.srng=srng
        self.image_shape=image_shape
        self.p=p
        self.is_training=is_training
        self.input = input

        self.mask = self.srng.binomial(n=1, size=self.image_shape, p=self.p, dtype=theano.config.floatX)
        self.output = T.switch(T.neq(self.is_training, 0), np.multiply(self.input, self.mask),np.multiply(self.input, self.p))  # np.multiply(input,mask) => entrenando


        # prob bajo = muchos ceros
        # prob alto = muchos unos

        #np.multiply(input,mask) => Training
        #np.multiply(input, p) => not training
        return
    def UpdateMask(self):
        self.mask = self.srng.binomial(n=1, size=self.image_shape, p=self.p, dtype=theano.config.floatX)
        return self.mask


