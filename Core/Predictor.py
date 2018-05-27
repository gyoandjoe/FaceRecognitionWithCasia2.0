import theano
import theano.tensor as T
import numpy as np


class Predictor(object):

    def __init__(self,images,cnn):
        self.images=images
        self.CNN = cnn
        noRowsInBatch = T.lscalar()

        self.predictor_model = theano.function(
            inputs=[noRowsInBatch],
            outputs=self.CNN.PredictorFunction,
            givens={
                self.CNN.image_input: self.images,
                self.CNN.batch_size:noRowsInBatch

            }
            # on_unused_input='warn'
        )

    def Predict(self):

        result = self.predictor_model(1)
        for r,img in ((result,self.images)):
            print("Resultado prediccion: " + str(r))
        return
