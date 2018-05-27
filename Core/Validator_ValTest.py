import theano
import theano.tensor as T
import numpy as np



"""
Esta clase solo recibe modelos instanciados y usa sus metodos genericos para calcular promedio errores y costos
"""
class Validator_ValTest(object):
    def __init__(self,
                 data_set_manager,
                 logger,
                 cnn,
                 weightsRepo):
        self.data_set_manager = data_set_manager

        self.logger = logger

        self.CNN = cnn




        index = T.lscalar()
        noRowsInBatch = T.lscalar()

        self.data_set_manager.Dataset_repo.ShuffleListPKLFiles()
        self.data_set_manager.LoadRandomOrderDataSetXBatch(batch_index=0)


        self.evaluate_model_with_cost = theano.function(
            inputs=[index, noRowsInBatch],
            outputs=self.CNN.CostFunction,
            givens={
                self.CNN.image_input: self.data_set_manager.dataSetX[
                                      index * noRowsInBatch: (index + 1) * noRowsInBatch],
                self.CNN.y: self.data_set_manager.dataSetY[index * noRowsInBatch: (index + 1) * noRowsInBatch],
                self.CNN.batch_size: noRowsInBatch
            }
            # on_unused_input='warn'
        )

        self.evaluate_model_with_error = theano.function(
            inputs=[index, noRowsInBatch],
            outputs=self.CNN.ErrorFunction,
            givens={
                self.CNN.image_input: self.data_set_manager.dataSetX[
                                      index * noRowsInBatch: (index + 1) * noRowsInBatch],
                self.CNN.y: self.data_set_manager.dataSetY[index * noRowsInBatch: (index + 1) * noRowsInBatch],
                self.CNN.batch_size: noRowsInBatch
            }
            # on_unused_input='warn'
        )

        return

    def CalculateCost(self):
        sumaCost = 0.0
        noBatchsToEvaluate = 0

        #Ordenamos aleatoriamente la lista de pkls
        self.data_set_manager.Dataset_repo.ShuffleListPKLFiles()

        #Recorremos todos los batchs (internamente el data set manager administra la carga de los super batchs)
        for batch_index in range(self.data_set_manager.Dataset_repo.No_batchs_in_dataset):

            #self.TryUpdateLearningRate(epoch_index,batch_index)
            batch_index_in_superbatch,noRowsInBatch =  self.data_set_manager.LoadRandomOrderDataSetXBatch(batch_index)

            cost = self.evaluate_model_with_cost(batch_index_in_superbatch, noRowsInBatch)
            sumaCost = sumaCost + cost
            noBatchsToEvaluate = noBatchsToEvaluate +1
            print("costo: " + str(cost)  + " Batch: " + str(batch_index) +"/" + str(self.data_set_manager.Dataset_repo.No_batchs_in_dataset-1) )
        promedio = sumaCost / noBatchsToEvaluate
        return promedio


    def CalculateError(self, noBatchsToEvaluate=-1):
        sumaCost = 0.0
        noBatchsToEvaluate = 0

        # Ordenamos aleatoriamente la lista de pkls
        self.data_set_manager.Dataset_repo.ShuffleListPKLFiles()

        # Recorremos todos los batchs (internamente el data set manager administra la carga de los super batchs)
        for batch_index in range(self.data_set_manager.Dataset_repo.No_batchs_in_dataset):
            # self.TryUpdateLearningRate(epoch_index,batch_index)
            batch_index_in_superbatch, noRowsInBatch = self.data_set_manager.LoadRandomOrderDataSetXBatch(batch_index)

            error = self.evaluate_model_with_error(batch_index_in_superbatch, noRowsInBatch)
            sumaCost = sumaCost + error
            noBatchsToEvaluate = noBatchsToEvaluate + 1
            print("costo: " + str(error) + " Batch: " + str(batch_index) + "/" + str(
                self.data_set_manager.Dataset_repo.No_batchs_in_dataset - 1))
        promedio = sumaCost / noBatchsToEvaluate
        return promedio





