import theano
import theano.tensor as T
import numpy as np

from Core.DataSetManager import DataSetManager
from IGenericServices.DataSetRepo import DataSetRepo


class Trainer(object):

    def __init__(self,
                 idExperiment,
                 logger,
                 cnn,
                 experimentsManager,
                 weightsRepo,norows_trainset=690569):
        """
        :param logger: ILoggerService
        :param cnn: Arquitecture implementation
        :param id_experiment: Para loggear errores debemos saber a que experimento estamos apuntando
        :param batch_size:
        :param initial_weights:
        :param max_epochs:
        :param with_lr_decay:
        :param initial_learning_rate:
        :param saveWeigthsFrecuency:
        :param frecuency_lr_decay:
        :param p_DropOut:
        """
        self.id_experiment = idExperiment
        self.logger = logger

        self.CNN = cnn
        self.experimentsManager = experimentsManager
        self.weightsRepo = weightsRepo
        self.experiment = experimentsManager.LoadExpermentById(self.id_experiment)

        self.data_set_repo = DataSetRepo(
            list_PKL_files=self.experiment.pkl_train_referenceList,
            batch_size=self.experiment.batch_size,
            superbatch_Size=self.experiment.super_batch_size,
            no_rows=norows_trainset
        )
        self.data_set_manager = DataSetManager(self.experiment.batch_size,self.experiment.super_batch_size,self.data_set_repo)

        self.saveWeigthsFrecuency = self.experiment.saveWeigthsFrecuency
        self.frecuency_lr_decay = self.experiment.frecuency_lr_decay

        learningRate = T.fscalar()
        index = T.lscalar()
        noRowsInBatch= T.lscalar()

        self.data_set_manager.Dataset_repo.ShuffleListPKLFiles()
        self.data_set_manager.LoadRandomOrderDataSetXBatch(batch_index=0)


        self.grads = T.grad(self.CNN.CostFunction, self.CNN.Weigths, disconnected_inputs="raise")
        self.updates = [
            (param_i, param_i + (learningRate * grad_i))
            for param_i, grad_i in zip(self.CNN.Weigths, self.grads)
        ]


        self.train_model = theano.function(
            inputs= [index,learningRate,noRowsInBatch],
            outputs= self.CNN.CostFunction,
            updates=self.updates,
            givens={
                self.CNN.image_input: self.data_set_manager.dataSetX[index * noRowsInBatch: (index + 1) * noRowsInBatch],
                self.CNN.y: self.data_set_manager.dataSetY          [index * noRowsInBatch: (index + 1) * noRowsInBatch],
                self.CNN.batch_size:noRowsInBatch
            }
            #on_unused_input='warn'
        )

        return

    def Train(self, current_epoch=0,current_batch=0):

        for epoch_index in range(self.experiment.max_epochs):

            if epoch_index < current_epoch:
                continue

            #Ordenamos aleatoriamente la lista de pkls
            self.data_set_manager.Dataset_repo.ShuffleListPKLFiles()

            #Recorremos todos los batchs (internamente el data set manager administra la carga de los super batchs)
            for batch_index in range(self.data_set_manager.Dataset_repo.No_batchs_in_dataset):
                if (batch_index <  current_batch):#self.data_set_manager.Dataset_repo.No_batchs_in_dataset -5):
                    continue

                if (self.experimentsManager.experimentRepo.ObtenerDecreaseNow() == True):
                    self.experiment.learning_rate *= 0.1
                    self.experimentsManager.experimentRepo.UpdateLearningRate(self.experiment.learning_rate)
                    self.experimentsManager.experimentRepo.SetFalseDecreaseNow()
                    print("Decremento mandatorio, learning rate: " + str(self.experiment.learning_rate))

                if (self.experimentsManager.experimentRepo.ObtenerIncreaseNow() == True):
                    self.experiment.learning_rate /= 0.1
                    self.experimentsManager.experimentRepo.UpdateLearningRate(self.experiment.learning_rate)
                    self.experimentsManager.experimentRepo.SetFalseIncreaseNow()
                    print("Incremento mandatorio, learning rate: " + str(self.experiment.learning_rate))

                #self.TryUpdateLearningRate(epoch_index,batch_index)
                batch_index_in_superbatch,noRowsInBatch =  self.data_set_manager.LoadRandomOrderDataSetXBatch(batch_index)

                cost = self.train_model(batch_index_in_superbatch, self.experiment.learning_rate,noRowsInBatch)

                print("costo: " + str(cost) + " epoca: " + str(epoch_index) + " Batch: " + str(batch_index) +"/" + str(self.data_set_manager.Dataset_repo.No_batchs_in_dataset-1) +" Learning Rate: " + str(self.experiment.learning_rate))
                self.logger.LogTrain(
                    costo=cost,
                    epoch_index= str(epoch_index),
                    super_batch_index=-1,
                    batch_index=str(batch_index),
                    learning_rate=str(self.experiment.learning_rate)
                )

            #En una etapa posterior se calculan los costos para el testset
            if (epoch_index + 1) % self.saveWeigthsFrecuency == 0:
                self.weightsRepo.SaveWeights(
                    weights=self.CNN.GetWeightsValues(),
                    folder_path= self.experiment.FolderWeightsPath,
                    epoch=epoch_index,
                    batch_index=-1,
                    super_batch_index = -1,
                    iteration = -1,
                    trainCost = -1,
                    trainError = -1,
                    costVal=-1,
                    errorVal=-1,
                    costTest =-1,
                    errorTest=-1)

"""
    def TryUpdateLearningRate(self, epoch_index, batch_index):
        if epoch_index != 0 and self.with_lr_decay == True and epoch_index % self.frecuency_lr_decay == 0:
            self.learning_rate *= self.learning_rate_ratio_decay
        elif self.with_lr_decay == False:
            decreaseNow = self.experimentsRepo.ObtenerDecreaseNow()
            increaseNow = self.experimentsRepo.ObtenerIncreaseNow()
            if decreaseNow == True:
                self.experimentsRepo.UpdateLearningRate(self.learning_rate)
                self.experimentsRepo.SetFalseDecreaseNow()
                self.learning_rate *= 0.1
                print("Decremento mandatorio, learning rate: " + str(self.learning_rate))
            elif increaseNow == True:
                self.experimentsRepo.UpdateLearningRate(self.learning_rate)
                self.experimentsRepo.SetFalseIncreaseNow()
                self.learning_rate /= 0.1

        return
"""