from theano.tensor.shared_randomstreams import RandomStreams

__author__ = 'Giovanni'

import theano.tensor as T
import theano


import numpy as np

class DataSetManager(object):
    def __init__(self, batch_size, superbatch_Size,dataset_repo):
        """
        :param batchSize: numero de registros por cada batch
        :param superBatchSize: numero de registros por cada super Batch
        :return:
        """
        self.Batch_size = int(batch_size)
        self.Superbatch_size =int(superbatch_Size)
        self.Dataset_repo = dataset_repo

        self.Current_superbatch_loaded = None
        self.batch_index_in_SuperBatch = None




        #calculamos el numero de batchs en cada super batch
        #self.No_batchs_x_superbatch = self.Superbatch_size // self.Batch_size
        #if ((self.Superbatch_size % self.Batch_size) != 0):
        #    raise ValueError('DataSetManagerError: El numero Superbatch_size debe ser divisible por el Batch_size')
            #self.No_batchs_x_superbatch=self.No_batchs_x_superbatch + 1



        #calculamos el numero de superbatchs, si el numero no es cerrado significa que hay batchs que no se estan tomando en cuenta,
        # en este caso debemos tomar en cuenta un superbatch mas y tambien afecta a el numero de batchs totales calculados en el dataset
        #self.No_SuperBatches = self.Dataset_repo.No_rows // self.Superbatch_size
        #if ((self.Dataset_repo.No_rows % self.Superbatch_size) != 0):
        #    no_rows_no_assigned = self.Dataset_repo.No_rows - (self.No_batchs_x_superbatch * self.No_SuperBatches * self.Dataset_repo.No_rows )
        #    no_batchs_faltantes =no_rows_no_assigned  //  self.Batch_size
        #    if (no_rows_no_assigned  %  self.Batch_size != 0):
        #        no_batchs_faltantes = no_batchs_faltantes + 1
        #    self.No_batchs_in_dataset = (self.No_batchs_x_superbatch * self.No_SuperBatches) + no_batchs_faltantes
        #    self.No_SuperBatches= self.No_SuperBatches + 1
        #else:
        #    self.No_batchs_in_dataset = self.No_batchs_x_superbatch * self.No_SuperBatches

        self.dataSetX = None
        self.dataSetY = None
        self.NoRowsInBatch = None


    def LoadRandomOrderDataSetXBatch(self, batch_index,removeRandom=False):
        """
        Metodo para cargar el shared dataset en memoria de gpu a partir del batch_index, si el batch solicitado ya esta cargado no se volver a acargar
        :param batch_index: es un numero que puede ser muy grande, ya que estos batches estan contenidos en los superbatches, y su numeracion atraviesa estos super batches, debe comenzar en 0 y no en 1
        :return:
        """
        if (batch_index > self.Dataset_repo.No_batchs_in_dataset):
            print("Index batch requested out of range")

        superBatchIndexRequested = batch_index // self.Dataset_repo.No_batchs_x_superbatch
        self.batch_index_in_SuperBatch = batch_index - (superBatchIndexRequested *  self.Dataset_repo.No_batchs_x_superbatch)

        self.NoRowsInBatch = self.Batch_size
        if ((self.Dataset_repo.No_batchs_in_dataset-1) == batch_index):  # Ultimo batch, tal vez no tenga suficientes registros
            self.NoRowsInBatch = self.Dataset_repo.NoRowsInLastBatch

        if (self.Current_superbatch_loaded == superBatchIndexRequested and self.Current_superbatch_loaded is not None):
            #print ("El Batch ya ha sido cargadp")
            return (self.batch_index_in_SuperBatch,self.NoRowsInBatch)
        else:
            #Se debera cargar el batch
            self.Current_superbatch_loaded = superBatchIndexRequested
            self.LoadRandomRawDataSetXSuperBatchIndex(self.Current_superbatch_loaded,removeRandom)
            print("Nueva carga de batch, No de registros: " + str(self.NoRowsInBatch))

        return (self.batch_index_in_SuperBatch,self.NoRowsInBatch)



    def LoadRandomRawDataSetXSuperBatchIndex(self, indexSuperBatch,removeRandom=False):
        rawX, rawY = self.Dataset_repo.GetRawDataSetBySuperBatchIndex(indexSuperBatch)
        newShapeX = (rawX.shape[0], 1, rawX.shape[1],rawX.shape[2])

        newx = np.reshape(np.asarray(rawX, dtype=theano.config.floatX),(newShapeX))
        newy = np.asarray(rawY, dtype=int)

        if removeRandom==False:
            p = np.random.permutation(int(rawX.shape[0]))
            newy = newy[p]
            newx = newx[p]

        if (self.dataSetX is None or self.dataSetY is None):
            self.dataSetX = theano.shared(newx, borrow=True)
            self.dataSetY = theano.shared(newy, borrow=True)
        else:
            self.dataSetX.set_value(newx,borrow=True)
            self.dataSetY.set_value(newy,borrow=True)
        return rawX.shape[0]