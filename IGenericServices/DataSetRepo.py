import _pickle as cPickle
import os
import numpy as np

class DataSetRepo(object):
    def __init__(self, list_PKL_files, batch_size, superbatch_Size, no_rows = -1):
        self.Batch_size = int(batch_size)
        self.Superbatch_size = int(superbatch_Size)
        self.List_PKL_files = list_PKL_files
        self.No_rows = 0
        if (no_rows == -1):
            for pklInfo in self.List_PKL_files:
                self.No_rows = self.No_rows + self.GetNoRowsInPklFile(pklInfo['pklFullPath'])
        else:
            self.No_rows = int(no_rows)

        # calculamos el numero de batchs en cada super batch
        self.No_batchs_x_superbatch = self.Superbatch_size // self.Batch_size
        if ((self.Superbatch_size % self.Batch_size) != 0):
            raise ValueError('El numero Superbatch_size debe ser divisible por el Batch_size')

        # calculamos el numero de superbatchs, si el numero no es cerrado significa que hay batchs que no se estan tomando en cuenta,
        # en este caso debemos tomar en cuenta un superbatch mas y tambien afecta a el numero de batchs totales calculados en el dataset
        self.No_SuperBatches = self.No_rows // self.Superbatch_size
        if ((self.No_rows % self.Superbatch_size) != 0):
            no_rows_no_assigned = self.No_rows - (self.Superbatch_size * self.No_SuperBatches)
            no_batchs_faltantes = no_rows_no_assigned // self.Batch_size
            if (no_rows_no_assigned % self.Batch_size != 0):
                no_batchs_faltantes = no_batchs_faltantes + 1

            self.No_batchs_in_dataset = (self.No_batchs_x_superbatch * self.No_SuperBatches) + no_batchs_faltantes
            self.No_SuperBatches = self.No_SuperBatches + 1
        else:
            self.No_batchs_in_dataset = self.No_batchs_x_superbatch * self.No_SuperBatches

        #tambien debemos calcular el numero de rows en el ultimo batch ya que posiblemente no de un numero cerrado
        self.NoRowsInLastBatch = self.Batch_size
        if ((self.No_rows % self.Batch_size) != 0):
            #no_rows_not_assignen_in_last_batch =
            self.NoRowsInLastBatch = self.No_rows % self.Batch_size #self.Batch_size -no_rows_not_assignen_in_last_batch


        print ("DataSetRepo: Initial values assigned")


    def ShuffleListPKLFiles(self):
        noItemsListForShuffle = len(self.List_PKL_files)-1
        p = np.random.permutation(noItemsListForShuffle)
        part1 = np.asarray( self.List_PKL_files[0:noItemsListForShuffle])[p]
        part2= self.List_PKL_files[noItemsListForShuffle]
        self.List_PKL_files= np.append(part1,part2)
        return self.List_PKL_files



    def GetNoRowsInPklFile(self, pkl_full_path):
        fLoaded = open(pkl_full_path, 'rb',)
        data = cPickle.load(fLoaded, encoding='latin1')
        fLoaded.close()
        return data[0].shape[0]

    def GetRawDataSetBySuperBatchIndex(self, super_batchIndex,debug_mode=False):
            '''
            super_batchIndex es un indice de arreglo con inicio en  0
            Primero retorna X y despues Y, retorna arrays de numpy
            '''

            #Calculamos rango inciial y final en el dataset
            fullDataSet_startIndex = super_batchIndex * self.Superbatch_size
            fullDataSet_endIndex = fullDataSet_startIndex + self.Superbatch_size
            if (fullDataSet_endIndex > self.No_rows):
                fullDataSet_endIndex = self.No_rows



            rowsAlready_Loaded_index = 0
            firstLoad = True
            allRowsLoaded = False

            for pklInfo in self.List_PKL_files:

                rowsAlready_Loaded_index_ifLoadPKL = rowsAlready_Loaded_index + pklInfo['pklNoRpws']
                if (rowsAlready_Loaded_index_ifLoadPKL < fullDataSet_startIndex):
                    rowsAlready_Loaded_index = rowsAlready_Loaded_index + pklInfo['pklNoRpws']
                    #print ("-- Skip PKL: " + pklInfo['pklFullPath'])
                    #Mientras no carguemos la cantidad de datos necesarios para comenzar a tomar datos del siguiente superbatch saltamos la iteracion
                    continue

                #Los registros cargados con este pkl exceden a los que necesitamos, debemos buscar cuantos ya ha sido cargados para sabdr a partir de que indice debemos comenzar la carga
                #puede ser que la suma de registros se hizo con un pkl anterior y al sumar con los registros del actual este fue el que hizo que se alcanzara la meta
                #o puede ser que no se tomo ningun registro del pkl anterior en este caso solo son los registros de este pkl sera suficiente


                #ajustamos los registros cargados realmente porque puede ocurrir que el pkl tiene mas registros


                #Obtenemos cuantos registros debemos saltarnos de este pkl en otras palabras cuantos registros de este pkl ya han sido asignados a otro super batch
                # Caso cuandp los que ya han sido cargados son menos que los que se necesitan
                #Este es el indice de inicio para comenzar a cargar en el pkl

                indexStartPKL = fullDataSet_startIndex - rowsAlready_Loaded_index #asumimos que los registros cargados son menos de los que minimamente necesitmos
                #pero hay otro caso cuando los registros que minimammente necesitamos cargar son son menos que los que ya estan cargos
                noRows_availableInPKL =  pklInfo['pklNoRpws'] - indexStartPKL
                fullDataSet_endIndex_withpkl = fullDataSet_startIndex + noRows_availableInPKL

                if (fullDataSet_endIndex_withpkl < fullDataSet_endIndex):
                    #Cheamos si con la suma de todo este pkl alcanzamos la meta
                    # Si aun no llegamos a la meta de carga de registros
                    #apendizamos todo y continuamos en el loop
                    indexEndPKL = pklInfo['pklNoRpws']

                    fullDataSet_startIndex = fullDataSet_startIndex + noRows_availableInPKL

                    #registramos todos los registros de este dataset como cargados para avanzar a la siguiente carga (iteracion)
                    rowsAlready_Loaded_index = rowsAlready_Loaded_index + pklInfo['pklNoRpws']

                else:
                    #Si llegamos a la meta de carga de registros verificamos que solo carguemos los que nos hacen falta
                    #Calculamos los registros que nos sobran
                    norowsToAVoidInPKL = fullDataSet_endIndex_withpkl - fullDataSet_endIndex
                    # Calculamos los registros que nos faltan
                    indexEndPKL =pklInfo['pklNoRpws'] - norowsToAVoidInPKL
                    allRowsLoaded = True


                fLoaded = open(pklInfo['pklFullPath'], 'rb')
                data = cPickle.load(fLoaded,encoding='latin1')
                fLoaded.close()
                print("Carga de PKL:" + str(pklInfo['pklFullPath']) + " startIndex: " + str(
                    indexStartPKL) + " endIndex : " + str(indexEndPKL))

                dataX = data[0][indexStartPKL:indexEndPKL]
                dataY = data[1][indexStartPKL:indexEndPKL]



                if (firstLoad == True):
                    allXData = dataX
                    allYData = dataY
                    firstLoad = False
                else:
                    allXData = np.concatenate((allXData,dataX))
                    allYData = np.concatenate((allYData,dataY))

                if allRowsLoaded == True:
                    break

            return allXData, allYData
