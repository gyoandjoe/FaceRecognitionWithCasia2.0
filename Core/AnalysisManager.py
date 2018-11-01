import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Arquitectures.GCrescencioArqui.CNNGCresc import CNNGCresc
from Arquitectures.GCrescencioArqui.ExperimentInitializer import ExperimentInitializer
from Arquitectures.GCrescencioArqui.MetaDataGCresc import layers_metaData
from Arquitectures.GCrescencioArqui.MetaDataGCresc import experimentMetaData
import theano.tensor as T
import numpy as np
from Core.DataSetManager import DataSetManager
from IGenericServices.DataSetRepo import DataSetRepo

#Primero debemos cargar los datos del experimento

#despues debemos cargar los pesos
from Core.DataSetManager import DataSetManager
from Core.ExperimentsManager import ExperimentsManager
from Core.Validator_ValTest import Validator_ValTest
from Core.WeightsManager import WeightsManager
from IGenericServices.ExperimentsRepo import ExperimentsRepo
from IGenericServices.LoggerRepo import LoggerRepo
from IGenericServices.WeightsRepo import WeigthsRepo
import csv

class AnalysisManager(object):
    def __init__(self, data_base, analisys_repo,id_experiment, ):
        self.data_base=data_base
        self.analisys_repo = analisys_repo #Analisys_Repo.AnalisysRepo(data_base=data_base )
        self.id_experiment = id_experiment

    def AnalizarInRealTIme(self, velocity_update=1, start_epoch=0):
        #plt.axis([0, 10, 0, 1])
        plt.ion()

        while True:
            registros = self.analisys_repo.ObtenerLogsTraining(self.id_experiment,start_epoch)
            df = pd.DataFrame(registros, columns=['Id','IdExperiment','FechaRegistro','Costo','TipoLog','EpochIndex','BatchIndex','SuperBatchIndex','LearningRate','Contenido']) #,dtype=[('Contenido', np.float64)]
            df.convert_objects(convert_numeric=True)
            df['Costo'] = df['Costo'].astype(np.float64)
            grouped = df.groupby('EpochIndex')
            #for registro in df['contenido'].values:
            #    print registro
            xx=grouped.groups.keys()

            yy=grouped['Costo'].mean().values * -1

            x = np.asarray(list(xx), dtype=int)
            y = np.asarray(yy, dtype=np.float64)
            plt.cla()
            plt.plot(x,y,'-')
            #print y
            #plt.show()0....

            plt.pause(velocity_update)

    def BuildWeigthsErrorAndCost_TestSet(self, noRows_Testset, skipuntilidw, justOne):
        # Calculo total del costo y error por todos los datos, pero por cada conjunto de pesos generados
        # experiment_repo = Experiments.ExperimentsRepo.ExperimentsRepo(bd, id_experiment)
        # weigths_repo = WeigthsRepo.WeigthsRepo(bd, weigths_path)
        database_relative_path = self.data_base
        id_experiment = self.id_experiment
        wr = WeigthsRepo(database_name=database_relative_path, id_experiment=id_experiment)
        wm = WeightsManager(
            database_name=database_relative_path,
            weights_repo=wr
        )

        weigthsOfExperiment = wm.GetListOfWeightsByIdExperiment(id_experiment)

        print('--------------------------- Test SET -------------------------------------------------')

        print("Calculando Errores en TestSet y costos en Test set")

        for w in weigthsOfExperiment:
            idW = w[0]
            if idW < skipuntilidw:
                continue
            # Ahora cargamos los weights


            iws = wm.LoadWeightsXId(idW)
            random_droput = np.random.RandomState(12345)
            rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999899))

            # terminar de instanciar esto
            cnn = CNNGCresc(
                layers_metaData=layers_metaData,
                initWeights=iws,
                srng=rng_droput,
                no_channels_imageInput=1,
                isTraining=0,
                pDropOut=0.7  # antes 0.60
            )

            # exp 9 pDropout=0.65
            logger = LoggerRepo(id_experiment=id_experiment, database_name=database_relative_path)
            experimentsManager = ExperimentsManager(database_relative_path)

            self.experiment = experimentsManager.LoadExpermentById(self.id_experiment)

            self.data_set_repo = DataSetRepo(
                list_PKL_files=self.experiment.pkl_test_referenceList,
                batch_size=self.experiment.batch_size,
                superbatch_Size=self.experiment.super_batch_size,
                no_rows=noRows_Testset
            )
            self.data_set_manager = DataSetManager(self.experiment.batch_size, self.experiment.super_batch_size,
                                                   self.data_set_repo)

            validator = Validator_ValTest(
                data_set_manager=self.data_set_manager,
                logger=logger,
                cnn=cnn,
                weightsRepo=wr
            )

            averageError = validator.CalculateError()
            wm.UpdateTestErrorWeigth(idW, averageError)
            print("--------[Test Set] El error promedio es: " + str(averageError))

            averageCost = validator.CalculateCost()
            wm.UpdateTestCostWeigth(idW, averageCost)
            print("--------[Test Set] El costo promedio es: " + str(averageCost))

            if justOne == True:
                break
        print("End Test :)")

    def BuildWeigthsErrorAndCost_ValSet(self,noRows_valset,skipuntilidw,justOne,pdropout):
        # Calculo total del costo y error por todos los datos, pero por cada conjunto de pesos generados
        #experiment_repo = Experiments.ExperimentsRepo.ExperimentsRepo(bd, id_experiment)
        #weigths_repo = WeigthsRepo.WeigthsRepo(bd, weigths_path)
        database_relative_path = self.data_base
        id_experiment = self.id_experiment
        wr = WeigthsRepo(database_name=database_relative_path, id_experiment=id_experiment)
        wm = WeightsManager(
            database_name=database_relative_path,
            weights_repo=wr
        )

        weigthsOfExperiment = wm.GetListOfWeightsByIdExperiment(id_experiment)


        print('--------------------------- Validation SET -------------------------------------------------')

        print("Calculando Errores en validationSet y costos en validation set")

        for w in weigthsOfExperiment:
            idW = w[0]
            if idW < skipuntilidw:
                continue
            # Ahora cargamos los weights


            iws = wm.LoadWeightsXId(idW)
            random_droput = np.random.RandomState(12345)
            rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999899))

            # terminar de instanciar esto
            cnn = CNNGCresc(
                layers_metaData=layers_metaData,
                initWeights=iws,
                srng=rng_droput,
                no_channels_imageInput=1,
                isTraining=0,
                pDropOut=pdropout  # antes 0.60
            )

            # exp 9 pDropout=0.65
            logger = LoggerRepo(id_experiment=id_experiment, database_name=database_relative_path)
            experimentsManager = ExperimentsManager(database_relative_path)

            self.experiment = experimentsManager.LoadExpermentById(self.id_experiment)

            self.data_set_repo = DataSetRepo(
                list_PKL_files=self.experiment.pkl_validation_referenceList,
                batch_size=self.experiment.batch_size,
                superbatch_Size=self.experiment.super_batch_size,
                no_rows=noRows_valset
            )
            self.data_set_manager = DataSetManager(self.experiment.batch_size, self.experiment.super_batch_size,
                                                   self.data_set_repo)

            validator = Validator_ValTest(
                data_set_manager=self.data_set_manager,
                logger=logger,
                cnn=cnn,
                weightsRepo=wr
            )

            averageError = validator.CalculateError()
            wm.UpdateValErrorWeigth(idW, averageError)
            print("--------[Validation Set] El error promedio es: " + str(averageError))

            averageCost = validator.CalculateCost()
            wm.UpdateValCostWeigth(idW, averageCost)
            print("--------[Validation Set] El costo promedio es: " + str(averageCost))

            if justOne == True:
                break

        print("End Validation :)")

    def BuildWeigthsErrorAndCost_TrainSet(self,no_rows_trainset, skipuntilidw, justOne):
        # Calculo total del costo y error por todos los datos, pero por cada conjunto de pesos generados
        #experiment_repo = Experiments.ExperimentsRepo.ExperimentsRepo(bd, id_experiment)
        #weigths_repo = WeigthsRepo.WeigthsRepo(bd, weigths_path)
        database_relative_path = self.data_base
        id_experiment = self.id_experiment
        wr = WeigthsRepo(database_name=database_relative_path, id_experiment=id_experiment)
        wm = WeightsManager(
            database_name=database_relative_path,
            weights_repo=wr
        )

        weigthsOfExperiment = wm.GetListOfWeightsByIdExperiment(id_experiment)


        print('--------------------------- Validation SET -------------------------------------------------')

        print("Calculando Errores en validationSet y costos en validation set")

        for w in weigthsOfExperiment:
            idW = w[0]
            if idW < skipuntilidw:
                continue
            # Ahora cargamos los weights


            iws = wm.LoadWeightsXId(idW)
            random_droput = np.random.RandomState(12345)
            rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999899))

            # terminar de instanciar esto
            cnn = CNNGCresc(
                layers_metaData=layers_metaData,
                initWeights=iws,
                srng=rng_droput,
                no_channels_imageInput=1,
                isTraining=0,
                pDropOut=0.7  # antes 0.60
            )

            # exp 9 pDropout=0.65
            logger = LoggerRepo(id_experiment=id_experiment, database_name=database_relative_path)
            experimentsManager = ExperimentsManager(database_relative_path)

            self.experiment = experimentsManager.LoadExpermentById(self.id_experiment)

            self.data_set_repo = DataSetRepo(
                list_PKL_files=self.experiment.pkl_train_referenceList,
                batch_size=self.experiment.batch_size,
                superbatch_Size=self.experiment.super_batch_size,
                no_rows=no_rows_trainset
            )
            self.data_set_manager = DataSetManager(self.experiment.batch_size, self.experiment.super_batch_size,
                                                   self.data_set_repo)

            validator = Validator_ValTest(
                data_set_manager=self.data_set_manager,
                logger=logger,
                cnn=cnn,

                weightsRepo=wr
            )


            averageError = validator.CalculateError()
            wm.UpdateTrainErrorWeigth(idW, averageError)
            print("--------[Test Set] El error promedio es: " + str(averageError))


            averageCost = validator.CalculateCost()
            wm.UpdateTrainCostWeigth(idW, averageCost)
            print("--------[Test Set] El costo promedio es: " + str(averageCost))

            if justOne == True:
                break

        print('--------------------------- TRAIN SET -------------------------------------------------')




        print("End Validation :)")

    def GraficarCostosXEpocaXDataSet(self, id_experiment):
        data_weigths = self.analisys_repo.GetWeigthsByXIdExperiment(id_experiment)
        data = pd.DataFrame(data_weigths, columns=['Id',
        'IdExperiment',
        'PKLFullPah',
        'FechaRegistro',
        'Epoch',
        'BatchIndex',
        'SuperBatchIndex',
        'Iteracion',
        'TrainError',
        'TrainCost',
        'ValidCost',
        'ValidError',
        'TestCost',
        'TestError'])
        epochs = data['Epoch']
        trainCost = data['TrainCost']
        valCost = data['ValidCost']
        testCost = data['TestCost']

        xEpochs = np.asarray(epochs.values,  dtype=int)
        yTrain = np.asarray(trainCost.values, dtype=np.float64)
        yTrain = yTrain * -1
        yVal = np.asarray(valCost.values, dtype=np.float64)
        yVal = yVal * -1
        yTest = np.asarray(testCost.values, dtype=np.float64)


        plt.plot(xEpochs,yTrain,'b--')
        plt.plot(xEpochs,yVal,'r--')
        #plt.plot(xEpochs,yTest,'g^')
        plt.title('Cost Curve id experiment(' + str(id_experiment) + ')')
        plt.show()
        return

    def GraficarLearningCurve(self, id_experiment,id_weigths_1_de_4,id_weigths_2_de_4,id_weigths_3_de_4,id_weigths_4_de_4):

        wr = WeigthsRepo(database_name=self.data_base, id_experiment=id_experiment)

        data_weigths_1_de_4 = wr.GetWeithsInfoById(id_weigths_1_de_4)
        test_error_1_de_4 = data_weigths_1_de_4[13]
        training_error_1_de_4 = data_weigths_1_de_4[8]

        data_weigths_2_de_4 = wr.GetWeithsInfoById(id_weigths_2_de_4)
        test_error_2_de_4 =data_weigths_2_de_4[13]
        training_error_2_de_4 = data_weigths_2_de_4[8]

        data_weigths_3_de_4 = wr.GetWeithsInfoById(id_weigths_3_de_4)
        test_error_3_de_4 = data_weigths_3_de_4[13]
        training_error_3_de_4 = data_weigths_3_de_4[8]

        data_weigths_4_de_4 = wr.GetWeithsInfoById(id_weigths_4_de_4)
        test_error_4_de_4 = data_weigths_4_de_4[13]
        training_error_4_de_4 = data_weigths_4_de_4[8]



        test_x_size = np.asarray([175500,351000,526500,684000],  dtype=int)
        test_y_error = np.asarray([test_error_1_de_4,test_error_2_de_4,test_error_3_de_4,test_error_4_de_4], dtype=np.float64)

        train_x_size = np.asarray([175500, 351000, 526500 , 684000], dtype=int)
        train_y_error = np.asarray([training_error_1_de_4, training_error_2_de_4, training_error_3_de_4, training_error_4_de_4],
                                  dtype=np.float64)

        my_xticks = [175500,351000,526500,684000]
        plt.xticks(test_x_size, my_xticks)
        plt.xticks(train_x_size, my_xticks)

        plt.plot(test_x_size,test_y_error,'r--',)
        plt.plot(train_x_size,train_y_error,'b--')
        plt.ylabel('Error')
        plt.xlabel('Tamaño training set')

        plt.title('Learning curve(' + str(id_experiment)+')')
        plt.show()
        return

    def GraficarErrorsXEpocaXDataSet(self, id_experiment):
        data_weigths = self.analisys_repo.GetWeigthsByXIdExperiment(id_experiment)
        data = pd.DataFrame(data_weigths, columns=['Id',
        'IdExperiment',
        'PKLFullPah',
        'FechaRegistro',
        'Epoch',
        'BatchIndex',
        'SuperBatchIndex',
        'Iteracion',
        'TrainError',
        'TrainCost',
        'ValidCost',
        'ValidError',
        'TestCost',
        'TestError'])
        epochs = data['Epoch']
        trainError = data['TrainError']
        valError = data['ValidError']
        testError = data['TestError']

        xEpochs = np.asarray(epochs.values,  dtype=int)
        yTrain = np.asarray(trainError.values, dtype=np.float64)
        yVal = np.asarray(valError.values, dtype=np.float64)
        yTest = np.asarray(testError.values, dtype=np.float64)

        plt.plot(xEpochs,yTrain,'b--')
        plt.plot(xEpochs,yVal,'r--')
        #plt.plot(xEpochs,yTest,'g^')
        plt.title('Error Curve id experiment(' + str(id_experiment)+')')
        plt.show()
        return








    def BuildResults_ValSet(self,noRows_valset,skipuntilidw,justOne):
        # Calculo total del costo y error por todos los datos, pero por cada conjunto de pesos generados
        #experiment_repo = Experiments.ExperimentsRepo.ExperimentsRepo(bd, id_experiment)
        #weigths_repo = WeigthsRepo.WeigthsRepo(bd, weigths_path)
        database_relative_path = self.data_base
        id_experiment = self.id_experiment
        wr = WeigthsRepo(database_name=database_relative_path, id_experiment=id_experiment)
        wm = WeightsManager(
            database_name=database_relative_path,
            weights_repo=wr
        )

        weigthsOfExperiment = wm.GetListOfWeightsByIdExperiment(id_experiment)


        print('--------------------------- Validation SET -------------------------------------------------')

        print("Calculando Errores en validationSet y costos en validation set")
        csvfile = open(r'D:\Gyo\Dev\Thesis\dist2\analisys\DataSet_analisys_resultts.csv', 'w', newline='')
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)

        for w in weigthsOfExperiment:
            idW = w[0]
            if idW < skipuntilidw:
                continue
            # Ahora cargamos los weights

            iws = wm.LoadWeightsXId(idW)
            random_droput = np.random.RandomState(12345)
            rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999899))

            # terminar de instanciar esto
            cnn = CNNGCresc(
                layers_metaData=layers_metaData,
                initWeights=iws,
                srng=rng_droput,
                no_channels_imageInput=1,
                isTraining=0,
                pDropOut=0.7  # antes 0.60
            )

            # exp 9 pDropout=0.65
            logger = LoggerRepo(id_experiment=id_experiment, database_name=database_relative_path)
            experimentsManager = ExperimentsManager(database_relative_path)

            self.experiment = experimentsManager.LoadExpermentById(self.id_experiment)

            self.data_set_repo = DataSetRepo(
                list_PKL_files=self.experiment.pkl_validation_referenceList,
                batch_size=self.experiment.batch_size,
                superbatch_Size=self.experiment.super_batch_size,
                no_rows=noRows_valset
            )
            self.data_set_manager = DataSetManager(self.experiment.batch_size, self.experiment.super_batch_size,
                                                   self.data_set_repo)



            validator = Validator_ValTest(
                data_set_manager=self.data_set_manager,
                logger=logger,
                cnn=cnn,
                weightsRepo=wr,
                removeRandom=True
            )

            result = validator.CalculateResults()
            rrr = zip(result[0],result[1],result[2])
            for rr in rrr:
                csvwriter.writerow([rr[0],rr[1],rr[2]])
            csvfile.close()

            if justOne == True:
                break

        print("End Validation :)")
