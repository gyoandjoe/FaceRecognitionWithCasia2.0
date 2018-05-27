import sqlite3

from Core import ExperimentDomain
from IGenericServices.ExperimentsRepo import ExperimentsRepo


class ExperimentsManager(object):

    def __init__(self,dbFile):
        self.dataBaseFile = dbFile
        self.experiment = None
        self.experimentRepo = None
        return

    def CreateExperiment(self, initialLearningRate, PKLsTrainReferenceFile, PKLsTestReferenceFile,PKLsValReferenceFile, status, batchSize, superBatchSize, maxEpochTraining, FolderWeightsPath,epochFrecSaveWeights = 10, withLRDecay = 0, EpochFrecLRDecay=0, BatchActual=0, ShouldDecreaseNow=0, ShouldIncreaseNow=0):
        conn = sqlite3.connect(self.dataBaseFile)
        c = conn.cursor()

        query="INSERT INTO Experiments VALUES (NULL ,"+str(initialLearningRate) + ",'" + str(PKLsTrainReferenceFile) + "','" + str(PKLsTestReferenceFile) + "','" + str(status) + "'," + str(batchSize) + "," + str(superBatchSize) + "," + str(maxEpochTraining) + "," + str(epochFrecSaveWeights) + "," + str(withLRDecay) + "," + str(EpochFrecLRDecay) + "," + str(BatchActual) + "," + str(ShouldDecreaseNow) + "," + str(ShouldIncreaseNow) + ",'"+str(FolderWeightsPath)+"','" + str(PKLsValReferenceFile) + "')"
        print (query)
        c.execute(query)
        newId = c.lastrowid
        """
        Id,
        TrainDataSetFile,
        TestDataSetFile,
        BatchSize,
        InitialLearningRate,
        Status,
        BatchActual,
        MaxEpoch,
        EpochFrecSaveWeights,
        WithLRDecay,
        EpochFrecLRDecay
        """
        conn.commit()


            #wg =
            #debemos generar los pesos iniciales y guardarlos y generar un primer registro


        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        conn.close()
        return newId

    def LoadExpermentById(self, idExperiment):
        self.experimentRepo = ExperimentsRepo(self.dataBaseFile, idExperiment)
        self.experiment = ExperimentDomain.ExperimentDomain(
            pkl_train_referenceFullpath=self.experimentRepo.ObtenerPKLsTrainReferenceFile(),
            pkl_test_referenceFullpath=self.experimentRepo.ObtenerPKLsTestReferenceFile(),
            pkl_validation_referenceFullpath=self.experimentRepo.ObtenerPKLsValidationReferenceFile(),

            pkl_train_referenceList=self.experimentRepo.ObtenerPKLTranRerferenceList(),
            pkl_test_referenceList=self.experimentRepo.ObtenerPKLTestRerferenceList(),
            pkl_validation_referenceList=self.experimentRepo.ObtenerPKLValidationReferenceList(),


            max_epochs=self.experimentRepo.ObtenerMaxEpoch(),
            with_lr_decay=self.experimentRepo.ObtenerWithLRDecay(),
            learning_rate_ratio_decay=0.1,
            initial_learning_rate=self.experimentRepo.ObtenerInitialLearningRate(),
            saveWeigthsFrecuency=self.experimentRepo.ObtenerFrecuencySaveWeigths(),
            frecuency_lr_decay=self.experimentRepo.ObtenerFrecuencyLRDecay(),
            batch_size=self.experimentRepo.ObtenerBatchSize(),
            super_batch_size=self.experimentRepo.ObtenerSuperBatchSize(),

            folderWeightsPath=self.experimentRepo.ObtenerFolderWeightsPath(),

        )

        return self.experiment







