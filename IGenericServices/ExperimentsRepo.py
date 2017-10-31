import sqlite3
import pandas as pd

class ExperimentsRepo(object):
    def __init__(self,database_name, id_experiment):
        self.database_name = database_name
        self.experiment = self.BuscarExperimento(id_experiment)
        self.id_experiment =id_experiment

    def BuscarExperimento(self, id_experimento):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        id_ex = str(id_experimento)
        c.execute("select * from Experiments where Id = ? ",[id_ex])
        registro= c.fetchone()
        return registro

    def ObtenerInitialLearningRate(self):
        if self.experiment is None:
            return None
        return self.experiment[1]

    def ObtenerPKLsTrainReferenceFile(self):
        if self.experiment is None:
            return None
        return self.experiment[2]

    def ObtenerPKLsTestReferenceFile(self):
        if self.experiment is None:
            return None
        return self.experiment[3]

    def ObtenerStatus(self):
        if self.experiment is None:
            return None
        return self.experiment[4]

    def ObtenerBatchSize(self):
        if self.experiment is None:
            return None
        return self.experiment[5]

    def ObtenerSuperBatchSize(self):
        if self.experiment is None:
            return None
        return self.experiment[6]

    def ObtenerPKLTranRerferenceList(self):
        referenceFullPath = self.ObtenerPKLsTrainReferenceFile()
        listPKLs = []
        referenceList = pd.read_csv(referenceFullPath)
        for index, row in referenceList.iterrows():
            itemDict = {
                "pklFullPath": row["full_path_pkl"],
                "pklNoRpws": row["size_pkl"]
            }
            listPKLs.append(itemDict)
        return listPKLs


    def ObtenerMaxEpoch(self):
        if self.experiment is None:
            return None
        return self.experiment[7]

    def ObtenerFrecuencySaveWeigths(self):
        if self.experiment is None:
            return None
        return self.experiment[8]

    def ObtenerWithLRDecay(self):
        if self.experiment is None:
            return None
        if self.experiment[9] == 1:
            return True
        return False

    def ObtenerFrecuencyLRDecay(self):
         if self.experiment is None:
            return None
         return self.experiment[10]

    def ObtenerBatchActual(self):
        if self.experiment is None:
            return None
        return self.experiment[11]


    def ObtenerDecreaseNow(self):
        self.experiment = self.BuscarExperimento(self.id_experiment)
        if self.experiment is None:
            return None
        if self.experiment[12] == 1:
            return True
        return False

    def ObtenerIncreaseNow(self):
        self.experiment = self.BuscarExperimento(self.id_experiment)
        if self.experiment is None:
            return None
        if self.experiment[13] == 1:
            return True
        return False

    def ObtenerFolderWeightsPath(self):
        if self.experiment is None:
            return None
        return self.experiment[14]





    def SetFalseDecreaseNow(self):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Experiments SET ShouldDecreaseNow = {0} WHERE Id = {1}".format('0', self.id_experiment)

        c.execute(query)
        conn.commit()
        conn.close()


    def SetTrueDecreaseNow(self):
        query = "UPDATE Experiments SET ShouldDecreaseNow = {0} WHERE Id = {1}".format('1', self.id_experiment)
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        c.execute(query)
        conn.commit()
        conn.close()


    def SetTrueIncreaseNow(self):
        query = "UPDATE Experiments SET ShouldIncreaseNow = {0} WHERE Id = {1}".format('1', self.id_experiment)
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        c.execute(query)
        conn.commit()
        conn.close()

    def SetFalseIncreaseNow(self):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Experiments SET ShouldIncreaseNow = {0} WHERE Id = {1}".format('0', self.id_experiment)

        c.execute(query)
        conn.commit()
        conn.close()

    def UpdateLearningRate(self, new_learning_rate):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Experiments SET InitialLearningRate = {0} WHERE Id = {1}".format(new_learning_rate,self.id_experiment)

        c.execute(query)
        conn.commit()
        conn.close()


