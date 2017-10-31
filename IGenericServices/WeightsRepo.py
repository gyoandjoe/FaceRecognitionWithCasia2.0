import datetime

__author__ = 'win-g'

import _pickle as cPickle
import sqlite3
import os

class WeigthsRepo(object):
    def __init__(self, database_name,id_experiment):
        self.database_name = database_name
        self.id_experiment = id_experiment


    def SaveWeights(self, weights, folder_path, epoch=-1, batch_index=-1,super_batch_index = -1, iteration = -1,trainCost = -1, trainError = -1, costVal=-1, errorVal=-1, costTest =-1, errorTest=-1):
        fecha = datetime.datetime.now().strftime("%d-%m-%Y %H %M %S")
        fileName = "w_idExp " + str(self.id_experiment) + "_SBatchindex " + str(super_batch_index) + "_fecha " + fecha + "_epoch " + str(epoch)+ "_batch " + str(batch_index) + "_iter " + str(iteration) + '.pkl'

        folder_path = folder_path + "\\" + str(self.id_experiment)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pklFullPath = folder_path + "\\" + fileName

        f = open(pklFullPath, 'w+b')
        cPickle.dump(weights, f, protocol=2)
        f.close()

        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "INSERT INTO Weights VALUES (NULL,{0},\'{1}\',\'{2}\',{3},{4},{5},{6},{7},{8},{9},{10},{11},{12})".format(
            str(self.id_experiment), #0
            pklFullPath, #1
            fecha, #2
            epoch, #3
            batch_index, #4
            super_batch_index, #5
            iteration, #6
            trainError, #7
            trainCost, #8
            costVal, #9
            errorVal, #10
            costTest, #11
            errorTest) #12

        c.execute(query)
        conn.commit()
        newId = c.lastrowid
        conn.close()
        return newId

    def GetWeithsInfoById(self, id_weigths):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        c.execute("SELECT * FROM Weights WHERE Id = ?",[str(id_weigths)])
        registro = c.fetchone()
        conn.close()

        return registro

    def GetGeigthsByExperimentId(self, id_experiment):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        c.execute("SELECT * FROM Weights WHERE IdExperiment = ?",[str(id_experiment)])
        registros = c.fetchall()
        conn.close()

        return registros

    def UpdateCostForWeigth(self, id_weigth, cost):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET TrainCost = {0} WHERE Id = {1}".format(cost,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateErrorForWeigth(self, id_weigth, error):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET TrainError = {0} WHERE Id = {1}".format(error,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateTestCostForWeigth(self, id_weigth, costTest):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET TestCost = {0} WHERE Id = {1}".format(costTest,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateTestErrorForWeigth(self, id_weigth, errorTest):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET TestError = {0} WHERE Id = {1}".format(errorTest,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateValCostForWeigth(self, id_weigth, costVal):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET ValidCost = {0} WHERE Id = {1}".format(costVal,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateValErrorForWeigth(self, id_weigth, errorVal):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET ValidError = {0} WHERE Id = {1}".format(errorVal,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return