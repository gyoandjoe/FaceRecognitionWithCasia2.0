
import _pickle as cPickle


class WeightsManager(object):
    def __init__(self, database_name, weights_repo):
        self.database_name = database_name
        self.repo = weights_repo
        self.paramsForInitWeigths = None

    def LoadWeightsXId(self, idWeigths):
        #primero obtenemos el fullpath
        #despues obtenemos datos raw del fullpath
        #con regresar los datos raw sera suficiente porque las capas son capaces de cargarlos en gpu (shared)
        rawWeights = self.LoadRawWeigths(idWeigths)
        return rawWeights


    def LoadRawWeigths(self, idWeigths):
        wf = self.repo.GetWeithsInfoById(idWeigths)
        if wf is not None:
            fileName = wf[2]  # FileName
            fLoaded = open(fileName, 'rb')
            data = cPickle.load(fLoaded)
            fLoaded.close()
            return data
        raise('No se encontro el id correcto')

    def GetListOfWeightsByIdExperiment(self, id_experiment):
        return self.repo.GetGeigthsByExperimentId(id_experiment)

    def UpdateValCostWeigth(self, id_weigth, cost):
        self.repo.UpdateValCostForWeigth(id_weigth, cost)
        return

    def UpdateValErrorWeigth(self, id_weigth, error):
        self.repo.UpdateValErrorForWeigth(id_weigth, error)
        return

    def UpdateTestCostWeigth(self, id_weigth, cost):
        self.repo.UpdateTestCostForWeigth(id_weigth, cost)
        return

    def UpdateTestErrorWeigth(self, id_weigth, error):
        self.repo.UpdateTestErrorForWeigth(id_weigth, error)
        return

    def UpdateTrainCostWeigth(self, id_weigth, cost):
        self.repo.UpdateCostForWeigth(id_weigth, cost)
        return

    def UpdateTrainErrorWeigth(self, id_weigth, error):
        self.repo.UpdateErrorForWeigth(id_weigth, error)
        return

    def SetInitParamsForTypeInit(self, params):
        """
        :type lowAndHigh_c1Values: object
        """
        self.paramsForInitWeigths = params

    def GetInitParamsWeigthsInStringFormat(self, type_init_weigths_function):
        niceFormat = ""
        if type_init_weigths_function is DistTypes.uniform:
            lowAndHigh_c1Values = self.paramsForInitWeigths['lowAndHigh_c1Values']
            niceFormat = "{UniformDistribution: {lowAndHigh_c1Values : { lowValue: " + str(
                lowAndHigh_c1Values[0]) + " highValue: " + str(lowAndHigh_c1Values[1]) + "},"
            lowAndHigh_c3Values = self.paramsForInitWeigths['lowAndHigh_c3Values']
            niceFormat += "lowAndHigh_c3Values : { lowValue: " + str(lowAndHigh_c3Values[0]) + " highValue: " + str(
                lowAndHigh_c3Values[1]) + "},"
            lowAndHigh_fc5Values = self.paramsForInitWeigths['lowAndHigh_fc5Values']
            niceFormat += "lowAndHigh_fc5Values : { lowValue: " + str(lowAndHigh_fc5Values[0]) + " highValue: " + str(
                lowAndHigh_fc5Values[1]) + "},"
            niceFormat += "fc6Values : All Zeros,"
            niceFormat += "fc6BiasValues : All Zeros,"
            niceFormat += "fc5BiasValues  : All Zeros}}"

        return niceFormat