from Arquitectures.GCrescencioArqui.InitGCresc import InitGCresc
from Arquitectures.GCrescencioArqui import MetaDataGCresc
from Core.ExperimentsManager import ExperimentsManager
from IGenericServices.WeightsRepo import WeigthsRepo

from Infra import DistTypes

class ExperimentInitializer(object):
    def __init__(self):
        return

    def CreateExperimentFromMetadata(self,dbRelativePath, experiment_metadata):
        #Creamos experimento
        ex = ExperimentsManager(dbFile=dbRelativePath)
        idNewExperiment = ex.CreateExperiment(
            initialLearningRate=experiment_metadata["initialLearningRate"],
            PKLsTrainReferenceFile=experiment_metadata["PKLsTrainReferenceFile"],
            PKLsTestReferenceFile=experiment_metadata["PKLsTestReferenceFile"],
            PKLsValReferenceFile=experiment_metadata["PKLsValReferenceFile"],
            status=experiment_metadata["status"],
            batchSize=experiment_metadata["batchSize"],
            superBatchSize=experiment_metadata["superBatchSize"],
            maxEpochTraining= experiment_metadata["maxEpochTraining"],
            FolderWeightsPath=experiment_metadata["folderWeightsPath"],
            epochFrecSaveWeights=experiment_metadata["epochFrecSaveWeights"],
            withLRDecay=experiment_metadata["withLRDecay"],
            EpochFrecLRDecay=experiment_metadata["EpochFrecLRDecay"],
            BatchActual=experiment_metadata["BatchActual"],
            ShouldDecreaseNow=experiment_metadata["ShouldDecreaseNow"],
            ShouldIncreaseNow=experiment_metadata["ShouldIncreaseNow"]
            )


        init = InitGCresc()
        ws = init.GenerateNewWeightsFromMetadata(DistTypes.DistTypes.normal, MetaDataGCresc.distributionNormalDistParams, MetaDataGCresc.layers_metaData)
        wr = WeigthsRepo(database_name=dbRelativePath, id_experiment=idNewExperiment)
        newidW = wr.SaveWeights(weights=ws,folder_path= experiment_metadata["folderWeightsPath"])
        return (idNewExperiment,newidW)
