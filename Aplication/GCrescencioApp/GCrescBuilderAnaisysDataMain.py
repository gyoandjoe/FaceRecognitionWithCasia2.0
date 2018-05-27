from Arquitectures.GCrescencioArqui.CNNGCresc import CNNGCresc
from Arquitectures.GCrescencioArqui.ExperimentInitializer import ExperimentInitializer
from Arquitectures.GCrescencioArqui.MetaDataGCresc import layers_metaData
from Arquitectures.GCrescencioArqui.MetaDataGCresc import experimentMetaData
import theano.tensor as T
import numpy as np

#Primero debemos cargar los datos del experimento

#despues debemos cargar los pesos
from Core.AnalysisManager import AnalysisManager
from Core.DataSetManager import DataSetManager
from Core.ExperimentsManager import ExperimentsManager
from Core.Validator_ValTest import Validator_ValTest
from Core.WeightsManager import WeightsManager
from IGenericServices.ExperimentsRepo import ExperimentsRepo
from IGenericServices.LoggerRepo import LoggerRepo
from IGenericServices.WeightsRepo import WeigthsRepo

database_relative_path = "../BD/FR2.0.db"





am = AnalysisManager(data_base=database_relative_path,
                     analisys_repo=None,
                     id_experiment=24)

am.BuildWeigthsErrorAndCost_ValSet(146900,645,True)

am.BuildWeigthsErrorAndCost_TestSet(146900,645,True)

am.BuildWeigthsErrorAndCost_TrainSet(690838,645,True)

print ("OK")
"""
#Ahora cargamos los weights
wr=WeigthsRepo(database_name=database_relative_path,id_experiment=idExperiment)
wm = WeightsManager(
    database_name=database_relative_path,
    weights_repo=wr
)

iws = wm.LoadWeightsXId(idW)
random_droput = np.random.RandomState(12345)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999899))

#terminar de instanciar esto
cnn = CNNGCresc(
    layers_metaData=layers_metaData,
    initWeights=iws,
    srng = rng_droput,
    no_channels_imageInput=1,
    isTraining=0,
    pDropOut=0.7 #antes 0.60
)
#exp 9 pDropout=0.65
logger = LoggerRepo(id_experiment=idExperiment,database_name=database_relative_path)
expRepo=ExperimentsManager(database_relative_path)

validator = Validator_ValTest(
    idExperiment=idExperiment,
    logger=logger,
    cnn=cnn,
    experimentsManager=expRepo,
    weightsRepo=wr
)

costo = validator.CalculateCost()



"""