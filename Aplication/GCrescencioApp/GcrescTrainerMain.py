from Arquitectures.GCrescencioArqui.CNNGCresc import CNNGCresc
from Arquitectures.GCrescencioArqui.ExperimentInitializer import ExperimentInitializer
from Arquitectures.GCrescencioArqui.MetaDataGCresc import layers_metaData
from Arquitectures.GCrescencioArqui.MetaDataGCresc import experimentMetaData
import theano.tensor as T
import numpy as np

#Primero debemos cargar los datos del experimento

#despues debemos cargar los pesos
from Core.DataSetManager import DataSetManager
from Core.ExperimentsManager import ExperimentsManager
from Core.Trainer import Trainer
from Core.WeightsManager import WeightsManager
from IGenericServices.ExperimentsRepo import ExperimentsRepo
from IGenericServices.LoggerRepo import LoggerRepo
from IGenericServices.WeightsRepo import WeigthsRepo

database_relative_path = "../BD/FR2.0.db"


idExperiment =40
idW=1604

#Creamos experimiento con ids en caso de que no exista uno asignado, esto es cuadno idExperiment == -1
if (idExperiment == -1):
    ei = ExperimentInitializer()
    newids = ei.CreateExperimentFromMetadata(database_relative_path,experimentMetaData)
    print("OK, nuevo id de experimento: " + str(newids[0]) + " nuevo id Weights: " + str(newids[1]))
    idExperiment=newids[0]
    idW = newids[1]

#Ahora cargamos los weights
wr=WeigthsRepo(database_name=database_relative_path,id_experiment=idExperiment)
wm = WeightsManager(
    database_name=database_relative_path,
    weights_repo=wr
)

iws = wm.LoadWeightsXId(idW)
random_droput = np.random.RandomState()
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999899))

#terminar de instanciar esto
cnn = CNNGCresc(
    #batch_size=507,
    layers_metaData=layers_metaData,
    initWeights=iws,    
    srng = rng_droput,
    no_channels_imageInput=1,
    isTraining=1,
    pDropOut=0.4#Entre mas bajo mas penalizador.  Epoca 38 con 0.75, epoca 40 con 0.4                             #0.7 con experiment 24 #antes 0.60
)

#exp 9 pDropout=0.65
logger = LoggerRepo(id_experiment=idExperiment,database_name=database_relative_path)
expRepo=ExperimentsManager(database_relative_path)
noRowsTrainSet = experimentMetaData["noRowsTrainSet"]



trainer = Trainer(
    idExperiment=idExperiment,
    logger=logger,
    cnn=cnn,
    experimentsManager=expRepo,
    weightsRepo=wr,
    norows_trainset=noRowsTrainSet
)
trainer.Train(current_batch=0,current_epoch=43)


print("OK")