from Arquitectures.GCrescencioArqui.CNNGCresc import CNNGCresc
from Arquitectures.GCrescencioArqui.ExperimentInitializer import ExperimentInitializer
from Arquitectures.GCrescencioArqui.MetaDataGCresc import layers_metaData
from Arquitectures.GCrescencioArqui.MetaDataGCresc import experimentMetaData
import theano.tensor as T
import numpy as np
import theano
#Primero debemos cargar los datos del experimento

#despues debemos cargar los pesos
from Core.DataSetManager import DataSetManager
from Core.ExperimentsManager import ExperimentsManager
from Core.Predictor import Predictor
from Core.Trainer import Trainer
from Core.WeightsManager import WeightsManager
from IGenericServices.ExperimentsRepo import ExperimentsRepo
from IGenericServices.LoggerRepo import LoggerRepo
from IGenericServices.WeightsRepo import WeigthsRepo
from PIL import Image

database_relative_path = "../BD/FR2.0.db"


idExperiment = 10
idW=265


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
    isTraining=1,
    pDropOut=0.7
)

expRepo=ExperimentsManager(database_relative_path)



imagePath = 'D:\\tmp\\S2.PNG'

finalImg = np.asarray(Image.open(r'D:\Gyo\Documents\Developer-g\Projects\Thesis\casia\webface\100\0000168\002-l.jpg'), dtype=np.float32) / 256

finalImg= finalImg.reshape((1, 1, 100, 100))

imgShared = theano.shared(finalImg)


trainer = Predictor(
    cnn=cnn,
    images=imgShared
)

result = trainer.Predict()

print("OK")