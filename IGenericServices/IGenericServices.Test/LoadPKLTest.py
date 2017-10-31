import _pickle as cPickle
import os
import numpy as np

fLoaded = open('D:\\Gyo\\Dev\\Thesis\\Distribute and random_1gb\\randTrain\\CASIA_TrainSet_Rand1GB_177.pkl',mode='rb')

data = cPickle.load(fLoaded, encoding='latin1')
fLoaded.close()

