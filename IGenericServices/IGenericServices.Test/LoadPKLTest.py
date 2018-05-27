import _pickle as cPickle
import os
import numpy as np

fLoaded = open('D:\\Gyo\\Dev\\Thesis\\dist2\\pkls\\randTrain\\CASIA_TrainSet_Rand1GB_177.pkl',mode='rb')

data = cPickle.load(fLoaded, encoding='latin1')
fLoaded.close()

#CASIA_TrainSet_Rand1GB_177 = 538
#CASIA_TrainSet_Rand1GB_167 = 3900


#CASIA_TestSet_Rand1GB_3 = 1300
#CASIA_ValidationSet_Rand1GB_15 = 1300