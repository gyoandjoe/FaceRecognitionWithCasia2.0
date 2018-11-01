from sklearn.model_selection import train_test_split
import numpy as np
import csv
import pandas as pd


def PersistToFile(targetName,data):
    csvfile =  open(targetName, 'a',newline='')
    csvwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONE)
    for item in data:
        csvwriter.writerow([item[0],item[1],item[2]])


X, y = np.arange(10).reshape((5, 2)), range(5)

#dfFull = pd.read_csv(r'D:\Gyo\Dev\Thesis\dist2\analisys\fullDataSet_rand - analisys.csv', usecols=[0,1,2])
dataSetBatchListFullFromFile = np.loadtxt( r'D:\Gyo\Dev\Thesis\dist2\analisys\fullDataSet_rand - analisys.csv',delimiter=',', dtype='str',skiprows =1)


#Primero eliminamos las clases que tengan menos de 18 muestras
classesToDelete = pd.DataFrame(columns=[0],dtype=str)
dataSetBatchListFull = pd.DataFrame(data=dataSetBatchListFullFromFile)
gdataSetBatchListFull = dataSetBatchListFull.groupby([0])
cont=0
for a in gdataSetBatchListFull:
        (key,vTrain) = a
        gfull = gdataSetBatchListFull.get_group(key)

        sgVal='0'
        if key in gdataSetBatchListFull.groups:
            gVal = gdataSetBatchListFull.get_group(key)
            if ((gVal.count()[1]) < 18):
                #classesToDelete.append(key, ignore_index=True)
                classesToDelete.loc[cont] = key
                cont=cont+1


filteredDataSet = dataSetBatchListFull.loc[~dataSetBatchListFull[0].isin(np.asarray(classesToDelete.values).reshape(cont) )]




testValidationSet,trainSet = train_test_split( filteredDataSet
                                                    , test_size=0.7
                                                    ,train_size=0.3
                                                    , random_state=4
                                                    ,stratify=np.asarray(filteredDataSet.values[:,0],dtype=int))


validationSet, testSet = train_test_split(testValidationSet
                                                    , test_size=0.5
                                                    ,train_size=0.5
                                                    , random_state=4
                                                    ,stratify=np.asarray(testValidationSet.values[:,0],dtype=int))

PersistToFile(r"D:\Gyo\Dev\Thesis\dist3\trainSet_rand.csv",trainSet.values)
PersistToFile(r"D:\Gyo\Dev\Thesis\dist3\validationSet_rand.csv",validationSet.values)
PersistToFile(r"D:\Gyo\Dev\Thesis\dist3\testSet_rand.csv",testSet.values)


pdTestValSet = pd.DataFrame(data=testValidationSet)
gpdTestValSet = pdTestValSet.groupby([0])

csvfile = open(r'D:\Gyo\Dev\Thesis\dist3\analisys\DataSet_analisys_distval4.csv', 'w',newline='')
csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)

for a in gpdTestValSet:
        (key,vTrain) = a
        gfull = gpdTestValSet.get_group(key)

        sgVal='0'
        if key in gpdTestValSet.groups:
                gVal = gpdTestValSet.get_group(key)
                sgVal = str(gVal.count()[1])

        csvwriter.writerow([str(key),sgVal ])





print("Ok")


