import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv


dfTrain = pd.read_csv(r'D:\Gyo\Dev\Thesis\dist2\analisys\trainSet_rand.csv', usecols=[0])
dfVal = pd.read_csv(r'D:\Gyo\Dev\Thesis\dist2\analisys\validationSet_rand.csv', usecols=[0])
dfTest = pd.read_csv(r'D:\Gyo\Dev\Thesis\dist2\analisys\testSet_rand.csv', usecols=[0])
dfFull = pd.read_csv(r'D:\Gyo\Dev\Thesis\dist2\analisys\fullDataSet_rand - analisys.csv', usecols=[0])

dfTrain.convert_objects(convert_numeric=True)
dfTrain[0] = dfTrain['clase'].astype(np.float64)
groupedTrain = dfTrain.groupby(['clase'])

dfVal.convert_objects(convert_numeric=True)
dfVal[0] = dfVal['clase'].astype(np.float64)
groupedVal = dfVal.groupby(['clase'])

dfTest.convert_objects(convert_numeric=True)
dfTest[0] = dfTest['clase'].astype(np.float64)
groupedTest = dfTest.groupby(['clase'])

dfFull.convert_objects(convert_numeric=True)
dfFull[0] = dfFull['clase'].astype(np.float64)
groupedFull = dfFull.groupby(['clase'])

#with open(r'D:\Gyo\Dev\Thesis\dist2\analisys\DataSet_analisys.csv', 'w') as target:
csvfile = open(r'D:\Gyo\Dev\Thesis\dist2\analisys\DataSet_analisys.csv', 'w',newline='')
csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)



for a in groupedTrain:
        (key,vTrain) = a
        gfull = groupedFull.get_group(key)

        sgVal='0'
        if key in groupedVal.groups:
                gVal = groupedVal.get_group(key)
                sgVal = str(gVal.count()[1])

        sgTest = '0'
        if key in groupedTest.groups:
                gTest = groupedTest.get_group(key)
                sgTest = str(gTest.count()[1])



        svTrain = str(vTrain.count()[1])
        sgFull =  str(gfull.count()[1])


        porcentTrain = (vTrain.count()[1] * 100) / gfull.count()[1]


        csvwriter.writerow([str(key),svTrain,sgVal,sgTest,sgFull,str(porcentTrain) ])
        #target.write(str(key)+","+str(val.count()[1])+","+str(gfull.count()[1]))








#xx=grouped.groups.keys()
#x = np.asarray(list(df[0]), dtype=int)
#yy=grouped[0].count()
#x = np.asarray(list(xx), dtype=int)n, bins, patches = plt.hist(x,bins=10575, facecolor='g')}

#plt.xlabel('Clases')
#plt.ylabel('Frecuencia')
#plt.title('Histogram of IQ')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([0, 10575, 0, 1500])
#plt.grid(True)
#plt.show()

#y = np.asarray(yy, dtype=np.float64)
#plt.cla()
#plt.plot(x,y,'-')
#plt.show()

print("OK")

