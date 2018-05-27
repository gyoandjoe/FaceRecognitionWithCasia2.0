import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\Gyo\Dev\Thesis\dist2\trainSet_rand.csv', header=None, usecols=[0])

df.convert_objects(convert_numeric=True)
df[0] = df[0].astype(np.float64)
#grouped = df.groupby(0)

#xx=grouped.groups.keys()
x = np.asarray(list(df[0]), dtype=int)
#yy=grouped[0].count()
#x = np.asarray(list(xx), dtype=int)
n, bins, patches = plt.hist(x,bins=10575, facecolor='g')

plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histogram of IQ')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([0, 10575, 0, 1500])
plt.grid(True)
plt.show()

#y = np.asarray(yy, dtype=np.float64)
#plt.cla()
#plt.plot(x,y,'-')
#plt.show()