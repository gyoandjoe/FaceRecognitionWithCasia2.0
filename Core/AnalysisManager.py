import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




class AnalysisManager(object):
    def __init__(self, data_base, analisys_repo,id_experiment):
        self.analisys_repo = analisys_repo #Analisys_Repo.AnalisysRepo(data_base=data_base )
        self.id_experiment = id_experiment

    def AnalizarInRealTIme(self, velocity_update=1, start_epoch=0):
        #plt.axis([0, 10, 0, 1])
        plt.ion()

        while True:
            registros = self.analisys_repo.ObtenerLogsTraining(self.id_experiment,start_epoch)
            df = pd.DataFrame(registros, columns=['Id','IdExperiment','FechaRegistro','Costo','TipoLog','EpochIndex','BatchIndex','SuperBatchIndex','LearningRate','Contenido']) #,dtype=[('Contenido', np.float64)]
            df.convert_objects(convert_numeric=True)
            df['Costo'] = df['Costo'].astype(np.float64)
            grouped = df.groupby('EpochIndex')
            #for registro in df['contenido'].values:
            #    print registro
            xx=grouped.groups.keys()

            yy=grouped['Costo'].mean().values

            x = np.asarray(list(xx), dtype=int)
            y = np.asarray(yy, dtype=np.float64)
            plt.cla()
            plt.plot(x,y,'-')
            #print y
            #plt.show()


            plt.pause(velocity_update)



