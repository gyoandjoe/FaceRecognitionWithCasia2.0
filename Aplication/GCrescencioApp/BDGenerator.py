__author__ = 'win-g'

import sqlite3



conn = sqlite3.connect('..\BD\FR2.0.db')

c = conn.cursor()


c.execute('''CREATE TABLE Experiments
             (Id INTEGER PRIMARY KEY,
             InitialLearningRate real, 
             PKLsTrainReferenceFile text, 
             PKLsTestReferenceFile text,
             Status text, 
             BatchSize INTEGER,    
             SuperBatchSize INTEGER,
             MaxEpoch INTEGER,
             EpochFrecSaveWeights INTEGER,
             WithLRDecay INTEGER, 
             EpochFrecLRDecay INTEGER,
             BatchActual INTEGER,
             ShouldDecreaseNow INTEGER,
             ShouldIncreaseNow INTEGER,
             FolderWeightsPath text
             )''')

c.execute('''CREATE TABLE Weights
             (Id INTEGER PRIMARY KEY,
             IdExperiment integer, 
             PKLFullPah text,             
             FechaRegistro text, 
             Epoch integer, 
             BatchIndex integer, 
             SuperBatchIndex integer,
             Iteracion integer,                                                       
             TrainError real,
             TrainCost real,
             ValidCost real,
             ValidError real,
             TestCost real,
             TestError real)''')




c.execute('''CREATE TABLE LogExperiment
             (Id INTEGER PRIMARY KEY, 
             IdExperiment integer, 
             FechaRegistro text, 
             Contenido text,  
             TipoLog text, 
             EpochIndex integer, 
             BatchIndex integer,
             SuperBatchIndex integer,
             InfoExtra text,
             Referencia text)''')

c.execute('''CREATE TABLE LogTraining
             (Id INTEGER PRIMARY KEY, 
             IdExperiment integer, 
             FechaRegistro text, 
             Costo REAL,  
             TipoLog text, 
             EpochIndex integer, 
             BatchIndex integer,
             SuperBatchIndex integer,
             LearningRate REAL,
             Contenido text
             )''')


c.execute('''CREATE TABLE LearningCurveAnalysisXNoExp (Id INTEGER PRIMARY KEY, IdExperiment integer, IdWeigths INTEGER)''')

c.execute('''CREATE TABLE LearningCurveXNoExamp (Id INTEGER PRIMARY KEY,
NoExperiments INTEGER, 
Cost REAL, 
Error REAL, 
TipoDataSet TEXT, 
DataSetSize INTEGER,
IdLearningCurveAnalysis integer)''')


#Agregar intentoId, batch, iteracion

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()


