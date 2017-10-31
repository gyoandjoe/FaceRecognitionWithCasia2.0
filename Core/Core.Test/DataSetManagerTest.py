
import pandas as pd

from Core.DataSetManager import DataSetManager
import numpy as np
from IGenericServices.DataSetRepo import DataSetRepo

listPKLs=[]
referenceList = pd.read_csv(r'D:\Gyo\Dev\Thesis\FaceRecognition2.0_metadata\reference_fr2.0.csv')
for index,row in referenceList.iterrows():
    itemDict = {
        "pklFullPath": row["full_path_pkl"],
        "pklNoRpws": row["size_pkl"]
    }
    listPKLs.append(itemDict)


batch_size='2000'
superbatch_Size='10000'

dsr = DataSetRepo(list_PKL_files=listPKLs,batch_size=batch_size,superbatch_Size=superbatch_Size,no_rows=690569)

randomStateReorderBatch = np.random.RandomState(1234)
seed_Random_batch_order = randomStateReorderBatch.randint(999999)
dsm = DataSetManager(batch_size=batch_size,superbatch_Size=superbatch_Size,dataset_repo=dsr)

res = dsm.LoadRandomOrderDataSetXBatch(14)
res1 = dsm.LoadRandomOrderDataSetXBatch(15)
res2 = dsm.LoadRandomOrderDataSetXBatch(16)

print ('ok')