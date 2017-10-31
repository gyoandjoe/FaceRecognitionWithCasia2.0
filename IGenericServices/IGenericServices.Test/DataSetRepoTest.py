from IGenericServices.DataSetRepo import DataSetRepo
import pandas as pd


res1 = 1//10
res2 = 9//10
listPKLs = []

referenceList = pd.read_csv(r'D:\Gyo\Dev\Thesis\FaceRecognition2.0_metadata\reference_fr2.0.csv')
for index,row in referenceList.iterrows():
    itemDict = {
        "pklFullPath": row["full_path_pkl"],
        "pklNoRpws": row["size_pkl"]
    }
    listPKLs.append(itemDict)


dsr = DataSetRepo(list_PKL_files=listPKLs,batch_size='2000',superbatch_Size='10000',no_rows=690569)

result2 = dsr.GetRawDataSetBySuperBatchIndex(super_batchIndex=0)
print ("--------" )
result3 = dsr.GetRawDataSetBySuperBatchIndex(super_batchIndex=1)
print ("--------")
result4 = dsr.GetRawDataSetBySuperBatchIndex(super_batchIndex=2)

print ("OK")

#2.3 GB x 10000 registros cargados