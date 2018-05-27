from Core.AnalysisManager import AnalysisManager
from IGenericServices import AnalysisRepo

database_relative_path = "../BD/FR2.0.db"

idExperiment=11

ar = AnalysisRepo.AnalysisRepo(data_base=database_relative_path)

am = AnalysisManager(data_base=database_relative_path,analisys_repo=ar,id_experiment=idExperiment)

am.GraficarCostosXEpocaXDataSet(id_experiment=idExperiment)

print("Fin")