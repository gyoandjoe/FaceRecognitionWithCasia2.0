from Core.AnalysisManager import AnalysisManager
from IGenericServices import AnalysisRepo

database_relative_path = "../BD/FR2.0.db"

idExperiment=9
ar = AnalysisRepo.AnalysisRepo(data_base=database_relative_path)

am = AnalysisManager(data_base=database_relative_path,analisys_repo=ar,id_experiment=idExperiment)

am.AnalizarInRealTIme(6,0)

print("Fin finalizacion analisys rapido")