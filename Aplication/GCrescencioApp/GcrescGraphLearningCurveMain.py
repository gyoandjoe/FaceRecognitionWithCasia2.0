from Core.AnalysisManager import AnalysisManager
from IGenericServices import AnalysisRepo

database_relative_path = "../BD/FR2.0.db"

idExperiment=11

ar = AnalysisRepo.AnalysisRepo(data_base=database_relative_path)

am = AnalysisManager(data_base=database_relative_path,analisys_repo=ar,id_experiment=idExperiment)

am.GraficarLearningCurve(id_experiment=idExperiment
                         ,id_weigths_1_de_4=495
                         ,id_weigths_2_de_4=427
                         ,id_weigths_3_de_4=474
                         ,id_weigths_4_de_4=328)

print("Fin")