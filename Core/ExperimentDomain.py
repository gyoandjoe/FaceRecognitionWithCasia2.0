class ExperimentDomain(object):
    def __init__(self, pkl_validation_referenceFullpath, pkl_train_referenceFullpath,pkl_test_referenceFullpath,max_epochs, with_lr_decay, learning_rate_ratio_decay, initial_learning_rate,
                 saveWeigthsFrecuency, frecuency_lr_decay,batch_size,super_batch_size,pkl_train_referenceList, pkl_test_referenceList, pkl_validation_referenceList, folderWeightsPath):
        self.max_epochs = max_epochs
        self.with_lr_decay = with_lr_decay

        self.pkl_train_referenceFullpath = pkl_train_referenceFullpath
        self.pkl_validation_referenceFullpath = pkl_validation_referenceFullpath
        self.pkl_test_referenceFullpath = pkl_test_referenceFullpath

        self.pkl_train_referenceList = pkl_train_referenceList
        self.pkl_test_referenceList = pkl_test_referenceList
        self.pkl_validation_referenceList = pkl_validation_referenceList

        self.learning_rate_ratio_decay = learning_rate_ratio_decay
        self.learning_rate = float(initial_learning_rate)
        self.saveWeigthsFrecuency = saveWeigthsFrecuency
        self.frecuency_lr_decay = frecuency_lr_decay
        self.batch_size = batch_size
        self.super_batch_size = super_batch_size
        self.FolderWeightsPath = folderWeightsPath
        return
