weightsMetadata={
    "folder_weights_path":"D:\Gyo\Dev\Thesis\FaceRecognition2.0\Aplication\Weights"
}

experimentMetaData={
    "initialLearningRate": 0.1,
    "PKLsTrainReferenceFile": 'D:\\Gyo\\Dev\\Thesis\\dist2\\referenceTrainfull\\pkls_train.csv',
    "PKLsTestReferenceFile": 'D:\\Gyo\\Dev\\Thesis\\dist2\\referencetest\\pkls_test.csv',
    "PKLsValReferenceFile": 'D:\\Gyo\\Dev\\Thesis\\dist2\\referencetest\\pkls_valid.csv',
    "status": 'inicializado',
    "batchSize": 400,
    "superBatchSize": 36000,
    "noRowsTrainSet":690838,
    "maxEpochTraining": 60,
    "epochFrecSaveWeights": 1,
    "withLRDecay":0,
    "EpochFrecLRDecay":5,
    "BatchActual":0,
    "ShouldDecreaseNow":0,
    "ShouldIncreaseNow":0,
    "folderWeightsPath":"D:\Gyo\Dev\Thesis\FaceRecognition2.0\Aplication\Weights"
}


distributionUniformParams = {
    "conv1LowValue":-0.1,
    "conv1HighValue":0.1,

    "conv2LowValue":-0.1,
    "conv2HighValue":0.1,

    "conv3LowValue":-0.1,
    "conv3HighValue":0.1,

    "conv4LowValue":-0.1,
    "conv4HighValue":0.1,

    "conv5LowValue":-0.01,
    "conv5HighValue":0.01,

    "conv6LowValue":-0.01,
    "conv6HighValue":0.01,

    "conv7LowValue":-0.01,
    "conv7HighValue":0.01,

    "conv8LowValue":-0.01,
    "conv8HighValue":0.01,

    "conv9LowValue":-0.01,
    "conv9HighValue":0.01,

    "conv10LowValue":-0.01,
    "conv10HighValue":0.01,

    "fc1LowValue":-0.0001,
    "fc1HighValue":0.0001,

    "SoftMLowValue":-0.00001,
    "SoftMHighValue":0.00001,

    "FC1BiasInit":1,

    "SoftMBiasInit":1
    }

distributionNormalDistParams = {
    "conv1InitMean":0,
    "conv1InitSD":0.1,

    "conv2InitMean":0,
    "conv2InitSD":0.1,

    "conv3InitMean":0,
    "conv3InitSD":0.1,

    "conv4InitMean":0,
    "conv4InitSD":0.1,

    "conv5InitMean":0,
    "conv5InitSD":0.1,

    "conv6InitMean":0,
    "conv6InitSD":0.1,

    "conv7InitMean":0,
    "conv7InitSD":0.1,

    "conv8InitMean":0,
    "conv8InitSD":0.1,

    "conv9InitMean":0,
    "conv9InitSD":0.01,

    "conv10InitMean":0,
    "conv10InitSD":0.01,

    "fc1InitMean":0,
    "fc1InitSD":0.01,
    "FC1BiasInit":1

   #"SoftMInitMean":0,
   # "SoftMInitSD":0.0001,
   # "SoftMBiasInit":1
}

layers_metaData = {
    #Conv 1
    'Conv1_NoFiltersOut': 32,
    'Conv1_NoFiltersIn': 1,
    'Conv1_sizeKernelW': 3,
    'Conv1_sizeKernelH': 3,
    'Conv1_sizeImgInH': 100,
    'Conv1_sizeImgInW': 100,

    #Conv 2
    'Conv2_NoFiltersOut': 64,
    'Conv2_NoFiltersIn': 32,
    'Conv2_sizeKernelW': 3,
    'Conv2_sizeKernelH': 3,
    'Conv2_sizeImgInH': 100,
    'Conv2_sizeImgInW': 100,

    # Pool 1 MAX
    'Poo11_sizeKernelW': 2,
    'Poo11_sizeKernelH': 2,
    'Poo11_mode':'max',

    # Conv 3
    'Conv3_NoFiltersOut': 64,
    'Conv3_NoFiltersIn': 64,
    'Conv3_sizeKernelW': 3,
    'Conv3_sizeKernelH': 3,
    'Conv3_sizeImgInH': 50,
    'Conv3_sizeImgInW': 50,

    # Conv 4
    'Conv4_NoFiltersOut': 128,
    'Conv4_NoFiltersIn': 64,
    'Conv4_sizeKernelW': 3,
    'Conv4_sizeKernelH': 3,
    'Conv4_sizeImgInH': 50,
    'Conv4_sizeImgInW': 50,

    # pool 2 MAx
    'Poo12_sizeKernelW': 2,
    'Poo12_sizeKernelH': 2,
    'Poo12_mode':'max',

    # Conv 5
    'Conv5_NoFiltersOut': 96,
    'Conv5_NoFiltersIn': 128,
    'Conv5_sizeKernelW': 3,
    'Conv5_sizeKernelH': 3,
    'Conv5_sizeImgInH': 25,
    'Conv5_sizeImgInW': 25,

    # Conv 6
    'Conv6_NoFiltersOut': 192,
    'Conv6_NoFiltersIn': 96,
    'Conv6_sizeKernelW': 3,
    'Conv6_sizeKernelH': 3,
    'Conv6_sizeImgInH': 25,
    'Conv6_sizeImgInW': 25,

    # pool 3 Max
    'Poo13_sizeKernelW': 2,
    'Poo13_sizeKernelH': 2,
    'Poo13_mode':'max',

    # Conv 7
    'Conv7_NoFiltersOut': 128,
    'Conv7_NoFiltersIn': 192,
    'Conv7_sizeKernelW': 3,
    'Conv7_sizeKernelH': 3,
    'Conv7_sizeImgInH': 13,
    'Conv7_sizeImgInW': 13,

    # Conv 8
    'Conv8_NoFiltersOut': 256,
    'Conv8_NoFiltersIn': 128,
    'Conv8_sizeKernelW': 3,
    'Conv8_sizeKernelH': 3,
    'Conv8_sizeImgInH': 13,
    'Conv8_sizeImgInW': 13,

    # pool 4 MAX
    'Poo14_sizeKernelW': 2,
    'Poo14_sizeKernelH': 2,
    'Poo14_mode':'max',

    # Conv 9
    'Conv9_NoFiltersOut': 160,
    'Conv9_NoFiltersIn': 256,
    'Conv9_sizeKernelW': 3,
    'Conv9_sizeKernelH': 3,
    'Conv9_sizeImgInH': 7,
    'Conv9_sizeImgInW': 7,

    # Conv 10
    'Conv10_NoFiltersOut': 320,
    'Conv10_NoFiltersIn': 160,
    'Conv10_sizeKernelW': 3,
    'Conv10_sizeKernelH': 3,
    'Conv10_sizeImgInH': 7,
    'Conv10_sizeImgInW': 7,

    # pool 5 AVG
    'Poo15_sizeKernelW': 7,
    'Poo15_sizeKernelH': 7,
    'Poo15_mode':'average_exc_pad',

    # dropout
    'DO1_percent': 0.4,
    'DO1_size_in': 320,

    #FC
    'FC1_NoFiltersIn': 320, #512 * 8 * 8
    'FC1_NoFiltersOut': 10575,

    #SoftMax
    'SoftM_NoFiltersIn':10575,
    'SoftM_NoFiltersOut':10575

    #Contrastive ?????????????

}

