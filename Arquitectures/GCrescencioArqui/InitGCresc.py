from Infra import DistTypes, WeightsGenerator
import theano
import numpy as np

class InitGCresc(object):

    def GenerateNewWeightsFromMetadata(self, distributionType, distributionParams, layers_metaData):
        conv1_filterShape = (layers_metaData['Conv1_NoFiltersOut'], layers_metaData['Conv1_NoFiltersIn'],
                             layers_metaData['Conv1_sizeKernelW'], layers_metaData['Conv1_sizeKernelH'])

        conv2_filterShape = (layers_metaData['Conv2_NoFiltersOut'], layers_metaData['Conv2_NoFiltersIn'],
                             layers_metaData['Conv2_sizeKernelW'], layers_metaData['Conv2_sizeKernelH'])

        conv3_filterShape = (layers_metaData['Conv3_NoFiltersOut'], layers_metaData['Conv3_NoFiltersIn'],
                             layers_metaData['Conv3_sizeKernelW'], layers_metaData['Conv3_sizeKernelH'])

        conv4_filterShape = (layers_metaData['Conv4_NoFiltersOut'], layers_metaData['Conv4_NoFiltersIn'],
                             layers_metaData['Conv4_sizeKernelW'], layers_metaData['Conv4_sizeKernelH'])

        conv5_filterShape = (layers_metaData['Conv5_NoFiltersOut'], layers_metaData['Conv5_NoFiltersIn'],
                             layers_metaData['Conv5_sizeKernelW'], layers_metaData['Conv5_sizeKernelH'])

        conv6_filterShape = (layers_metaData['Conv6_NoFiltersOut'], layers_metaData['Conv6_NoFiltersIn'],
                             layers_metaData['Conv6_sizeKernelW'], layers_metaData['Conv6_sizeKernelH'])

        conv7_filterShape = (layers_metaData['Conv7_NoFiltersOut'], layers_metaData['Conv7_NoFiltersIn'],
                             layers_metaData['Conv7_sizeKernelW'], layers_metaData['Conv7_sizeKernelH'])

        conv8_filterShape = (layers_metaData['Conv8_NoFiltersOut'], layers_metaData['Conv8_NoFiltersIn'],
                             layers_metaData['Conv8_sizeKernelW'], layers_metaData['Conv8_sizeKernelH'])

        conv9_filterShape = (layers_metaData['Conv9_NoFiltersOut'], layers_metaData['Conv9_NoFiltersIn'],
                             layers_metaData['Conv9_sizeKernelW'], layers_metaData['Conv9_sizeKernelH'])

        conv10_filterShape = (layers_metaData['Conv10_NoFiltersOut'], layers_metaData['Conv10_NoFiltersIn'],
                              layers_metaData['Conv10_sizeKernelW'], layers_metaData['Conv10_sizeKernelH'])

        FC1_filterShape = (layers_metaData["FC1_NoFiltersIn"] ,layers_metaData["FC1_NoFiltersOut"])

        FC1Bias_filterShape = ((layers_metaData["FC1_NoFiltersOut"]))

        SoftM_filterShape = (layers_metaData['SoftM_NoFiltersIn'], layers_metaData['SoftM_NoFiltersOut'])

        SoftMBias_filterShape = ((layers_metaData['SoftM_NoFiltersOut']))


        if distributionType is DistTypes.DistTypes.uniform:
            ws = self.GetUniformDistributionWeithts(distributionParams,conv1_filterShape,conv2_filterShape,conv3_filterShape,conv4_filterShape,conv5_filterShape,conv6_filterShape,conv7_filterShape,conv8_filterShape,conv9_filterShape,conv10_filterShape,FC1_filterShape,FC1Bias_filterShape,SoftM_filterShape,SoftMBias_filterShape)
        elif (distributionType is DistTypes.DistTypes.normal):
            ws = self.GetNormalDistributionWeights(distributionParams,conv1_filterShape,conv2_filterShape,conv3_filterShape,conv4_filterShape,conv5_filterShape,conv6_filterShape,conv7_filterShape,conv8_filterShape,conv9_filterShape,conv10_filterShape,FC1_filterShape,FC1Bias_filterShape,SoftM_filterShape,SoftMBias_filterShape)

        print("conv1" + str(ws["conv1Values"].shape))
        print("conv2" + str(ws["conv2Values"].shape))
        print("conv3" + str(ws["conv3Values"].shape))
        print("conv4" + str(ws["conv4Values"].shape))
        print("conv5" + str(ws["conv5Values"].shape))
        print("conv6" + str(ws["conv6Values"].shape))
        print("conv7" + str(ws["conv7Values"].shape))
        print("conv8" + str(ws["conv8Values"].shape))
        print("conv9" + str(ws["conv9Values"].shape))
        print("conv10" + str(ws["conv10Values"].shape))

        print("FC1" + str(ws["FC1Values"].shape))
        print("FC1Bias" + str(ws["FC1BiasValues"].shape))

        #print("SoftMax1" + str(ws["SoftMax1Values"].shape))
        #print("SoftMax1Bias" + str(ws["SoftMax1BiasValues"].shape))
        return ws

    def GetUniformDistributionWeithts(self,distributionParams,conv1_filterShape,conv2_filterShape,conv3_filterShape,conv4_filterShape,conv5_filterShape,conv6_filterShape,conv7_filterShape,conv8_filterShape,conv9_filterShape,conv10_filterShape,FC1_filterShape,FC1Bias_filterShape,SoftM_filterShape,SoftMBias_filterShape):
        conv1Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv1_filterShape,
                                                                                            distributionParams[
                                                                                                "conv1LowValue"],
                                                                                            distributionParams[
                                                                                                "conv1HighValue"])

        conv2Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv2_filterShape,
                                                                                            distributionParams[
                                                                                                "conv2LowValue"],
                                                                                            distributionParams[
                                                                                                "conv2HighValue"])

        conv3Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv3_filterShape,
                                                                                            distributionParams[
                                                                                                "conv3LowValue"],
                                                                                            distributionParams[
                                                                                                "conv3HighValue"])

        conv4Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv4_filterShape,
                                                                                            distributionParams[
                                                                                                "conv4LowValue"],
                                                                                            distributionParams[
                                                                                                "conv4HighValue"])

        conv5Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv5_filterShape,
                                                                                            distributionParams[
                                                                                                "conv5LowValue"],
                                                                                            distributionParams[
                                                                                                "conv5HighValue"])

        conv6Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv6_filterShape,
                                                                                            distributionParams[
                                                                                                "conv6LowValue"],
                                                                                            distributionParams[
                                                                                                "conv6HighValue"])

        conv7Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv7_filterShape,
                                                                                            distributionParams[
                                                                                                "conv7LowValue"],
                                                                                            distributionParams[
                                                                                                "conv7HighValue"])

        conv8Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv8_filterShape,
                                                                                            distributionParams[
                                                                                                "conv8LowValue"],
                                                                                            distributionParams[
                                                                                                "conv8HighValue"])

        conv9Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv9_filterShape,
                                                                                            distributionParams[
                                                                                                "conv9LowValue"],
                                                                                            distributionParams[
                                                                                                "conv9HighValue"])

        conv10Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv10_filterShape,
                                                                                             distributionParams[
                                                                                                 "conv10LowValue"],
                                                                                             distributionParams[
                                                                                                 "conv10HighValue"])

        fc1Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(FC1_filterShape,
                                                                                          distributionParams[
                                                                                              "fc1LowValue"],
                                                                                          distributionParams[
                                                                                              "fc1HighValue"])

        if (distributionParams["FC1BiasInit"] == 1):
            fc1BiasValues = np.ones(FC1Bias_filterShape, dtype=theano.config.floatX)
            print("Ones in fc1BiasValues")
        else:
            print("Zeros in fc1BiasValues")
            fc1BiasValues = np.zeros(FC1Bias_filterShape, dtype=theano.config.floatX)

        fc2Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(FC2_filterShape,
                                                                                          distributionParams[
                                                                                              "fc2LowValue"],
                                                                                          distributionParams[
                                                                                              "fc2HighValue"])

        SoftMValues = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(SoftM_filterShape,
                                                                                            distributionParams[
                                                                                                "SoftMLowValue"],
                                                                                            distributionParams[
                                                                                                "SoftMHighValue"])
        if (distributionParams["SoftMBiasInit"] == 1):
            SoftMBiasValues = np.ones(SoftMBias_filterShape, dtype=theano.config.floatX)
            print("Ones in SoftMBiasValues")
        else:
            print("Zeros in SoftMBiasValues")
            SoftMBiasValues = np.zeros(SoftMBias_filterShape, dtype=theano.config.floatX)


        initial_weights = {
            "conv1Values": np.asarray(conv1Values, dtype=theano.config.floatX),
            "conv2Values": np.asarray(conv2Values, dtype=theano.config.floatX),
            "conv3Values": np.asarray(conv3Values, dtype=theano.config.floatX),
            "conv4Values": np.asarray(conv4Values, dtype=theano.config.floatX),
            "conv5Values": np.asarray(conv5Values, dtype=theano.config.floatX),
            "conv6Values": np.asarray(conv6Values, dtype=theano.config.floatX),
            "conv7Values": np.asarray(conv7Values, dtype=theano.config.floatX),
            "conv8Values": np.asarray(conv8Values, dtype=theano.config.floatX),
            "conv9Values": np.asarray(conv9Values, dtype=theano.config.floatX),
            "conv10Values": np.asarray(conv10Values, dtype=theano.config.floatX),

            "FC1Values": np.asarray(fc1Values, dtype=theano.config.floatX),
            "FC1BiasValues": np.asarray(fc1BiasValues, dtype=theano.config.floatX)

            #""SoftMax1Values": np.asarray(SoftMValues, dtype=theano.config.floatX),
            #"SoftMax1BiasValues": np.asarray(SoftMBiasValues, dtype=theano.config.floatX)
        }

        return initial_weights

    def GetNormalDistributionWeights(self,distributionParams,conv1_filterShape,conv2_filterShape,conv3_filterShape,conv4_filterShape,conv5_filterShape,conv6_filterShape,conv7_filterShape,conv8_filterShape,conv9_filterShape,conv10_filterShape,FC1_filterShape,FC1Bias_filterShape,SoftM_filterShape,SoftMBias_filterShape):

        conv1Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv1_filterShape,
                                                                                           distributionParams["conv1InitMean"],
                                                                                           distributionParams["conv1InitSD"])

        conv2Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv2_filterShape,
                                                                                           distributionParams["conv2InitMean"],
                                                                                           distributionParams["conv2InitSD"])

        conv3Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv3_filterShape,
                                                                                           distributionParams["conv3InitMean"],
                                                                                           distributionParams["conv3InitSD"])

        conv4Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv4_filterShape,
                                                                                           distributionParams[
                                                                                               "conv4InitMean"],
                                                                                           distributionParams[
                                                                                               "conv4InitSD"])

        conv5Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv5_filterShape,
                                                                                           distributionParams[
                                                                                               "conv5InitMean"],
                                                                                           distributionParams[
                                                                                               "conv5InitSD"])

        conv6Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv6_filterShape,
                                                                                           distributionParams[
                                                                                               "conv6InitMean"],
                                                                                           distributionParams[
                                                                                               "conv6InitSD"])

        conv7Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv7_filterShape,
                                                                                           distributionParams[
                                                                                               "conv7InitMean"],
                                                                                           distributionParams[
                                                                                               "conv7InitSD"])

        conv8Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv8_filterShape,
                                                                                           distributionParams[
                                                                                               "conv8InitMean"],
                                                                                           distributionParams[
                                                                                               "conv8InitSD"])

        conv9Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv9_filterShape,
                                                                                           distributionParams[
                                                                                               "conv9InitMean"],
                                                                                           distributionParams[
                                                                                               "conv9InitSD"])

        conv10Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv10_filterShape,
                                                                                           distributionParams[
                                                                                               "conv10InitMean"],
                                                                                           distributionParams[
                                                                                               "conv10InitSD"])

        fc1Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(FC1_filterShape,
                                                                                         distributionParams["fc1InitMean"],
                                                                                         distributionParams["fc1InitSD"])
        if (distributionParams["FC1BiasInit"] == 1):
            fc1BiasValues = np.ones(FC1Bias_filterShape, dtype=theano.config.floatX)
            print("Ones en fc1BiasValues")
        else:
            fc1BiasValues = np.zeros(FC1Bias_filterShape, dtype=theano.config.floatX)
            print("Zeros en fc1BiasValues")




        #SoftMValues = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(SoftM_filterShape,
        #                                                                                   distributionParams[
        #                                                                                       "SoftMInitMean"],
        #                                                                                   distributionParams[
        #                                                                                       "SoftMInitSD"])

        #if (distributionParams["SoftMBiasInit"] == 1):
        #    SoftMBiasValues = np.ones(SoftMBias_filterShape, dtype=theano.config.floatX)
        #    print("Ones en SoftMBiasValues")
        #else:
        #    print("Zeros en SoftMBiasValues")
        #    SoftMBiasValues = np.zeros(SoftMBias_filterShape, dtype=theano.config.floatX)

        initial_weights = {
                "conv1Values": np.asarray(conv1Values, dtype=theano.config.floatX),
                "conv2Values": np.asarray(conv2Values, dtype=theano.config.floatX),
                "conv3Values": np.asarray(conv3Values, dtype=theano.config.floatX),
                "conv4Values": np.asarray(conv4Values, dtype=theano.config.floatX),
                "conv5Values": np.asarray(conv5Values, dtype=theano.config.floatX),
                "conv6Values": np.asarray(conv6Values, dtype=theano.config.floatX),
                "conv7Values": np.asarray(conv7Values, dtype=theano.config.floatX),
                "conv8Values": np.asarray(conv8Values, dtype=theano.config.floatX),
                "conv9Values": np.asarray(conv9Values, dtype=theano.config.floatX),
                "conv10Values": np.asarray(conv10Values, dtype=theano.config.floatX),


                "FC1Values": np.asarray(fc1Values, dtype=theano.config.floatX),
                "FC1BiasValues": np.asarray(fc1BiasValues, dtype=theano.config.floatX)


                #"SoftMax1Values": np.asarray(SoftMValues, dtype=theano.config.floatX),
                #"SoftMax1BiasValues": np.asarray(SoftMBiasValues, dtype=theano.config.floatX)
            }

        return initial_weights

    def SaveWeighsToPKL(self, weights):

        return