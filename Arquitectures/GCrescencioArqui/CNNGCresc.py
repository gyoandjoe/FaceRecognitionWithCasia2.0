import theano
import theano.tensor as T
from theano.tensor.signal import pool
import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
from Infra import ConvLayer, WeightsGenerator, DistTypes, FCLayer, SoftMaxLayer
from Infra import Utiles
from Infra import DropOutLayer


class CNNGCresc(object):
    def __init__(self, layers_metaData, initWeights, srng,no_channels_imageInput=1,isTraining=1,pDropOut=0.5):
        """
        :param image_input: size: 64x64, channel: 1
        :param conv1_noFilters:
        :return:
        """

        self.image_input = T.tensor4('ximgin')  # the data is presented as rasterized images
        self.y = T.ivector('y')
        self.batch_size_dynamyc=None
        self.batch_size =T.lscalar()
        layers_metaData['Conv1_NoFiltersIn'] = no_channels_imageInput
        """
        convolution 1
        kernel: 3x3,
        channel out: 128
        Channel input: 1
        sizeImage input: 64
        ReLU Function
        Out Size: ((64-3 + 1*2) / 1) + 1 = 64
        ((W?F+2P)/S)+1
        W = Input volume size = 64
        F = Filter shape = 3
        S = stride = 1
        P = Padding = 1

        outputShape = batchsize x channel out x sizeImage input x sizeImage input = 10x128x64x64
        """

        """
        convolution 1 
        ReLU
        """
        conv1_filterShape = (layers_metaData['Conv1_NoFiltersOut'], layers_metaData['Conv1_NoFiltersIn'],
                             layers_metaData['Conv1_sizeKernelW'], layers_metaData['Conv1_sizeKernelH'])
        c1Values = initWeights['conv1Values']
        c1ImageShape = (self.batch_size_dynamyc, no_channels_imageInput, layers_metaData['Conv1_sizeImgInH'], layers_metaData['Conv1_sizeImgInW'])
        self.conv1 = ConvLayer.ConvLayer('conv1Layer', self.image_input, c1Values, conv1_filterShape, c1ImageShape)
        self.conv_relu_1 = Utiles.Relu(self.conv1.Out)

        GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        freeGPUMemInGBs = GPUFreeMemoryInBytes / 1024. / 1024 / 1024
        print("Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs))

        """
        convolution 2
        ReLU
        """
        conv2_filterShape = (layers_metaData['Conv2_NoFiltersOut'], layers_metaData['Conv2_NoFiltersIn'],
                             layers_metaData['Conv2_sizeKernelW'], layers_metaData['Conv2_sizeKernelH'])
        c2Values = initWeights['conv2Values']
        c2ImageShape = (self.batch_size_dynamyc, layers_metaData['Conv1_NoFiltersOut'], layers_metaData['Conv2_sizeImgInH'], layers_metaData['Conv2_sizeImgInW'])
        self.conv2 = ConvLayer.ConvLayer('conv2Layer', self.conv_relu_1, c2Values, conv2_filterShape, c2ImageShape)
        self.conv_relu_2 = Utiles.Relu(self.conv2.Out)

        """
        Pool Layer 1
        OutShape: (N, 32, 50, 50)
        """
        self.MaxPool_1 = pool.pool_2d(
            input=self.conv_relu_2,
            stride=(2, 2), #stride
            ws =(layers_metaData['Poo11_sizeKernelW'], layers_metaData['Poo11_sizeKernelH']),
            mode=layers_metaData['Poo11_mode'],
            ignore_border=True
        )

        """
        convolution 3
        outShape: (N, 64, 50, 50)
        ReLU
        """
        conv3_filterShape = (layers_metaData['Conv3_NoFiltersOut'], layers_metaData['Conv3_NoFiltersIn'],
                             layers_metaData['Conv3_sizeKernelW'], layers_metaData['Conv3_sizeKernelH'])
        c3Values = initWeights['conv3Values']
        c3ImageShape = (self.batch_size_dynamyc, layers_metaData['Conv2_NoFiltersOut'], layers_metaData['Conv3_sizeImgInH'], layers_metaData['Conv3_sizeImgInW'])
        self.conv3 = ConvLayer.ConvLayer('conv3ayer', self.MaxPool_1, c3Values, conv3_filterShape, c3ImageShape)
        self.conv_relu_3 = Utiles.Relu(self.conv3.Out)

        #GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        #freeGPUMemInGBs = GPUFreeMemoryInBytes / 1024. / 1024 / 1024
        #print("Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs))

        """
        convolution 4
        ReLU
        outShape: (N, 128, 50, 50)
        """
        conv4_filterShape = (layers_metaData['Conv4_NoFiltersOut'], layers_metaData['Conv4_NoFiltersIn'],
                             layers_metaData['Conv4_sizeKernelW'], layers_metaData['Conv4_sizeKernelH'])
        c4Values = initWeights['conv4Values']
        c4ImageShape = (self.batch_size_dynamyc, layers_metaData['Conv3_NoFiltersOut'], layers_metaData['Conv4_sizeImgInH'],
                        layers_metaData['Conv4_sizeImgInW'])
        self.conv4 = ConvLayer.ConvLayer('conv4Layer', self.conv_relu_3, c4Values, conv4_filterShape, c4ImageShape)
        self.conv_relu_4 = Utiles.Relu(self.conv4.Out)

        """
        Pool Layer  2
        """
        self.MaxPool_2 = pool.pool_2d(
            input=self.conv_relu_4,
            stride=(2, 2),  # stride
            ws=(layers_metaData['Poo12_sizeKernelW'], layers_metaData['Poo12_sizeKernelH']),
            mode=layers_metaData['Poo12_mode'],
            ignore_border=True
        )

        """
            convolution 5
            ReLU
            """
        conv5_filterShape = (layers_metaData['Conv5_NoFiltersOut'], layers_metaData['Conv5_NoFiltersIn'],
                             layers_metaData['Conv5_sizeKernelW'], layers_metaData['Conv5_sizeKernelH'])
        c5Values = initWeights['conv5Values']
        c5ImageShape = (
            self.batch_size_dynamyc, layers_metaData['Conv4_NoFiltersOut'], layers_metaData['Conv5_sizeImgInH'], layers_metaData['Conv5_sizeImgInW'])
        self.conv5 = ConvLayer.ConvLayer('conv5Layer', self.MaxPool_2, c5Values, conv5_filterShape, c5ImageShape)
        self.conv_relu_5 = Utiles.Relu(self.conv5.Out)

        #GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        #freeGPUMemInGBs = GPUFreeMemoryInBytes / 1024. / 1024 / 1024
        #print("Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs))

        """
        convolution 6
        ReLU
        outShape (N, 192, 25, 25)
        """
        conv6_filterShape = (layers_metaData['Conv6_NoFiltersOut'], layers_metaData['Conv6_NoFiltersIn'],
                             layers_metaData['Conv6_sizeKernelW'], layers_metaData['Conv6_sizeKernelH'])
        c6Values = initWeights['conv6Values']
        c6ImageShape = (self.batch_size_dynamyc, layers_metaData['Conv5_NoFiltersOut'], layers_metaData['Conv6_sizeImgInH'],
                        layers_metaData['Conv6_sizeImgInW'])
        self.conv6 = ConvLayer.ConvLayer('conv6Layer', self.conv_relu_5, c6Values, conv6_filterShape, c6ImageShape)
        self.conv_relu_6 = Utiles.Relu(self.conv6.Out)

        """
        Pool Layer 3
        outshape: (N, 192, 12, 12)
        """
        self.MaxPool_3 = pool.pool_2d(
            input=self.conv_relu_6,
            stride=(2, 2),  # stride
            pad=(1,1),
            ws=(layers_metaData['Poo13_sizeKernelW'], layers_metaData['Poo13_sizeKernelH']),
            mode=layers_metaData['Poo13_mode'],
            ignore_border=True
        )

        """
            convolution 7
            ReLU
            """
        conv7_filterShape = (layers_metaData['Conv7_NoFiltersOut'], layers_metaData['Conv7_NoFiltersIn'],
                             layers_metaData['Conv7_sizeKernelW'], layers_metaData['Conv7_sizeKernelH'])
        c7Values = initWeights['conv7Values']
        c7ImageShape = (
            self.batch_size_dynamyc, layers_metaData['Conv6_NoFiltersOut'], layers_metaData['Conv7_sizeImgInH'], layers_metaData['Conv7_sizeImgInW'])
        self.conv7 = ConvLayer.ConvLayer('conv7Layer', self.MaxPool_3, c7Values, conv7_filterShape, c7ImageShape)
        self.conv_relu_7 = Utiles.Relu(self.conv7.Out)

        #GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        #freeGPUMemInGBs = GPUFreeMemoryInBytes / 1024. / 1024 / 1024
        #print("Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs))

        """
        convolution 8
        ReLU
        out shape: (200, 256, 13, 13)
        """
        conv8_filterShape = (layers_metaData['Conv8_NoFiltersOut'], layers_metaData['Conv8_NoFiltersIn'],
                             layers_metaData['Conv8_sizeKernelW'], layers_metaData['Conv8_sizeKernelH'])
        c8Values = initWeights['conv8Values']
        c8ImageShape = (self.batch_size_dynamyc, layers_metaData['Conv7_NoFiltersOut'], layers_metaData['Conv8_sizeImgInH'],
                        layers_metaData['Conv8_sizeImgInW'])
        self.conv8 = ConvLayer.ConvLayer('conv8Layer', self.conv_relu_7, c8Values, conv8_filterShape, c8ImageShape)
        self.conv_relu_8 = Utiles.Relu(self.conv8.Out)

        """
        Pool Layer 4
        out shape: (200, 256, 7, 7)
        """
        self.MaxPool_4 = pool.pool_2d(
            input=self.conv_relu_8,
            stride=(2, 2),  # stride
            ws=(layers_metaData['Poo14_sizeKernelW'], layers_metaData['Poo14_sizeKernelH']),
            mode=layers_metaData['Poo14_mode'],
            ignore_border=False
        )

        """
        convolution 9
        ReLU
        """
        conv9_filterShape = (layers_metaData['Conv9_NoFiltersOut'], layers_metaData['Conv9_NoFiltersIn'],
                             layers_metaData['Conv9_sizeKernelW'], layers_metaData['Conv9_sizeKernelH'])
        c9Values = initWeights['conv9Values']
        c9ImageShape = (
            self.batch_size_dynamyc, layers_metaData['Conv8_NoFiltersOut'], layers_metaData['Conv9_sizeImgInH'], layers_metaData['Conv9_sizeImgInW'])
        self.conv9 = ConvLayer.ConvLayer('conv9Layer', self.MaxPool_4, c9Values, conv9_filterShape, c9ImageShape)
        self.conv_relu_9 = Utiles.Relu(self.conv9.Out)



        """
        convolution 10
        ReLU
        outshape: (N, 320, 7, 7)
        """
        conv10_filterShape = (layers_metaData['Conv10_NoFiltersOut'], layers_metaData['Conv10_NoFiltersIn'],
                             layers_metaData['Conv10_sizeKernelW'], layers_metaData['Conv10_sizeKernelH'])
        c10Values = initWeights['conv10Values']
        c10ImageShape = (self.batch_size_dynamyc, layers_metaData['Conv9_NoFiltersOut'], layers_metaData['Conv10_sizeImgInH'],
                         layers_metaData['Conv10_sizeImgInW'])
        self.conv10 = ConvLayer.ConvLayer('conv10Layer', self.conv_relu_9, c10Values, conv10_filterShape, c10ImageShape)
        self.conv_relu_10 = Utiles.Relu(self.conv10.Out)

        """
        Pool Layer 5 AVG
        outShape: (200, 320, 1, 1)
        """
        self.MaxPool_5 = pool.pool_2d(
            input=self.conv_relu_10,
            stride=(1, 1),  # stride
            ws=(layers_metaData['Poo15_sizeKernelW'], layers_metaData['Poo15_sizeKernelH']),
            mode=layers_metaData['Poo15_mode']
            #ignore_border=True
        )


        self.DO_1 = DropOutLayer.DropOutLayer(self.MaxPool_5, srng, (self.batch_size, layers_metaData['DO1_size_in'], 1, 1),
                                              isTraining, pDropOut)

        self.DO_1_reshape = theano.tensor.reshape(self.DO_1.output, (self.batch_size, 320))


        """
        FC1
        outputShape: (200, 10575)
        """
        FC1Values = initWeights['FC1Values']
        FC1_BiasInitial_bias_values = initWeights['FC1BiasValues']

        self.FC_1 = FCLayer.FCLayer(
            input_image=self.DO_1_reshape,
            initial_filter_values=FC1Values,
            initial_bias_values=FC1_BiasInitial_bias_values,
            layer_name="FcLayer_1"
        )
        self.FC_relu_1 = Utiles.Relu(self.FC_1.ProductoCruz)

        #SoftMaxValues = initWeights['SoftMax1Values']
        #SoftMaxBiasInitial_bias_values = initWeights['SoftMax1BiasValues']
        self.SoftMax_1 = SoftMaxLayer.SoftMaxLayer(
            input_image=self.FC_relu_1,
            layer_name="SoftMax_1"
        )

        self.CostFunction = self.SoftMax_1.negative_log_likelihood(self.y)

        self.Weigths =[self.conv1.Filter,
                       self.conv2.Filter,
                       self.conv3.Filter,
                       self.conv4.Filter,
                       self.conv5.Filter,
                       self.conv6.Filter,
                       self.conv7.Filter,
                       self.conv8.Filter,
                       self.conv9.Filter,
                       self.conv10.Filter,
                       self.FC_1.Filter,
                       self.FC_1.Bias ]

    def GetWeightsValues(self):
        WeightNPValues =  {
            "conv1Values": self.conv1.GetNPValue(),
            "conv2Values": self.conv2.GetNPValue(),
            "conv3Values": self.conv3.GetNPValue(),
            "conv4Values": self.conv4.GetNPValue(),
            "conv5Values": self.conv5.GetNPValue(),
            "conv6Values": self.conv6.GetNPValue(),
            "conv7Values": self.conv7.GetNPValue(),
            "conv8Values": self.conv8.GetNPValue(),
            "conv9Values": self.conv9.GetNPValue(),
            "conv10Values":self.conv10.GetNPValue(),
            "FC1Values": self.FC_1.GetNPFilterValue(),
            "FC1BiasValues": self.FC_1.GetNPBiasValue()
        }
        return WeightNPValues
    """  
            FC2Values = initWeights['FC2Values']
            FC2_BiasInitial_bias_values = initWeights['FC2BiasValues']
            self.FC_2 = fc_layer.FCLayer(
                input_image=self.DO_1.output,
                initial_filter_values=FC2Values,
                initial_bias_values=FC2_BiasInitial_bias_values,
                layer_name="FcLayer_2"
            )
            self.FC_relu_2 = utils.Relu(self.FC_2.ProductoCruz)
    
            self.DO_2 = dropout_layer.DropOutLayer(self.FC_relu_2, srng, (batch_size, layers_metaData['DO2_size_in']), isTraining, pDropOut)
    
            SoftMaxValues = initWeights['SoftMax1Values']
            SoftMaxBiasInitial_bias_values = initWeights['SoftMax1BiasValues']
            self.SoftMax_1 = softmax_layer.SoftMaxLayer(
                input_image=self.DO_2.output,
                initial_filter_values=SoftMaxValues,
                initial_bias_values=SoftMaxBiasInitial_bias_values,
                layer_name="SoftMax_1"
            )
            """



    def negative_log_likelihood(self):
        """
        Esta funcion es nuestro criterio de medida de que tan bien ha realizado el calculo la CNN,
        Checamos la probabilidad predecida para Y, despues sumamos todas las probabilidades y les sacamos el promedio de que tanto se equivoca
        Cost Function with mean and not sum
        :param y: es una coleccion de indices donde cada row representa un ejemplo y su valor el indice correcto
        :return:
        """
        listaindices = T.arange(self.y.shape[0])  # creamos una secuencia de 0 hasta el numero de de elementos en y

        resultLog = T.log(
            self.p_y_given_x)  # aplicamos la funcion log a las probabilidades predecidas por cada una de las posibles clases, Siempre resultara un numero negativo, si la probabilidad es muy baja dara un resultado mas negativo
        result = resultLog[
            listaindices, self.y]  # por cada row(ejemplo predecido) obtenemos la probabilidad de la clase correcta(y), si es correcta debe ser muy alta y si es incorrecta debe ser muy baja

        return T.mean(
            result)  # Regresamos el promedio de las respuestas calculadas, si es muy alto significa que va bien por lo que queremos Maximizar el resultado

