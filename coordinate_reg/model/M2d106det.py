import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):


    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.conv_1_conv2d = self.__conv(2, name='conv_1_conv2d', in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.conv_1_batchnorm = self.__batch_normalization(2, 'conv_1_batchnorm', num_features=16, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_2_dw_conv2d = self.__conv(2, name='conv_2_dw_conv2d', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=16, bias=False)
        self.conv_2_dw_batchnorm = self.__batch_normalization(2, 'conv_2_dw_batchnorm', num_features=16, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_2_conv2d = self.__conv(2, name='conv_2_conv2d', in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_2_batchnorm = self.__batch_normalization(2, 'conv_2_batchnorm', num_features=32, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_3_dw_conv2d = self.__conv(2, name='conv_3_dw_conv2d', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), groups=32, bias=False)
        self.conv_3_dw_batchnorm = self.__batch_normalization(2, 'conv_3_dw_batchnorm', num_features=32, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_3_conv2d = self.__conv(2, name='conv_3_conv2d', in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_3_batchnorm = self.__batch_normalization(2, 'conv_3_batchnorm', num_features=64, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_4_dw_conv2d = self.__conv(2, name='conv_4_dw_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=64, bias=False)
        self.conv_4_dw_batchnorm = self.__batch_normalization(2, 'conv_4_dw_batchnorm', num_features=64, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_4_conv2d = self.__conv(2, name='conv_4_conv2d', in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_4_batchnorm = self.__batch_normalization(2, 'conv_4_batchnorm', num_features=64, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_5_dw_conv2d = self.__conv(2, name='conv_5_dw_conv2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), groups=64, bias=False)
        self.conv_5_dw_batchnorm = self.__batch_normalization(2, 'conv_5_dw_batchnorm', num_features=64, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_5_conv2d = self.__conv(2, name='conv_5_conv2d', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_5_batchnorm = self.__batch_normalization(2, 'conv_5_batchnorm', num_features=128, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_6_dw_conv2d = self.__conv(2, name='conv_6_dw_conv2d', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=128, bias=False)
        self.conv_6_dw_batchnorm = self.__batch_normalization(2, 'conv_6_dw_batchnorm', num_features=128, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_6_conv2d = self.__conv(2, name='conv_6_conv2d', in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_6_batchnorm = self.__batch_normalization(2, 'conv_6_batchnorm', num_features=128, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_7_dw_conv2d = self.__conv(2, name='conv_7_dw_conv2d', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), groups=128, bias=False)
        self.conv_7_dw_batchnorm = self.__batch_normalization(2, 'conv_7_dw_batchnorm', num_features=128, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_7_conv2d = self.__conv(2, name='conv_7_conv2d', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_7_batchnorm = self.__batch_normalization(2, 'conv_7_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_8_dw_conv2d = self.__conv(2, name='conv_8_dw_conv2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv_8_dw_batchnorm = self.__batch_normalization(2, 'conv_8_dw_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_8_conv2d = self.__conv(2, name='conv_8_conv2d', in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_8_batchnorm = self.__batch_normalization(2, 'conv_8_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_9_dw_conv2d = self.__conv(2, name='conv_9_dw_conv2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv_9_dw_batchnorm = self.__batch_normalization(2, 'conv_9_dw_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_9_conv2d = self.__conv(2, name='conv_9_conv2d', in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_9_batchnorm = self.__batch_normalization(2, 'conv_9_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_10_dw_conv2d = self.__conv(2, name='conv_10_dw_conv2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv_10_dw_batchnorm = self.__batch_normalization(2, 'conv_10_dw_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_10_conv2d = self.__conv(2, name='conv_10_conv2d', in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_10_batchnorm = self.__batch_normalization(2, 'conv_10_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_11_dw_conv2d = self.__conv(2, name='conv_11_dw_conv2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv_11_dw_batchnorm = self.__batch_normalization(2, 'conv_11_dw_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_11_conv2d = self.__conv(2, name='conv_11_conv2d', in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_11_batchnorm = self.__batch_normalization(2, 'conv_11_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_12_dw_conv2d = self.__conv(2, name='conv_12_dw_conv2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv_12_dw_batchnorm = self.__batch_normalization(2, 'conv_12_dw_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_12_conv2d = self.__conv(2, name='conv_12_conv2d', in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_12_batchnorm = self.__batch_normalization(2, 'conv_12_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_13_dw_conv2d = self.__conv(2, name='conv_13_dw_conv2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=256, bias=False)
        self.conv_13_dw_batchnorm = self.__batch_normalization(2, 'conv_13_dw_batchnorm', num_features=256, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_13_conv2d = self.__conv(2, name='conv_13_conv2d', in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_13_batchnorm = self.__batch_normalization(2, 'conv_13_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_14_dw_conv2d = self.__conv(2, name='conv_14_dw_conv2d', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=512, bias=False)
        self.conv_14_dw_batchnorm = self.__batch_normalization(2, 'conv_14_dw_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_14_conv2d = self.__conv(2, name='conv_14_conv2d', in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_14_batchnorm = self.__batch_normalization(2, 'conv_14_batchnorm', num_features=512, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.conv_15_conv2d = self.__conv(2, name='conv_15_conv2d', in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.conv_15_batchnorm = self.__batch_normalization(2, 'conv_15_batchnorm', num_features=64, eps=0.0010000000474974513, momentum=0.8999999761581421)
        self.fc1 = self.__dense(name = 'fc1', in_features = 576, out_features = 212, bias = True)

        self.conv_1_prelu = self.__prelu(name='conv_1_relu', num_parameters=16)
        self.conv_2_dw_relu  = self.__prelu(name='conv_2_dw_relu',  num_parameters=16)
        self.conv_2_relu  = self.__prelu(name='conv_2_relu',  num_parameters=32)
        self.conv_3_dw_relu  = self.__prelu(name='conv_3_dw_relu',  num_parameters=32)
        self.conv_3_relu  = self.__prelu(name='conv_3_relu',  num_parameters=64)
        self.conv_4_dw_relu  = self.__prelu(name='conv_4_dw_relu',  num_parameters=64)
        self.conv_4_relu  = self.__prelu(name='conv_4_relu',  num_parameters=64)
        self.conv_5_dw_relu  = self.__prelu(name='conv_5_dw_relu',  num_parameters=64)
        self.conv_5_relu  = self.__prelu(name='conv_5_relu',  num_parameters=128)

        self.conv_6_dw_relu  = self.__prelu(name='conv_6_dw_relu',  num_parameters=128)
        self.conv_6_relu  = self.__prelu(name='conv_6_relu',  num_parameters=128)
        self.conv_7_dw_relu  = self.__prelu(name='conv_7_dw_relu',  num_parameters=128)
        self.conv_7_relu  = self.__prelu(name='conv_7_relu',  num_parameters=256)
        self.conv_8_dw_relu  = self.__prelu(name='conv_8_dw_relu',  num_parameters=256)
        self.conv_8_relu  = self.__prelu(name='conv_8_relu',  num_parameters=256)
        self.conv_9_dw_relu  = self.__prelu(name='conv_9_dw_relu',  num_parameters=256)
        self.conv_9_relu  = self.__prelu(name='conv_9_relu',  num_parameters=256)
        self.conv_10_dw_relu  = self.__prelu(name='conv_10_dw_relu',  num_parameters=256)
        self.conv_10_relu  = self.__prelu(name='conv_10_relu',  num_parameters=256)

        self.conv_11_dw_relu  = self.__prelu(name='conv_11_dw_relu',  num_parameters=256)
        self.conv_11_relu  = self.__prelu(name='conv_11_relu',  num_parameters=256)
        self.conv_12_dw_relu  = self.__prelu(name='conv_12_dw_relu',  num_parameters=256)
        self.conv_12_relu  = self.__prelu(name='conv_12_relu',  num_parameters=256)
        self.conv_13_dw_relu  = self.__prelu(name='conv_13_dw_relu',  num_parameters=256)
        self.conv_13_relu  = self.__prelu(name='conv_13_relu',  num_parameters=512)
        self.conv_14_dw_relu  = self.__prelu(name='conv_14_dw_relu',  num_parameters=512)
        self.conv_14_relu  = self.__prelu(name='conv_14_relu',  num_parameters=512)

        self.conv_15_relu  = self.__prelu(name='conv_15_relu',  num_parameters=64)

        self.minusscalar0_second = nn.Parameter(torch.from_numpy(_weights_dict['minusscalar0_second']['value']))
        self.mulscalar0_second = nn.Parameter(torch.from_numpy(_weights_dict['mulscalar0_second']['value']))

    def forward(self, x):
        minusscalar0    = x - self.minusscalar0_second
        mulscalar0      = minusscalar0 * self.mulscalar0_second
        conv_1_conv2d_pad = F.pad(mulscalar0, (1, 1, 1, 1))
        conv_1_conv2d   = self.conv_1_conv2d(conv_1_conv2d_pad)
        conv_1_batchnorm = self.conv_1_batchnorm(conv_1_conv2d)
        conv_1_relu     = self.conv_1_prelu(conv_1_batchnorm)
        conv_2_dw_conv2d_pad = F.pad(conv_1_relu, (1, 1, 1, 1))
        conv_2_dw_conv2d = self.conv_2_dw_conv2d(conv_2_dw_conv2d_pad)
        conv_2_dw_batchnorm = self.conv_2_dw_batchnorm(conv_2_dw_conv2d)
        conv_2_dw_relu  = self.conv_2_dw_relu(conv_2_dw_batchnorm)
        conv_2_conv2d   = self.conv_2_conv2d(conv_2_dw_relu)
        conv_2_batchnorm = self.conv_2_batchnorm(conv_2_conv2d)
        conv_2_relu     = self.conv_2_relu(conv_2_batchnorm)
        conv_3_dw_conv2d_pad = F.pad(conv_2_relu, (1, 1, 1, 1))
        conv_3_dw_conv2d = self.conv_3_dw_conv2d(conv_3_dw_conv2d_pad)
        conv_3_dw_batchnorm = self.conv_3_dw_batchnorm(conv_3_dw_conv2d)
        conv_3_dw_relu  = self.conv_3_dw_relu(conv_3_dw_batchnorm)
        conv_3_conv2d   = self.conv_3_conv2d(conv_3_dw_relu)
        conv_3_batchnorm = self.conv_3_batchnorm(conv_3_conv2d)
        conv_3_relu     = self.conv_3_relu(conv_3_batchnorm)
        conv_4_dw_conv2d_pad = F.pad(conv_3_relu, (1, 1, 1, 1))
        conv_4_dw_conv2d = self.conv_4_dw_conv2d(conv_4_dw_conv2d_pad)
        conv_4_dw_batchnorm = self.conv_4_dw_batchnorm(conv_4_dw_conv2d)
        conv_4_dw_relu  = self.conv_4_dw_relu(conv_4_dw_batchnorm)
        conv_4_conv2d   = self.conv_4_conv2d(conv_4_dw_relu)
        conv_4_batchnorm = self.conv_4_batchnorm(conv_4_conv2d)
        conv_4_relu     = self.conv_4_relu(conv_4_batchnorm)
        conv_5_dw_conv2d_pad = F.pad(conv_4_relu, (1, 1, 1, 1))
        conv_5_dw_conv2d = self.conv_5_dw_conv2d(conv_5_dw_conv2d_pad)
        conv_5_dw_batchnorm = self.conv_5_dw_batchnorm(conv_5_dw_conv2d)
        conv_5_dw_relu  = self.conv_5_dw_relu(conv_5_dw_batchnorm)
        conv_5_conv2d   = self.conv_5_conv2d(conv_5_dw_relu)
        conv_5_batchnorm = self.conv_5_batchnorm(conv_5_conv2d)
        conv_5_relu     = self.conv_5_relu(conv_5_batchnorm)
        conv_6_dw_conv2d_pad = F.pad(conv_5_relu, (1, 1, 1, 1))
        conv_6_dw_conv2d = self.conv_6_dw_conv2d(conv_6_dw_conv2d_pad)
        conv_6_dw_batchnorm = self.conv_6_dw_batchnorm(conv_6_dw_conv2d)
        conv_6_dw_relu  = self.conv_6_dw_relu(conv_6_dw_batchnorm)
        conv_6_conv2d   = self.conv_6_conv2d(conv_6_dw_relu)
        conv_6_batchnorm = self.conv_6_batchnorm(conv_6_conv2d)
        conv_6_relu     = self.conv_6_relu(conv_6_batchnorm)
        conv_7_dw_conv2d_pad = F.pad(conv_6_relu, (1, 1, 1, 1))
        conv_7_dw_conv2d = self.conv_7_dw_conv2d(conv_7_dw_conv2d_pad)
        conv_7_dw_batchnorm = self.conv_7_dw_batchnorm(conv_7_dw_conv2d)
        conv_7_dw_relu  = self.conv_7_dw_relu(conv_7_dw_batchnorm)
        conv_7_conv2d   = self.conv_7_conv2d(conv_7_dw_relu)
        conv_7_batchnorm = self.conv_7_batchnorm(conv_7_conv2d)
        conv_7_relu     = self.conv_7_relu(conv_7_batchnorm)
        conv_8_dw_conv2d_pad = F.pad(conv_7_relu, (1, 1, 1, 1))
        conv_8_dw_conv2d = self.conv_8_dw_conv2d(conv_8_dw_conv2d_pad)
        conv_8_dw_batchnorm = self.conv_8_dw_batchnorm(conv_8_dw_conv2d)
        conv_8_dw_relu  = self.conv_8_dw_relu(conv_8_dw_batchnorm)
        conv_8_conv2d   = self.conv_8_conv2d(conv_8_dw_relu)
        conv_8_batchnorm = self.conv_8_batchnorm(conv_8_conv2d)
        conv_8_relu     = self.conv_8_relu(conv_8_batchnorm)
        conv_9_dw_conv2d_pad = F.pad(conv_8_relu, (1, 1, 1, 1))
        conv_9_dw_conv2d = self.conv_9_dw_conv2d(conv_9_dw_conv2d_pad)
        conv_9_dw_batchnorm = self.conv_9_dw_batchnorm(conv_9_dw_conv2d)
        conv_9_dw_relu  = self.conv_9_dw_relu(conv_9_dw_batchnorm)
        conv_9_conv2d   = self.conv_9_conv2d(conv_9_dw_relu)
        conv_9_batchnorm = self.conv_9_batchnorm(conv_9_conv2d)
        conv_9_relu     = self.conv_9_relu(conv_9_batchnorm)
        conv_10_dw_conv2d_pad = F.pad(conv_9_relu, (1, 1, 1, 1))
        conv_10_dw_conv2d = self.conv_10_dw_conv2d(conv_10_dw_conv2d_pad)
        conv_10_dw_batchnorm = self.conv_10_dw_batchnorm(conv_10_dw_conv2d)
        conv_10_dw_relu = self.conv_10_dw_relu(conv_10_dw_batchnorm)
        conv_10_conv2d  = self.conv_10_conv2d(conv_10_dw_relu)
        conv_10_batchnorm = self.conv_10_batchnorm(conv_10_conv2d)
        conv_10_relu    = self.conv_10_relu(conv_10_batchnorm)
        conv_11_dw_conv2d_pad = F.pad(conv_10_relu, (1, 1, 1, 1))
        conv_11_dw_conv2d = self.conv_11_dw_conv2d(conv_11_dw_conv2d_pad)
        conv_11_dw_batchnorm = self.conv_11_dw_batchnorm(conv_11_dw_conv2d)
        conv_11_dw_relu = self.conv_11_dw_relu(conv_11_dw_batchnorm)
        conv_11_conv2d  = self.conv_11_conv2d(conv_11_dw_relu)
        conv_11_batchnorm = self.conv_11_batchnorm(conv_11_conv2d)
        conv_11_relu    = self.conv_11_relu(conv_11_batchnorm)
        conv_12_dw_conv2d_pad = F.pad(conv_11_relu, (1, 1, 1, 1))
        conv_12_dw_conv2d = self.conv_12_dw_conv2d(conv_12_dw_conv2d_pad)
        conv_12_dw_batchnorm = self.conv_12_dw_batchnorm(conv_12_dw_conv2d)
        conv_12_dw_relu = self.conv_12_dw_relu(conv_12_dw_batchnorm)
        conv_12_conv2d  = self.conv_12_conv2d(conv_12_dw_relu)
        conv_12_batchnorm = self.conv_12_batchnorm(conv_12_conv2d)
        conv_12_relu    = self.conv_12_relu(conv_12_batchnorm)
        conv_13_dw_conv2d_pad = F.pad(conv_12_relu, (1, 1, 1, 1))
        conv_13_dw_conv2d = self.conv_13_dw_conv2d(conv_13_dw_conv2d_pad)
        conv_13_dw_batchnorm = self.conv_13_dw_batchnorm(conv_13_dw_conv2d)
        conv_13_dw_relu = self.conv_13_dw_relu(conv_13_dw_batchnorm)
        conv_13_conv2d  = self.conv_13_conv2d(conv_13_dw_relu)
        conv_13_batchnorm = self.conv_13_batchnorm(conv_13_conv2d)
        conv_13_relu    = self.conv_13_relu(conv_13_batchnorm)
        conv_14_dw_conv2d_pad = F.pad(conv_13_relu, (1, 1, 1, 1))
        conv_14_dw_conv2d = self.conv_14_dw_conv2d(conv_14_dw_conv2d_pad)
        conv_14_dw_batchnorm = self.conv_14_dw_batchnorm(conv_14_dw_conv2d)
        conv_14_dw_relu = self.conv_14_dw_relu(conv_14_dw_batchnorm)
        conv_14_conv2d  = self.conv_14_conv2d(conv_14_dw_relu)
        conv_14_batchnorm = self.conv_14_batchnorm(conv_14_conv2d)
        conv_14_relu    = self.conv_14_relu(conv_14_batchnorm)
        conv_15_conv2d_pad = F.pad(conv_14_relu, (1, 1, 1, 1))
        conv_15_conv2d  = self.conv_15_conv2d(conv_15_conv2d_pad)
        conv_15_batchnorm = self.conv_15_batchnorm(conv_15_conv2d)
        conv_15_relu    = self.conv_15_relu(conv_15_batchnorm)
        flatten0        = conv_15_relu.view(conv_15_relu.size(0), -1)
        fc1             = self.fc1(flatten0)
        return fc1


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __prelu(name, **kwargs):
        layer = nn.PReLU(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

