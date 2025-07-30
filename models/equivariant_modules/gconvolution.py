import torch
from escnn import gspaces
from escnn import nn
import numpy as np
class EquivariantUpsampler(nn.EquivariantModule):
    def __init__(self,
                group_act,
                in_type,
                out_type,
                kernel_size,  
                upsample_factors,
                channel_sizes,
                mode='bilinear',
                stride=1):
        super().__init__()
        assert len(upsample_factors) == len(channel_sizes), "upsample_factors and channel_sizes must have the same length"
        self.group_act = group_act
        self.in_type = in_type
        self.out_type = out_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample_factors = upsample_factors
        self.channel_sizes = channel_sizes
        self.mode = mode
        self.layers = []
        self.padding = (self.kernel_size-1)//2
        self.in_channels = in_type.size
        intermidiate_types = []
        in_rep = self.in_type
        out_rep =  nn.FieldType(group_act, self.in_channels*[group_act.regular_repr])
        intermidiate_types.append(out_rep)
        self.lifting = nn.R2Conv(in_rep,
                                out_rep, 
                                self.kernel_size, 
                                stride=self.stride, 
                                padding=self.padding)
        
        for i in range(len(self.channel_sizes)):
            in_rep = out_rep
            out_rep = nn.FieldType(group_act, self.channel_sizes[i]*[group_act.regular_repr])
            intermidiate_types.append(out_rep)
            self.layers.append(nn.R2Upsampling(in_rep,scale_factor=self.upsample_factors[i], mode=self.mode))
            self.layers.append(nn.R2Conv(in_rep,
                                         out_rep,
                                         self.kernel_size,
                                         stride=self.stride,
                                         padding=self.padding))
            self.layers.append(nn.ReLU(out_rep))
        self.upsampler = nn.SequentialModule(*self.layers)
        self.pool = nn.R2Conv(out_rep, 
                              self.out_type, 
                              self.kernel_size, 
                              stride=self.stride, 
                              padding=self.padding)

    def forward(self, x):
        x = self.lifting(x)
        x = self.upsampler(x)
        x = self.pool(x)
        return x

    def evaluate_output_shape(self, input_shape: tuple):
        pass