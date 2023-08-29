import torch.nn
import numpy as np
import ipdb

from complexPyTorch.complexLayers import ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu, complex_upsample

from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d


def complex_leaky_relu(negative_slope, input):
    """
    Implementation of Complex-valued Leaky Rectified Linear Unit (LReLU)
    :param num_feat
    :param input:
    :return:
    """
    return torch.nn.LeakyReLU(negative_slope=negative_slope).to(input.device)(input.real).type(torch.complex64)+1j*torch.nn.LeakyReLU(negative_slope=negative_slope).to(input.device)(input.imag).type(torch.complex64)

def complex_prelu(num_feat,input):
    """
    Implementation of Complex-valued Parametric Rectified Linear Unit (PReLU)
    :param num_feat
    :param input:
    :return:
    """
    return torch.nn.PReLU(num_feat).to(input.device)(input.real).type(torch.complex64)+1j*torch.nn.PReLU(num_feat).to(input.device)(input.imag).type(torch.complex64)

class EncoderBlock(torch.nn.Module):
    def __init__(self, 
                channel_in, 
                out_channels, 
                kernel_size, 
                activation,
                bn=True
            ):
        
        super(EncoderBlock, self).__init__()
        self.bn = bn
        self.activation = activation
        self.out_channels = out_channels
        #TODO: the only way to have same dimension as the paper?
        self.conv2d_c = Conv2d(channel_in, out_channels, kernel_size, stride=2, padding=1, dtype=torch.complex64) 
        self.bn_c = ComplexBatchNorm2d(out_channels) #BatchNorm from pytorch do not support complex values
        
        
    def forward(self, x):
        x = self.conv2d_c(x)
        
        if self.bn: 
            x = self.bn_c(x)
        
        
        if self.activation == "relu":
            x = complex_relu(x) 
        
        elif self.activation == "lrelu":
            x = complex_leaky_relu(0.2, x)
        
        elif self.activation == "prelu":
            x = complex_prelu(self.out_channels, x)
        
        return x
    
class DecoderBlock(torch.nn.Module):
    def __init__(self, 
                in_channels,
                out_channels, 
                kernel_size, 
                activation="relu",
                layer="up",
                bn=True, 
            ):
        
        super(DecoderBlock, self).__init__()
        
        # https://github.com/wavefrontshaping/complexPyTorch/blob/70a511c1bedc4c7eeba0d571638b35ff0d8347a2/complexPyTorch/complexFunctions.py#L65
        # alternative conv2transpose, for artifacts could be the same since we are not working with images 
        #TODO: it is not implemented for complex values
        
        self.layer = layer 
        self.bn = bn
        self.activation = "lrelu"
        self.out_channels = out_channels
        self.upsampling = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, dtype=torch.complex64) #paddding = same -> get how, stride??
        
        self.conv_tran = ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dtype=torch.complex64)
        self.bn_c = ComplexBatchNorm2d(out_channels)
        
        
    def forward(self, x, enc):    
        
        if self.layer == "up":
            x = complex_upsample(x, scale_factor=(2, 2)) 
            x = torch.cat((x, enc), dim=1) 
            x = self.conv(x) 
        
        elif self.layer == "ct": #conv transpose
            x = complex_upsample(x, scale_factor=(2, 2)) 
            x = torch.cat((x, enc), dim=1) 
            x = self.conv_tran(x) 
        
        if self.bn:
            x = self.bn_c(x)
            
        
        if self.activation == "relu":
            x = complex_relu(x) 
        
        elif self.activation == "lrelu":
            x = complex_leaky_relu(0.2, x)
        
        elif self.activation == "prelu":
            x = complex_prelu(self.out_channels, x)
        
        else:
            print("No activation function implemented") #TODO: Error
            
        
        return x
        

class ComplexUnet(torch.nn.Module):
    def __init__(self,
                 config,
                ): 
        super(ComplexUnet, self).__init__()
        
        self.activation = config["activation"]
        self.layer = config["layer"]
        
        # Encoder
        self.enc1 = EncoderBlock(80, 128, kernel_size=3, activation=self.activation, bn=False) #is 5 instead of 3
        self.enc2 = EncoderBlock(128, 256, kernel_size=3, activation=self.activation)
        self.enc3 = EncoderBlock(256, 512, kernel_size=3, activation=self.activation)
        self.enc4 = EncoderBlock(512, 1024, kernel_size=3, activation=self.activation)

        # decoder
        self.dec1 = DecoderBlock((1024+512), 512, 3, self.activation, self.layer) 
        self.dec2 = DecoderBlock((512+256), 256, 3, self.activation, self.layer)
        self.dec3 = DecoderBlock((256+128), 128, 3, self.activation, self.layer)
        self.dec4 = DecoderBlock((128+80), 80, 3, self.activation, self.layer, bn=False) 
        
        #TODO: Should we keep this layer? no sigmoid as activation (was included in the original paper)
        self.outputs = Conv2d(80, 40, kernel_size=1, dtype=torch.complex64)
    
    
    def forward(self, x):
        
        # encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # decoder
        dec1 = self.dec1(enc4, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)
        dec4 = self.dec4(dec3, x)
        output = self.outputs(dec4)
       
        
        return output