import torch.nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu, complex_upsample
from torch.nn import Conv2d, ConvTranspose2d


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
                stride,
                padding,
                activation,
                bn=True
            ):
        
        super(EncoderBlock, self).__init__()
        
        # convolutional layer config
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bn = bn
        
        # activation function
        self.activation = activation
        
        self.conv2d_c = Conv2d(channel_in, out_channels, kernel_size, stride=stride, padding=padding, dtype=torch.complex64) 
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
                padding,
                activation="relu",
                up_layer="up",
                scale_factor=2,
                bn=True, 
            ):
        
        super(DecoderBlock, self).__init__()

        #TODO: it is not implemented for complex values
        
        # conv config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = kernel_size
        self.padding = padding
        self.bn = bn
        
        #upsampling layer
        self.up_layer = up_layer 
        self.scale_factor = scale_factor
        
        # activation function
        self.activation = activation

        self.upsampling = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.ks, padding=self.padding, dtype=torch.complex64) #paddding = same -> get how, stride??
        
        self.conv_tran = ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size, dtype=torch.complex64)
        self.bn_c = ComplexBatchNorm2d(self.out_channels)
        
        
    def forward(self, x, enc):    
        
        # select upsampling layer
        if self.up_layer == "up":
            x = complex_upsample(x, scale_factor=(self.scale_factor, self.scale_factor)) 
            x = torch.cat((x, enc), dim=1) 
            x = self.conv(x) 
        
        elif self.up_layer == "ct": #conv transpose
            x = complex_upsample(x, scale_factor=(self.scale_factor, self.scale_factor)) 
            x = torch.cat((x, enc), dim=1) 
            x = self.conv_tran(x) 
        
        if self.bn:
            x = self.bn_c(x)
            
        # select activation function
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
        
        # encoder configuration
        self.enc_ks = config["encoder"]["kernel_size"]
        self.enc_stride = config["encoder"]["stride"]
        self.enc_padding = config["encoder"]["padding"]
        self.enc_acti = config["encoder"]["activation"]
        
        # decoder configuration
        self.dec_ks = config["decoder"]["kernel_size"]
        self.dec_padding = config["decoder"]["padding"]
        self.dec_acti = config["decoder"]["activation"]
        self.up_layer = config["decoder"]["up_layer"]
        self.scale_factor = config["decoder"]["scale_factor"]
    
        # last output layer configuration
        self.output_ks = config["output"]["kernel_size"]
        self.do_bn_encoder = config["encoder"]["batch_norm"]
        self.do_bn_decoder = config["encoder"]["batch_norm"]

        self.factor = 1
        # Encoder
        self.enc1 = EncoderBlock(80, 128//self.factor, kernel_size=self.enc_ks, stride=self.enc_stride, padding=self.enc_padding, activation=self.enc_acti, bn=False) #is 5 instead of 3
        self.enc2 = EncoderBlock(128//self.factor, 256//self.factor, kernel_size=self.enc_ks, stride=self.enc_stride, padding=self.enc_padding, activation=self.enc_acti,bn=self.do_bn_encoder)
        self.enc3 = EncoderBlock(256//self.factor, 512//self.factor, kernel_size=self.enc_ks, stride=self.enc_stride, padding=self.enc_padding, activation=self.enc_acti,bn=self.do_bn_encoder)
        self.enc4 = EncoderBlock(512//self.factor, 1024//self.factor, kernel_size=self.enc_ks, stride=self.enc_stride, padding=self.enc_padding, activation=self.enc_acti,bn=self.do_bn_encoder)

        # decoder
        self.dec1 = DecoderBlock((1024//self.factor+512//self.factor), 512//self.factor, kernel_size=self.dec_ks, padding=self.dec_padding, activation=self.dec_acti, up_layer=self.up_layer, scale_factor=self.scale_factor,bn=self.do_bn_decoder)
        self.dec2 = DecoderBlock((512//self.factor+256//self.factor), 256//self.factor, kernel_size=self.dec_ks, padding=self.dec_padding, activation=self.dec_acti, up_layer=self.up_layer, scale_factor=self.scale_factor,bn=self.do_bn_decoder)
        self.dec3 = DecoderBlock((256//self.factor+128//self.factor), 128//self.factor, kernel_size=self.dec_ks, padding=self.dec_padding, activation=self.dec_acti, up_layer=self.up_layer, scale_factor=self.scale_factor,bn=self.do_bn_decoder)
        self.dec4 = DecoderBlock((128//self.factor+80), 80, kernel_size=self.dec_ks, padding=self.dec_padding, activation=self.dec_acti, up_layer=self.up_layer, scale_factor=self.scale_factor, bn=False)
        
        #TODO: Should we keep this layer? no sigmoid as activation (was included in the original paper)
        self.outputs = Conv2d(80, 40, kernel_size=self.output_ks, dtype=torch.complex64)
    
    
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

