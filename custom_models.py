import torch
import torch.nn as nn
import torch.nn.functional as F

class EncDec(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        #out = F.sigmoid(d3)
        return d3


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(67, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x))) # --> 64
        e1 = self.pool1(F.relu(self.enc_conv1(e0))) # --> 32
        e2 = self.pool2(F.relu(self.enc_conv2(e1))) # --> 16
        e3 = self.pool3(F.relu(self.enc_conv3(e2))) # --> 8

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3)) # --> 8

        # decoder
        d0 = F.relu(self.dec_conv0(torch.cat([e2,self.upsample0(b)],dim=1))) # --> 16
        d1 = F.relu(self.dec_conv1(torch.cat([e1,self.upsample1(d0)],dim=1))) # --> 32
        d2 = F.relu(self.dec_conv2(torch.cat([e0,self.upsample2(d1)],dim=1))) # --> 64
        d3 = self.dec_conv3(torch.cat([x,self.upsample3(d2)],dim=1))  # no activation
        return d3


class UNet2(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3,stride=2, padding=1) # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1) # 8

        # decoder (upsampling)
        self.dec_conv0 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 8 -> 16
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 16 -> 32
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 32 -> 64
        self.dec_conv3 = nn.ConvTranspose2d(128, 1, 3,stride=2, padding=1, output_padding=1) # 64 -> 128

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x)) # --> 64
        e1 = F.relu(self.enc_conv1(e0)) # --> 32
        e2 = F.relu(self.enc_conv2(e1)) # --> 16
        e3 = F.relu(self.enc_conv3(e2)) # --> 8

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3)) # --> 8

        # decoder
        d0 = F.relu(self.dec_conv0(torch.cat([e3,b],dim=1))) # --> 16
        d1 = F.relu(self.dec_conv1(torch.cat([e2,d0],dim=1))) # --> 32
        d2 = F.relu(self.dec_conv2(torch.cat([e1,d1],dim=1))) # --> 64
        d3 = self.dec_conv3(torch.cat([e0,d2],dim=1))  # no activation
        return d3



class UNet3(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3,stride=2, padding=1) # 256 -> 128
        self.enc_conv1 = nn.Conv2d(64, 64, 3,stride=2, padding=1) # 128 -> 64
        self.enc_conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 64 -> 32
        self.enc_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 32 -> 16
        self.enc_conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1) # 8

        # decoder (upsampling)
        self.dec_conv0 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 8 -> 16
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 16 -> 32
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 32 -> 64
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 64 -> 128
        self.dec_conv4 = nn.ConvTranspose2d(128, 1, 3,stride=2, padding=1, output_padding=1) # 128 -> 256

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x)) # --> 128
        e1 = F.relu(self.enc_conv1(e0)) # --> 64
        e2 = F.relu(self.enc_conv2(e1)) # --> 32
        e3 = F.relu(self.enc_conv3(e2)) # --> 16
        e4 = F.relu(self.enc_conv4(e3)) # --> 8

        # bottleneck
        b = F.relu(self.bottleneck_conv(e4)) # --> 8

        # decoder
        d0 = F.relu(self.dec_conv1(torch.cat([e4,b],dim=1))) # --> 16
        d1 = F.relu(self.dec_conv1(torch.cat([e3,d0],dim=1))) # --> 32
        d2 = F.relu(self.dec_conv2(torch.cat([e2,d1],dim=1))) # --> 64
        d3 = F.relu(self.dec_conv3(torch.cat([e1,d2],dim=1))) # --> 128
        d4 = self.dec_conv4(torch.cat([e0,d3],dim=1))  # no activation
        return d4


class UNet4(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, stride=2, padding=1)   # 512 --> 256
        self.enc_conv1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 256 --> 128
        self.enc_conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 128 --> 64
        self.enc_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 64 --> 32
        self.enc_conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 32 --> 16
        self.enc_conv5 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 16 --> 8

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)      # 8 --> 8

        # Decoder (upsampling)
        self.dec_conv0 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 8 --> 16
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 16 --> 32
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 32 --> 64
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 64 --> 128
        self.dec_conv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 128 --> 256
        self.dec_conv5 = nn.ConvTranspose2d(128, 1, 3, stride=2, padding=1, output_padding=1)   # 256 --> 512

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))  # 256
        e1 = F.relu(self.enc_conv1(e0)) # 128
        e2 = F.relu(self.enc_conv2(e1)) # 64
        e3 = F.relu(self.enc_conv3(e2)) # 32
        e4 = F.relu(self.enc_conv4(e3)) # 16
        e5 = F.relu(self.enc_conv5(e4)) # 8

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e5)) # 8

        # Decoder
        d0 = F.relu(self.dec_conv0(torch.cat([e5, b], dim=1)))  # 16
        d1 = F.relu(self.dec_conv1(torch.cat([e4, d0], dim=1))) # 32
        d2 = F.relu(self.dec_conv2(torch.cat([e3, d1], dim=1))) # 64
        d3 = F.relu(self.dec_conv3(torch.cat([e2, d2], dim=1))) # 128
        d4 = F.relu(self.dec_conv4(torch.cat([e1, d3], dim=1))) # 256
        d5 = self.dec_conv5(torch.cat([e0, d4], dim=1))         # 512

        return d5

class PH2UNet(nn.Module):
    # Segmentation Net based on UNet architecture, that respects aspect ratio
    # Original Aspect ratio 765,572 --> /2 --> 378,286
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3) # 378,286 --> 189,143        
        self.enc_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 189,143 --> 95,72
        self.enc_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 95,72 --> 48,36
        self.enc_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 48,36 --> 24,18
        
        # Dilations to inflate RF
        self.enc_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=2, padding=2) # 24,18 --> 24,18
        self.enc_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=3, padding=3) # 24,18 --> 24,18
        self.enc_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=4, padding=4) # 24,18 --> 24,18
        


        # bottleneck
        self.bottleneck_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=5, padding=5) # 24,18 --> 24,18 # RF: 483

        # decoder (upsampling)
        self.dec_conv0 = nn.Conv2d(128, 64, 3,stride=1, padding=1) # 24,18 --> 24,18
        self.dec_conv1 = nn.Conv2d(128, 64, 3,stride=1, padding=1) # 24,18 --> 24,18
        self.dec_conv2 = nn.Conv2d(128, 64, 3,stride=1, padding=1) # 24,18 --> 24,18
        self.dec_conv3 = nn.Conv2d(128, 64, 3,stride=2, padding=1) # 24,18 --> 24,18 
        self.upsample34 = nn.Upsample(size=(36,48))  
        self.dec_conv4 = nn.Conv2d(128, 64, 3,stride=2, padding=1) # 48,36 --> 48,36
        self.upsample45 = nn.Upsample(size=(72,95))  
        self.dec_conv5 = nn.Conv2d(128, 64, 3,stride=2, padding=1) # 95,72 --> 95,72
        self.upsample56 = nn.Upsample(size=(143,189))  
        self.dec_conv6 = nn.Conv2d(128, 1, 3,stride=2, padding=11) # 189,143 --> 189,143
        self.upsample67 = nn.Upsample(size=(286,378))  
        self.dec_conv7 = nn.Conv2d(4, 1, 3,stride=1, padding=1) # 378,286 --> 378,286

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))       # 378,286 --> 189,143
        e1 = F.relu(self.enc_conv1(e0))      # 189,143 --> 95,72
        e2 = F.relu(self.enc_conv2(e1))      # 95,72 --> 48,36
        e3 = F.relu(self.enc_conv3(e2))      # 48,36 --> 24,18

        # Dilated layers (same spatial size)
        d4 = F.relu(self.enc_conv4(e3))      # dilation=2
        d5 = F.relu(self.enc_conv5(d4))      # dilation=3
        d6 = F.relu(self.enc_conv6(d5))      # dilation=4

        # Bottleneck
        b = F.relu(self.bottleneck_conv(d6)) # dilation=5

        # Decoder
        u0 = F.relu(self.dec_conv0(torch.cat([d6, b], dim=1)))     # 24,18
        u1 = F.relu(self.dec_conv1(torch.cat([d5, u0], dim=1)))    # 24,18
        u2 = F.relu(self.dec_conv2(torch.cat([d4, u1], dim=1)))    # 24,18
        u3 = F.relu(self.dec_conv3(torch.cat([e3, u2], dim=1)))    # 24,18
        u3 = self.upsample34(u3)
        u4 = F.relu(self.dec_conv4(torch.cat([e2, u3], dim=1)))    
        u4 = self.upsample45(u4)
        u5 = F.relu(self.dec_conv5(torch.cat([e1, u4], dim=1)))    
        u5 = self.upsample56(u5)
        u6 = self.dec_conv6(torch.cat([e0, u5], dim=1))           
        u6 = self.upsample67(u6)

        u7 = self.dec_conv7(torch.cat([x,u6],dim=1))

        return u7
    
class DRIVEUNet(nn.Module):
    # Segmentation Net based on UNet architecture, that respects aspect ratio
    # Original Aspect ratio 765,572 --> /2 --> 378,286
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3) # 378,286 --> 189,143        
        self.enc_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 189,143 --> 95,72
        self.enc_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 95,72 --> 48,36
        self.enc_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 48,36 --> 24,18
        
        # Dilations to inflate RF
        self.enc_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=2, padding=2) # 24,18 --> 24,18
        self.enc_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=3, padding=3) # 24,18 --> 24,18
        self.enc_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=4, padding=4) # 24,18 --> 24,18
        


        # bottleneck
        self.bottleneck_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=5, padding=5) # 24,18 --> 24,18 # RF: 483

        # decoder (upsampling)
        self.dec_conv0 = nn.Conv2d(128, 64, 3,stride=1, padding=1) # 24,18 --> 24,18
        self.dec_conv1 = nn.Conv2d(128, 64, 3,stride=1, padding=1) # 24,18 --> 24,18
        self.dec_conv2 = nn.Conv2d(128, 64, 3,stride=1, padding=1) # 24,18 --> 24,18
        self.dec_conv3 = nn.Conv2d(128, 64, 3,stride=2, padding=1) # 24,18 --> 24,18 
        self.upsample34 = nn.Upsample(size=(36,37))  
        self.dec_conv4 = nn.Conv2d(128, 64, 3,stride=2, padding=1) # 48,36 --> 48,36
        self.upsample45 = nn.Upsample(size=(71,73))  
        self.dec_conv5 = nn.Conv2d(128, 64, 3,stride=2, padding=1) # 95,72 --> 95,72
        self.upsample56 = nn.Upsample(size=(141,146))  
        self.dec_conv6 = nn.Conv2d(128, 1, 3,stride=2, padding=11) # 189,143 --> 189,143
        self.upsample67 = nn.Upsample(size=(282,292))  
        self.dec_conv7 = nn.Conv2d(4, 1, 3,stride=1, padding=1) # 378,286 --> 378,286

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))       # 378,286 --> 189,143
        e1 = F.relu(self.enc_conv1(e0))      # 189,143 --> 95,72
        e2 = F.relu(self.enc_conv2(e1))      # 95,72 --> 48,36
        e3 = F.relu(self.enc_conv3(e2))      # 48,36 --> 24,18

        # Dilated layers (same spatial size)
        d4 = F.relu(self.enc_conv4(e3))      # dilation=2
        d5 = F.relu(self.enc_conv5(d4))      # dilation=3
        d6 = F.relu(self.enc_conv6(d5))      # dilation=4

        # Bottleneck
        b = F.relu(self.bottleneck_conv(d6)) # dilation=5

        # Decoder
        u0 = F.relu(self.dec_conv0(torch.cat([d6, b], dim=1)))     # 24,18
        u1 = F.relu(self.dec_conv1(torch.cat([d5, u0], dim=1)))    # 24,18
        u2 = F.relu(self.dec_conv2(torch.cat([d4, u1], dim=1)))    # 24,18
        u3 = F.relu(self.dec_conv3(torch.cat([e3, u2], dim=1)))    # 24,18
        u3 = self.upsample34(u3)
        u4 = F.relu(self.dec_conv4(torch.cat([e2, u3], dim=1)))    
        u4 = self.upsample45(u4)
        u5 = F.relu(self.dec_conv5(torch.cat([e1, u4], dim=1)))    
        u5 = self.upsample56(u5)
        u6 = self.dec_conv6(torch.cat([e0, u5], dim=1))           
        u6 = self.upsample67(u6)

        u7 = self.dec_conv7(torch.cat([x,u6],dim=1))

        return u7




class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3,stride=2, padding=1,dilation=1) # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, stride=2, padding=2,dilation=2) # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=3,dilation=3) # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=4,dilation=4) # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1) # 8

        # decoder (upsampling)
        self.dec_conv0 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 8 -> 16
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 16 -> 32
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1, output_padding=1) # 32 -> 64
        self.dec_conv3 = nn.ConvTranspose2d(128, 1, 3,stride=2, padding=1, output_padding=1) # 64 -> 128

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x)) # --> 64
        e1 = F.relu(self.enc_conv1(e0)) # --> 32
        e2 = F.relu(self.enc_conv2(e1)) # --> 16
        e3 = F.relu(self.enc_conv3(e2)) # --> 8

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3)) # --> 8

        # decoder
        d0 = F.relu(self.dec_conv0(torch.cat([e3,b],dim=1))) # --> 16
        d1 = F.relu(self.dec_conv1(torch.cat([e2,d0],dim=1))) # --> 32
        d2 = F.relu(self.dec_conv2(torch.cat([e1,d1],dim=1))) # --> 64
        d3 = self.dec_conv3(torch.cat([e0,d2],dim=1))  # no activation
        return d3
