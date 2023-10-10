import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
# import torchdataset_prep as dsprep
from torchsummary import summary
import argparse
import helpers

# --------------------------------------------------------------------------------------------------
# -------------------------------------------- ENCODER: --------------------------------------------
# --------------------------------------------------------------------------------------------------


class EncBlock(nn.Module):
    # -------- One encoder block: --------
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=2, padding=0):
        super().__init__()
        # direct connection
        self.direct_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.direct_bn = nn.BatchNorm1d(out_channels)
        self.direct_prelu = nn.PReLU()
        # residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.residual_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # residual connection
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)
        # direct connection
        x = self.direct_conv(x)
        x = self.direct_bn(x)
        x = self.direct_prelu(x)
        # add outputs of direct and residual connections
        x += residual
        return x
    

class sig2ir_encoder(nn.Module):
    # -------- Encoder: --------
    def __init__(self, x_len=16000*3, l_len=512, N_layers=9):
        super().__init__() 

        # internal parameters of the network:
        kernel_size=15
        stride=2
        padding= (kernel_size - 1) // 2

        # convolutional layers are a series of encoder blocks with increasing channels
        conv_layers = []
        block_channels=1
        for i in range(N_layers):
            conv_layers.append(EncBlock(block_channels, block_channels*2, kernel_size=15, stride=stride,padding=padding))
            block_channels*=2
            # compute heigth of the ouput (width=1,depth=block_channels)
            x_len=np.floor((x_len-kernel_size+2*padding)/stride)+1 

        self.conv_layers = nn.Sequential(*conv_layers)

        # adaptive pooling layer to flatten and aggregate information
        self.aggregate = nn.AdaptiveAvgPool1d(1)
        #self.aggregate = nn.Conv1d(block_channels, z_len, kernel_size=int(x_len), stride=1, padding=0)
    
        # final mlp layers
        self.mlp = nn.Sequential(nn.Linear(block_channels, block_channels),
                                 nn.Linear(block_channels,int(block_channels/2)),
                                 nn.Linear(int(block_channels/2),l_len))
        
    def forward(self, x):
        # Convolutional residual blocks:
        x = self.conv_layers(x)
        # Aggregate info & flatten:
        x = self.aggregate(x)
        x = x.view([x.shape[0],1,-1]) 
        # Dense layers after aggregation:
        x = self.mlp(x)
        return x

# --------------------------------------------------------------------------------------------------
# -------------------------------------------- DECODER: --------------------------------------------
# --------------------------------------------------------------------------------------------------

class DecBlock(nn.Module):
    # -------- One decoder block: --------
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 z_channels,
                 upsample_factor,
                 device):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.z_channels = z_channels
        self.upsample_factor = upsample_factor
        self.device=device

        # Decoder block, stage A
        self.A_direct_bn1=nn.BatchNorm1d(in_channels)
        self.A_direct_film1=FiLM(z_channels,in_channels)
        self.A_direct_prelu1=nn.PReLU()
        self.A_direct_upsmpl=UpsampleNet(in_channels, in_channels, upsample_factor)
        self.A_direct_conv=customConv1d(in_channels, hidden_channels, kernel_size=3)
        self.A_direct_bn2=nn.BatchNorm1d(hidden_channels)
        self.A_direct_film2=FiLM(z_channels,hidden_channels)
        self.A_direct_prelu2=nn.PReLU()

        self.A_residual_upsmpl=UpsampleNet(in_channels, in_channels, upsample_factor)
        self.A_residual_conv=customConv1d(in_channels, hidden_channels, kernel_size=1)

        # Decoder block, stage B
        self.B_direct_bn1=nn.BatchNorm1d(hidden_channels)
        self.B_direct_film1=FiLM(z_channels,hidden_channels)
        self.B_direct_prelu1=nn.PReLU()
        self.B_direct_dilconv1=customConv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=4)
        self.B_direct_bn2=nn.BatchNorm1d(hidden_channels)
        self.B_direct_film2=FiLM(z_channels,hidden_channels)
        self.B_direct_prelu2=nn.PReLU()
        self.B_direct_dilconv2=customConv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=8)

    def concat_noise(self,z):
        n=torch.randn(z.shape).to(self.device)
        zn = torch.cat((z, n), dim=-1)
        return zn

    def forward(self, input, z):
        # block A - direct connection
        x=input
        x=self.A_direct_bn1(x)
        x=self.A_direct_film1(x,self.concat_noise(z))
        x=self.A_direct_prelu1(x)
        x=self.A_direct_upsmpl(x)
        x=self.A_direct_conv(x)
        x=self.A_direct_bn2(x)
        x=self.A_direct_film2(x,self.concat_noise(z))
        x=self.A_direct_prelu2(x)
        # block A - residual connection
        res=self.A_residual_upsmpl(input)
        res=self.A_residual_conv(res)
        # output of block A - sum of direct and residual
        x_a = res + x
        # block B - direct connection
        x_b=self.B_direct_bn1(x_a)
        x_b=self.B_direct_film1(x_b,self.concat_noise(z))
        x_b=self.B_direct_prelu1(x_b)
        x_b=self.B_direct_dilconv1(x_b)
        x_b=self.B_direct_bn2(x_b)
        x_b=self.B_direct_film2(x_b,self.concat_noise(z))
        x_b=self.B_direct_prelu2(x_b)
        x_b=self.B_direct_dilconv2(x_b)
        # output of block A - sum of direct and residual
        output = x_a + x_b

        return output

class sig2ir_decoder(nn.Module):
    # -------- Decoder: --------
    def __init__(self,
                 in_channels=567,
                 z_len=128,
                 device="cpu"):
        super().__init__()

        self.in_channels = in_channels
        self.z_channels = z_len
        self.device = device

        self.preprocess = customConv1d(in_channels, 768, kernel_size=3)
        self.gblocks = nn.ModuleList ([
            DecBlock(768, 768, z_len, 1,self.device),
            DecBlock(768, 768, z_len, 1,self.device),
            DecBlock(768, 384, z_len, 2,self.device),
            DecBlock(384, 384, z_len, 2,self.device),
            DecBlock(384, 384, z_len, 2,self.device),
            DecBlock(384, 192, z_len, 3,self.device),
            DecBlock(192, 96, z_len, 5,self.device)
        ])
        self.postprocess = nn.Sequential(
            customConv1d(96, 1, kernel_size=3),
            nn.Tanh()
        )

    def forward(self, inputs, z):
        inputs = self.preprocess(inputs)
        outputs = inputs
        for (i, layer) in enumerate(self.gblocks):
            outputs = layer(outputs, z)
        outputs = self.postprocess(outputs)
        return outputs

class UpsampleNet(nn.Module):
    # ----- Upsampling with transposed 1d convolutions -----
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor):

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer = nn.ConvTranspose1d(input_size, output_size, upsample_factor * 2,
                                   upsample_factor, padding=upsample_factor // 2)
        nn.init.orthogonal_(layer.weight)
        self.layer = nn.utils.spectral_norm(layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs

class customConv1d(nn.Module):
    # ---- Conv1d for spectral normalisation and orthogonal initialisation ----
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        pad = dilation * (kernel_size - 1) // 2

        layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=pad, dilation=dilation, groups=groups)
        nn.init.orthogonal_(layer.weight)
        self.layer = nn.utils.spectral_norm(layer)

    def forward(self, inputs):
        return self.layer(inputs)
    

# --------------------------------------------------------------------------------------------------
# ----------------------------------------- CONDITIONING: ------------------------------------------
# --------------------------------------------------------------------------------------------------

class FiLM(nn.Module):
    # -------- Definition of a FiLM layer: --------
    def __init__(self, zdim, n_channels):
        super().__init__()
        self.gamma = nn.Linear(zdim, n_channels)   
        self.beta = nn.Linear(zdim, n_channels)   

    def forward(self, x, z):
        
        # gamma_shaped = torch.tile(self.gamma(z).permute(0,2,1),(1,1,x.shape[-1]))
        # beta_shaped = torch.tile(self.beta(z).permute(0,2,1),(1,1,x.shape[-1]))
        gamma_shaped=self.gamma(z).permute(0,2,1).repeat(1,1,x.shape[-1])
        beta_shaped=self.beta(z).permute(0,2,1).repeat(1,1,x.shape[-1])
        assert gamma_shaped.shape==x.shape, f"wrong gamma shape"
        assert beta_shaped.shape==x.shape, f"wrong beta shape"
        x = gamma_shaped * x + beta_shaped

        return x
    
# --------------------------------------------------------------------------------------------------
# ----------------------------------------- FULL NETWORK: ------------------------------------------
# --------------------------------------------------------------------------------------------------

class sig2ir_encdec(nn.Module):
    # -------- Encoder-Decoder: --------
    def __init__(self, 
                 sig_len=int(48000*2.4), 
                 l_len=128, 
                 v_len=400, 
                 z_len=128*2, 
                 ir_len=48000,
                 device="cpu"):
        super().__init__()

        self.sig_len=sig_len #input waveform
        self.l_len=l_len #bottleneck embedding
        self.v_len=v_len #input sequence to decoder 
        self.z_len=z_len #conditioning vector
        self.ir_len=ir_len # output waveform
        self.device=device

        self.encode=sig2ir_encoder(x_len=sig_len, l_len=l_len, N_layers=9)
        self.decode=sig2ir_decoder(in_channels=1,z_len=z_len,device=self.device)
        self.trainable_v = nn.Parameter(torch.randn(1,1,v_len),requires_grad=True)

    def forward(self,x):
        l=self.encode(x)
        v=self.trainable_v.repeat((x.shape[0],1,1)).to(self.device)
        ir=self.decode(v,l)
        return ir



if __name__ == "__main__":
    # ---- check if the model definitions are correct ----
    
    # example input tensor
    FS=48000
    sig_len=int(2.4*FS)
    l_len=512
    v_len=400 
    z_len=512*2
    ir_len=FS
    
    x_wave=torch.randn(1,1,sig_len)

    # check encoder
    encoder=sig2ir_encoder(x_len=sig_len,l_len=l_len,N_layers=9)
    encoder.to("cpu")
    summary(encoder,input_size = (1, sig_len), batch_size = -1, device="cpu")# torch summary expects 2 dim input for 1d conv
    encoder.eval()
    l=encoder(x_wave)
    print(f"encoder input shape: {x_wave.shape}")
    print(f"encoder output shape: {l.shape}")

    # check decoder
    decoder=sig2ir_decoder(in_channels=1,z_len=z_len)
    decoder.to("cpu")
    decoder.eval()
    v=torch.nn.Parameter(torch.randn(1,1,v_len)) 
    y=decoder(v,l)
    print(f"decoder input shape: {l.shape}")
    print(f"output shape: {y.shape}")

    # check enc-dec
    model=sig2ir_encdec(sig_len=sig_len, l_len=l_len, v_len=v_len, z_len=z_len, ir_len=ir_len,device="cpu")
    model.to("cpu")
    model.eval
    ir=model(x_wave)
    print(f"encdec input shape: {x_wave.shape}")
    print(f"encdec output shape: {ir.shape}")


