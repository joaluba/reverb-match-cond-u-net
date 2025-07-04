import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

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
# ------------------------------------ REVERB ENCODER NETWORK --------------------------------------
# --------------------------------------------------------------------------------------------------

class ReverbEncoderBlock(nn.Module):
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

class ReverbEncoder(nn.Module):
    # -------- Encoder: --------
    def __init__(self, args):
        super().__init__() 
        # constants
        self.sig_len=args.sig_len
        self.z_len=args.enc_len
        self.N_layers=args.n_layers_revenc
        # internal parameters of the network:
        kernel_size=15
        stride=2
        padding= (kernel_size - 1) // 2

        # convolutional layers are a series of encoder blocks with increasing channels
        conv_layers = []
        block_channels=1
        x_len=self.sig_len
        for i in range(self.N_layers):
            conv_layers.append(ReverbEncoderBlock(block_channels, block_channels*2, kernel_size=15, stride=stride,padding=padding))
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
                                 nn.Linear(int(block_channels/2),self.z_len))
        
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
# -------------------------------------- REVERB TRANSFER NETWORK: ----------------------------------
# --------------------------------------------------------------------------------------------------

class DownSamplingLayer(nn.Module):
    # ------- Definition of downsampling (encoder) block ------
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7, z_channels=512):
        super().__init__()
        self.conv=nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        self.bn=nn.BatchNorm1d(channel_out)
        self.film=FiLM(z_channels,channel_out)
        self.acti=nn.PReLU()

    def forward(self, x, z):
        x=self.conv(x)
        x=self.bn(x)
        x=self.film(x,z)
        x=self.acti(x)
        return x 

class UpSamplingLayer(nn.Module):
    # ------- Definition of upsampling (decoder) block ------
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2,z_channels=512):
        super().__init__()
        self.conv=nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding)
        self.bn=nn.BatchNorm1d(channel_out)
        self.film=FiLM(z_channels,channel_out)
        self.acti=nn.PReLU()

    def forward(self, x, z):
        x=self.conv(x)
        x=self.bn(x)
        x=self.film(x,z)
        x=self.acti(x)
        return x

class waveunet(nn.Module):
    # ------- Definition of a wave-u-net encoder-decoder ------
    def __init__(self, args):
        super().__init__()
        # constants:
        self.n_layers = args.n_layers_waveunet
        self.z_channels = args.enc_len 
        self.channels_interval = 24
        # encoder:
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        # 1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                    z_channels=self.z_channels)
                )
            

        # layers in the middle:
        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # decoder:
        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                    z_channels=self.z_channels)
            )

        # final output layer:
        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x, z_enc,z_dec):
        # x - input waveform
        # o - waveform flowing through the net
        # z_enc - conditioning vector for decoder
        # z_dec - conditioning vector for decoder
        skipfeats = []
        o = x

        # Up Sampling
        for i in range(self.n_layers):
            # encoder layer
            o = self.encoder[i](o,z_enc)
            # store skip features for later
            skipfeats.append(o)
            # decimate, [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        # middle layer:
        o = self.middle(o)

        # Down Sampling
        for i in range(self.n_layers):
            # interpolate, [batch_size, T * 2, channels]:
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # concatenate with skip features
            o = torch.cat([o, skipfeats[self.n_layers - i - 1]], dim=1)
            # decoder layer 
            o = self.decoder[i](o,z_dec)
            

        # concatenate output with input
        o = torch.cat([o, x], dim=1)
        # final layer with tanh activation
        o = self.out(o)
        return o




  