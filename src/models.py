import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import helpers as hlp

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
    def __init__(self, config):
        super().__init__() 
        # constants
        self.sig_len=config["sig_len"]
        self.z_len=config["rev_emb_len"]
        self.N_layers=config["n_blocks_revenc"]
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
# -------------------------------- (STANDARD & VARIATIONAL WAVE-U-NET) -----------------------------
# --------------------------------------------------------------------------------------------------

class waveunet_DownSamplingLayer(nn.Module):
    # ------- Downsampling (encoder) block ------
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

class waveunet_UpSamplingLayer(nn.Module):
    # ------- Upsampling (decoder) block ------
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
    # ------- Standard wave-u-net encoder-decoder ------
    def __init__(self, config):
        super().__init__()
        # constants:
        self.n_layers = config["n_blocks_enc"]
        self.z_channels = config["rev_emb_len"]
        self.channels_interval = 24
        self.symmetric_film = bool(config["symmetric_film"])
        # encoder:
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        # [1, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        # [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288]
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                waveunet_DownSamplingLayer(
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
                waveunet_UpSamplingLayer(
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
        # z_enc - conditioning vector for encoder
        # z_dec - conditioning vector for decoder
        skipfeats = []
        o = x

        # Down Sampling
        for i in range(self.n_layers):
            # encoder layer
            o = self.encoder[i](o,z_enc)
            # store skip features for later
            skipfeats.append(o)
            # decimate, [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        # middle layer:
        o = self.middle(o)

        # Up Sampling
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
    

class varwaveunet(nn.Module):
    # ------- Variational wave-u-net encoder-decoder ------
    def __init__(self, config):
        super().__init__()
        # constants:
        self.n_layers = config["n_blocks_enc"]
        self.z_channels = config["rev_emb_len"]
        self.channels_interval = 24
        self.h_channels = config["gauss_len"]
        self.symmetric_film = bool(config["symmetric_film"])
        self.sig_len = config["sig_len"]
        # encoder:
        encoder_in_channels_list = [1] + [int(i * self.channels_interval) for i in range(1, self.n_layers)]
        # [1, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264]
        encoder_out_channels_list = [int(i * self.channels_interval) for i in range(1, self.n_layers + 1)]
        # [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288]
        list_out_lenghts = [int(self.sig_len/(np.power(2,i))) for i in range(1, self.n_layers+1)]
        # [49152, 24576, 12288, 6144, 3072, 1536, 768, 384, 192, 96, 48]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                waveunet_DownSamplingLayer(
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
                waveunet_UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                    z_channels=self.z_channels)
            )

        self.flatten = nn.AdaptiveAvgPool1d(1)
        self.fc_mu = nn.Linear(encoder_out_channels_list[-1], self.h_channels)
        self.fc_var = nn.Linear(encoder_out_channels_list[-1], self.h_channels)
        self.decoder_input = nn.Linear(self.h_channels,encoder_out_channels_list[-1])
        self.unflatten = nn.Conv1d(1,24,kernel_size=1, stride=1, padding=0)

        # final output layer:
        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )
    
    def reparameterize(self, mu, log_var):
        # Reparameterization takes in the input mu and logVar 
        # and samples the mu + std * eps
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def myflatten(self,x):
        x_dims=x.shape
        n_batches=x_dims[0]
        unflat_dims=x_dims[1:]
        x=x.view(n_batches, -1)
        return x, unflat_dims
        
    def myunflatten(self,x,unflat_dims):
        dim_final_unflat= tuple([x.size(0)])+tuple(unflat_dims)
        x=x.view(dim_final_unflat)
        return x

    def forward(self, x, z_enc,z_dec):
        # x - input waveform
        # o - waveform flowing through the net
        # z_enc - conditioning vector for decoder
        # z_dec - conditioning vector for decoder
        skipfeats = []
        o = x

        # Down Sampling
        for i in range(self.n_layers):
            # encoder layer
            o = self.encoder[i](o,z_enc)
            # store skip features for later
            skipfeats.append(o)
            # decimate, [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        # middle layer:
        o = self.middle(o)

        # --------- Variational bottleneck: ---------
        # flatten the output of the previous layer:
        o = self.flatten(o)
        o = o.view(o.shape[0],1,-1)
        # generate mean and standard deviation 
        # (from z_channels to h_channels)
        mu = self.fc_mu(o)
        log_var = self.fc_var(o)
        # generate a sample from gaussian noise
        o = self.reparameterize(mu, log_var)
        # go from h_channels to z_channels
        o = self.decoder_input(o)
        # unflatten the output of the previous layer:
        o = self.unflatten(o)
        o = o.view(o.shape[0],o.shape[2],o.shape[1])
    
        # Up Sampling
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
        return o, mu, log_var
    

# --------------------------------------------------------------------------------------------------
# -------------------------------------- REVERB TRANSFER NETWORK: ----------------------------------
# ---------------------------------- (ENCODER AND DECODER FROM FINS) -------------------------------
# --------------------------------------------------------------------------------------------------

class fins_UpsampleNet(nn.Module):
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


class fins_EncBlock(nn.Module):
    # -------- Fins encoder block + film conditioning: --------
    def __init__(self, in_channels, out_channels, z_channels, kernel_size=15, stride=2, padding=0):
        
        super().__init__()
        # direct connection
        self.direct_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.direct_bn = nn.BatchNorm1d(out_channels)
        self.direct_film=FiLM(z_channels,out_channels)
        self.direct_prelu = nn.PReLU()
        # residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.residual_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, z):
        # residual connection
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)
        # direct connection
        x = self.direct_conv(x)
        x = self.direct_bn(x)
        x = self.direct_film(x,z)
        x = self.direct_prelu(x)
        # add outputs of direct and residual connections
        x += residual
        return x
    

class fins_encoder(nn.Module):
    # -------- Fins encoder: --------
    def __init__(self, x_len=98304, z_len=128, l_len=567, N_layers=14):
        super().__init__() 

        # internal parameters of the network:
        kernel_size=15
        stride=2
        padding= (kernel_size - 1) // 2

        # convolutional layers are a series of encoder blocks with increasing channels
        self.conv_layers = nn.ModuleList([])
        block_channels=1
        for i in range(N_layers):
            self.conv_layers.append(fins_EncBlock(block_channels, block_channels*2, z_len, kernel_size=15, stride=stride,padding=padding))
            block_channels*=2
            # compute heigth of the ouput (width=1,depth=block_channels)
            x_len=np.floor((x_len-kernel_size+2*padding)/stride)+1 

        # adaptive pooling layer to flatten and aggregate information
        self.aggregate = nn.AdaptiveAvgPool1d(1)
    
        # final mlp layers
        self.mlp = nn.Sequential(nn.Linear(block_channels, block_channels),
                                 nn.Linear(block_channels,int(block_channels/2)),
                                 nn.Linear(int(block_channels/2),l_len))
        
    def forward(self, x, z):
        # Convolutional residual blocks:
        for layer in self.conv_layers:
            x = layer(x,z)
        # Aggregate info & flatten:
        x = self.aggregate(x)
        x = x.view([x.shape[0],1,-1]) 
        # Dense layers after aggregation:
        x = self.mlp(x)
        return x


class fins_DecBlock(nn.Module):
    # -------- Fins decoder block + film conditioning: --------
    def __init__(self,
                 in_channels,
                 out_channels,
                 z_channels,
                 upsample_factor,
                 device):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.upsample_factor = upsample_factor
        self.device=device

        # Decoder block, stage A
        self.A_direct_bn1=nn.BatchNorm1d(in_channels)
        self.A_direct_film1=FiLM(z_channels,in_channels)
        self.A_direct_prelu1=nn.PReLU()
        self.A_direct_upsmpl=fins_UpsampleNet(in_channels, in_channels, upsample_factor)
        self.A_direct_conv=customConv1d(in_channels, out_channels, kernel_size=3)
        self.A_direct_bn2=nn.BatchNorm1d(out_channels)
        self.A_direct_film2=FiLM(z_channels,out_channels)
        self.A_direct_prelu2=nn.PReLU()

        self.A_residual_upsmpl=fins_UpsampleNet(in_channels, in_channels, upsample_factor)
        self.A_residual_conv=customConv1d(in_channels, out_channels, kernel_size=1)

        # Decoder block, stage B
        self.B_direct_bn1=nn.BatchNorm1d(out_channels)
        self.B_direct_film1=FiLM(z_channels,out_channels)
        self.B_direct_prelu1=nn.PReLU()
        self.B_direct_dilconv1=customConv1d(out_channels, out_channels, kernel_size=3, dilation=4)
        self.B_direct_bn2=nn.BatchNorm1d(out_channels)
        self.B_direct_film2=FiLM(z_channels,out_channels)
        self.B_direct_prelu2=nn.PReLU()
        self.B_direct_dilconv2=customConv1d(out_channels, out_channels, kernel_size=3, dilation=8)

    def concat_noise(self,z):
        n=torch.randn(z.shape).to(self.device)
        zn = torch.cat((z, n), dim=-1)
        return zn

    def forward(self, x_in, z):
        # block A - direct connection
        x=x_in
        x=self.A_direct_bn1(x)
        x=self.A_direct_film1(x,self.concat_noise(z))
        x=self.A_direct_prelu1(x)
        x=self.A_direct_upsmpl(x)
        x=self.A_direct_conv(x)
        x=self.A_direct_bn2(x)
        x=self.A_direct_film2(x,self.concat_noise(z))
        x=self.A_direct_prelu2(x)
        # block A - residual connection
        res=self.A_residual_upsmpl(x_in)
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

class fins_decoder(nn.Module):
    # -------- Decoder: --------
    def __init__(self,
                 in_channels=1,
                 z_len=128,
                 l_len=512,
                 device="cuda",
                 N_layers=7):
        super().__init__()

        self.in_channels = in_channels
        self.z_channels = z_len
        self.l_len = l_len
        self.device = device
        self.n_layers = N_layers


        self.preprocess = customConv1d(in_channels, 512, kernel_size=3)

        if self.n_layers==7:
            self.gblocks = nn.ModuleList ([
                fins_DecBlock(512, 512, z_len, 1,self.device),
                fins_DecBlock(512, 512, z_len, 1,self.device),
                fins_DecBlock(512, 256, z_len, 2,self.device),
                fins_DecBlock(256, 256, z_len, 2,self.device),
                fins_DecBlock(256, 256, z_len, 2,self.device),
                fins_DecBlock(256, 128, z_len, 3,self.device),
                fins_DecBlock(128, 64, z_len,8,self.device)
            ])
        elif self.n_layers==5:
            self.gblocks = nn.ModuleList ([
                fins_DecBlock(512, 512, z_len, 1,self.device),
                fins_DecBlock(512, 256, z_len, 3,self.device),
                fins_DecBlock(256, 256, z_len, 4,self.device),
                fins_DecBlock(256, 128, z_len, 4,self.device),
                fins_DecBlock(128, 64, z_len,4,self.device)
            ])
        else:
            "a decoder with this number of blocks is not yet implemented"
        
        self.postprocess = nn.Sequential(
            customConv1d(64, 1, kernel_size=3),
            nn.Tanh()
        )

    def forward(self, inputs, z):
        inputs = self.preprocess(inputs)
        outputs = inputs
        for (i, layer) in enumerate(self.gblocks):
            outputs = layer(outputs, z)
        outputs = self.postprocess(outputs)
        return outputs


class fins_encdec(nn.Module):
    # -------- Encoder-Decoder: --------
    def __init__(self, config):
        super().__init__()

        self.sig_len = config["sig_len"] #input waveform
        self.l_len = config["btl_len"] # bottleneck embedding
        self.z_len = config["rev_emb_len"] # conditioning vector
        self.n_layers = config["n_blocks_enc"] # number of encoder blocks
        self.device = config["device"]

        self.encode = fins_encoder(x_len=self.sig_len, l_len=self.l_len, N_layers=self.n_layers)
        self.decode = fins_decoder(in_channels=1,z_len=self.z_len*2,device=self.device)

    def forward(self,x,z_enc,z_dec):
        x = self.encode(x,z_enc)
        x = self.decode(x,z_dec)
        return x


class cond_reverb_transfer(nn.Module):
    # ------- Full conditional architecture ------
    def __init__(self, autoencoder,conditioning_network):
        super(cond_reverb_transfer, self).__init__()
        self.autoencoder = autoencoder
        self.conditioning_network = conditioning_network

    def forward(self, content, style):
        style_emb = self.conditioning_network(style)
        content_emb = self.conditioning_network(content)
        output = self.autoencoder(content,content_emb,style_emb)
        return output



if __name__ == "__main__":
# ---- test definitions ----
    
    # load default parameters
    config = hlp.load_config("basic.yaml")

    # specify parameters of the model
    config["fs"] = 48000
    config["sig_len"] = 98304
    config["rev_emb_len"] = 128
    config["btl_len"] = 512
    config["n_blocks_revenc"] = 12
    config["n_blocks_enc"] = 12
    config["gauss_len"] = 5
    config["device"]="cpu"

    # create random tensor with the size of the expected content signal
    s_content=torch.randn(8,1,config["sig_len"]).to(config["device"])
    # create random tensor with the size of the expected style signal
    s_style=torch.randn(8,1,config["sig_len"]).to(config["device"])
    reverb_emb=torch.randn(8,1,config["rev_emb_len"]).to(config["device"])
    v_bottleneck=torch.randn(8,1,config["btl_len"]).to(config["device"])
    

    # # check reverb encoder
    # model=ReverbEncoder(config).to(config["device"])
    # model.eval()
    # reverb_emb=model(s_style)
    # summary(model,(1, config["sig_len"]),device=config["device"])# torch summary expects 2 dim input for 1d conv
    # print(f"reverb encoder network input shape: {s_style.shape}")
    # print(f"reverb encoder network output shape: {reverb_emb.shape}")

    # # check waveunet 
    # model=waveunet(config).to(config["device"])
    # model.eval()
    # s_target=model(s_content,reverb_emb,reverb_emb)
    # summary(model,[(1, config["sig_len"]),(1, config["rev_emb_len"]),(1, config["rev_emb_len"])],device=config["device"]) # torch summary expects 2 dim input for 1d conv
    # print(f"waveunet input shape: {s_content.shape}")
    # print(f" output shape: {s_target.shape}")

    # # check varwaveunet 
    # model=varwaveunet(config).to(config["device"])
    # model.eval()
    # s_target, mu, log_var =model(s_content,reverb_emb,reverb_emb)
    # summary(model,[(1, config["sig_len"]),(1, config["rev_emb_len"]),(1, config["rev_emb_len"])],device=config["device"]) # torch summary expects 2 dim input for 1d conv
    # print(f"variational waveunet input shape: {s_content.shape}")
    # print(f" output shape: {s_target.shape}") # y_wave contains prediction, mean, std

    # # check fins encoder 
    # model= fins_encoder(x_len=config["sig_len"], l_len=config["btl_len"], N_layers=config["n_blocks_revenc"]).to(config["device"])
    # model.eval()
    # v_bottleneck=model(s_content,reverb_emb)
    # summary(model,[(1, config["sig_len"]),(1, config["rev_emb_len"])],device=config["device"]) # torch summary expects 2 dim input for 1d conv
    # print(f"fins encoder input shape: {s_content.shape}")
    # print(f" output shape: {v_bottleneck.shape}")

    # # check fins decoder 
    # model=fins_decoder(in_channels=1,z_len=config["rev_emb_len"]*2,device=config["device"],N_layers=5).to(config["device"])
    # model.eval()
    # s_target=model(v_bottleneck,reverb_emb)
    # summary(model,[(1, config["btl_len"]),(1, config["rev_emb_len"])],device=config["device"]) # torch summary expects 2 dim input for 1d conv
    # print(f"fins decoder input shape: {s_content.shape}")
    # print(f" output shape: {s_target.shape}")


    # # check fins encoder-decoder  
    # model=fins_encdec(config).to(config["device"])
    # s_target=model(s_content,reverb_emb,reverb_emb)
    # summary(model,[(1, config["sig_len"]),(1, config["rev_emb_len"]),(1, config["rev_emb_len"])],device=config["device"]) # torch summary expects 2 dim input for 1d conv
    # print(f"fins enc-dec input shape: {s_content.shape}")
    # print(f" output shape: {s_target.shape}")


    # check full conditional encoder-decoder model
    cond_generator = ReverbEncoder(config)
    autoencoder = fins_encdec(config)
    model=cond_reverb_transfer(autoencoder,cond_generator).to(config["device"])
    s_style=model(s_content,s_style)
    summary(model,[(1, config["sig_len"]),(1, config["sig_len"])],device=config["device"]) # torch summary expects 2 dim input for 1d conv
    print(f"fins enc-dec input shape: {s_content.shape}")
    print(f" output shape: {s_style.shape}")


  