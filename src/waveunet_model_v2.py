import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# --------------------------------------------------------------------------------------------------
# -------------------------------------------- ENCODER: --------------------------------------------
# --------------------------------------------------------------------------------------------------

class DownSamplingLayer(nn.Module):
    # -------- Definition of a downsampling layer: --------
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)
    
class waveunet_encoder(nn.Module):  
    # -------- Wave-U-Net Encoder: --------
    def __init__(self,
                 n_layers=12,
                 channels_interval=24):
        super().__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval

        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        #  1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

    def forward(self,input):
        o=input
        skipfeats = []
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            skipfeats.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2] 
        return o, skipfeats
    

# --------------------------------------------------------------------------------------------------
# -------------------------------------------- DECODER: --------------------------------------------
# --------------------------------------------------------------------------------------------------

class UpSamplingLayer(nn.Module):
    # -------- Definition of an upsampling layer: --------
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)
    

class waveunet_decoder(nn.Module):  
    # -------- Wave-U-Net Decoder: --------
    def __init__(self,
                n_layers=12,
                channels_interval=24):
        super().__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

    def forward(self,input,skipfeat):
        o=input
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, skipfeat[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)
        return o
    

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

class waveunet_encdec(nn.Module):
    # -------- Definition of a full wave-u-net: --------
    def __init__(self, n_layers=12, channels_interval=24):
        super().__init__()

        # Constants:
        self.n_layers = n_layers
        self.channels_interval = channels_interval

        # Encoder (downsampling):
        self.encoder=waveunet_encoder(self.n_layers,self.channels_interval)

        # Middle Layer:
        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # Decoder (upsampling):
        self.decoder=waveunet_decoder(self.n_layers,self.channels_interval)

        # Output Layer:
        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        o = input
        # encoder:
        o, skipfeat =self.encoder(o)
        # middle:
        o = self.middle(o)
        # decoder:
        o = self.decoder(o,skipfeat)
        # concatenate with input:
        o = torch.cat([o, input], dim=1)
        # final layer:
        o = self.out(o)
        return o
    
if __name__ == "__main__":
    # ---- check if the model definitions are correct ----
    
    # example input tensor
    FS=48000
    sig_len=16384 #int(2*FS)
    l_len=512
    v_len=400 
    z_len=512*2
    ir_len=FS
    
    x_wave=torch.randn(1,1,sig_len)

    # check encoder
    encoder=waveunet_encoder(n_layers=12,channels_interval=24)
    encoder.to("cpu")
    encoder.eval()
    l, skipfeats =encoder(x_wave)
    print(f"encoder input shape: {x_wave.shape}")
    print(f"encoder output shape: {l.shape}")

    # check decoder
    decoder=waveunet_decoder(n_layers=12,channels_interval=24)
    decoder.to("cpu")
    decoder.eval()
    y=decoder(l,skipfeats)
    print(f"decoder input shape: {l.shape}")
    print(f"decoder output shape: {y.shape}")

    # check enc-dec
    model=waveunet_encdec(n_layers=12,channels_interval=24)
    model.to("cpu")
    model.eval
    y_wave=model(x_wave)
    summary(model,input_size = (1, sig_len), batch_size = -1, depth=3)# torch summary expects 2 dim input for 1d conv
    print(f"encdec input shape: {x_wave.shape}")
    print(f"encdec output shape: {y_wave.shape}")
    
