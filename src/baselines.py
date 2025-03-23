import sys
import importlib
# Dereverberation baseline
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from df import enhance, init_df

# Blind RIR estimation baseling
sys.path.append("/home/ubuntu/joanna/reverb-match-cond-u-net/baseline/fins")
import model_fins
import audio_fins
importlib.reload(model_fins)
importlib.reload(audio_fins)
from os.path import join as pjoin
import helpers as hlp
import torch
import numpy as np
import scipy 
from torchaudio.functional import fftconvolve
from torchaudio import save as audiosave

class Baselines(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        # load fins config
        fins_project_path="/home/ubuntu/joanna/fins/fins"
        fins_checkpoints_path="/home/ubuntu/Data/RESULTS-VAE-IR/fins/checkpoints/m-240416-111748"
        # load fins config
        fins_config_path = pjoin(fins_project_path,"config.yaml")
        self.fins_config = hlp.load_config_fins(fins_config_path)
        # load fins model
        self.fins_model = model_fins.FilteredNoiseShaper(self.fins_config.model.params).to(self.config["device"])
        state_dicts = torch.load(pjoin(fins_checkpoints_path,"epoch-122.pt"), map_location=self.config["device"],weights_only=True)
        self.fins_model.load_state_dict(state_dicts["model_state_dict"])
        self.fins_model.eval()
        # load dfnet
        self.modeldf, self.df_state, _ = init_df()  # Load default model

    def infer_baseline(self,data,baseline):

        device= self.config["device"]

        # Function to infer target audio
        with torch.no_grad():
            sContent_in = hlp.unsqueezeif2D(data[0]).to(device) # (batch_size, num_channels, signal_length)
            sStyle_in=hlp.unsqueezeif2D(data[1]).to(device) # (batch_size, num_channels, signal_length)
            sTarget_gt=hlp.unsqueezeif2D(data[2]).to(device) # (batch_size, num_channels, signal_length)
            sAnecho=hlp.unsqueezeif2D(data[3]).to(device) # (batch_size, num_channels, signal_length)

            batch_size=sContent_in.shape[0]

            # ----- step 1: dereverberate content sound -----
            if baseline == "wpe+fins":
                sContent_in_np = sContent_in.cpu().numpy()
                # Using a list comprehension to apply fit_nara to each slice and collect the results
                results = [fit_nara_online(sContent_in_np[i,:,:].T) for i in range(sContent_in_np.shape[0])] 
                # Extracting the first element of each tuple returned by fit_nara
                sContent_derev = np.array([result[0] for result in results])
                # Transpose back to match the original shape before processing
                sContent_derev = torch.tensor(sContent_derev,dtype=torch.float32).unsqueeze(1).to(device) # (batch_size, num_channels, signal_length)
                sContent_derev=torch.stack([hlp.torch_normalize_max_abs(sContent_derev[i,:,:]) for i in range(batch_size)])
            elif baseline == "anecho+fins":
                sContent_derev=sAnecho
            elif baseline == "dfnet+fins":
                sContent_derev = enhance(self.modeldf, self.df_state, sContent_in.squeeze(1).cpu()).unsqueeze(1)
                sContent_derev=torch.stack([hlp.torch_normalize_max_abs(sContent_derev[i,:,:]) for i in range(batch_size)]).to(device)
            else: 
                print("Baseline" + baseline + " is not implemented")
            
            # ----- step 2: estimate RIR of the style -----
            # Noise for late part
            rir_length = int(self.fins_config.model.params.rir_duration * self.fins_config.model.params.sr)

            # Noise for decoder conditioning
            batch_stochastic_noise = torch.randn((batch_size, self.fins_config.model.params.num_filters, rir_length), device="cuda")
            batch_noise_condition = torch.randn((batch_size, self.fins_config.model.params.noise_condition_length), device="cuda")
            reverberated_source = audio_fins.audio_normalize_batch(sStyle_in, "rms", self.fins_config.model.params.rms_level) # (batch_size, num_channels, signal_length)
            predicted_rir = self.fins_model(reverberated_source, batch_stochastic_noise, batch_noise_condition) # (batch_size, num_channels, ir_length)
            predicted_rir=torch.stack([hlp.truncate_ir_silence(predicted_rir[i,:,:], 48000, threshold_db=20) for i in range(batch_size)])
            predicted_rir=torch.stack([hlp.torch_normalize_max_abs(predicted_rir[i,:,:]) for i in range(batch_size)])

            # ----- step 3: convolve derev.signal with the predicted rir in each batch  ---- 
            sTarget_prediction = torch.stack([fftconvolve(sContent_derev[i].unsqueeze(0), predicted_rir[i].unsqueeze(0), mode="full").squeeze(0) for i in range(batch_size)])

            # cut to the length of original signal
            sTarget_prediction=sTarget_prediction[:,:,:sTarget_gt.shape[2]]
            # normalize predicted output
            sTarget_prediction=torch.stack([hlp.torch_normalize_max_abs(sTarget_prediction[i,:,:]) for i in range(batch_size)])
            # synchronize predicted output with anechoic signal for each batch
            sTarget_prediction= torch.stack([hlp.synch_sig2(sAnecho[i,:,:].cpu(),sTarget_prediction[i,:,:].cpu())[1] for i in range(batch_size)]).to(device)

            return sContent_in, sStyle_in, sTarget_gt, sTarget_prediction
    
def fit_nara_online(y):
    # grab all the nara params
    size = 512
    shift = 256
    taps = 10
    delay = 1
    alpha = 0.9999
    channel = 1
    frequency_bins = size // 2 + 1

    Y = stft(y.T, size=size, shift=shift).transpose(1, 2, 0)
    T, _, _ = Y.shape

    def aquire_framebuffer():
        # buffer init with zeros so output is time alligned
        buffer = list(np.zeros((taps + delay, Y.shape[1], Y.shape[2]), dtype=Y.dtype))
        # buffer = list(Y[: taps + delay, :, :])
        # for t in range(taps + delay + 1, T):

        for t in range(0, T):

            buffer.append(Y[t, :, :])
            yield np.array(buffer)
            buffer.pop(0)

    Z_list = []
    online_wpe = OnlineWPE(
        taps=taps,
        delay=delay,
        alpha=alpha,
        channel=channel,
        frequency_bins=frequency_bins,
    )

    for Y_step in aquire_framebuffer():
        Z_list.append(online_wpe.step_frame(Y_step))

    Z = np.stack(Z_list)
    z = istft(
        np.asarray(Z).transpose(2, 0, 1),
        size=size,
        shift=shift,
    )

    # return the output for the first channel
    return z[0], Y, Z
