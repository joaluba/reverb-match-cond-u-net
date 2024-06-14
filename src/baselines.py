import sys
import importlib
# Dereverberation baseline
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
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

class Baseline(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        if self.config["baseline"]=="wpe+fins":
            # load fins config
            fins_project_path="/home/ubuntu/joanna/fins/fins"
            fins_checkpoints_path="/home/ubuntu/Data/RESULTS-VAE-IR/fins/checkpoints/m-240416-111748"
            # load fins config
            fins_config_path = pjoin(fins_project_path,"config.yaml")
            self.fins_config = hlp.load_config_fins(fins_config_path)
            # load fins model
            self.fins_model = model_fins.FilteredNoiseShaper(self.fins_config.model.params)
            state_dicts = torch.load(pjoin(fins_checkpoints_path,"epoch-122.pt"), map_location="cuda")
            self.fins_model.load_state_dict(state_dicts["model_state_dict"])

    def infer_baseline(self,data):

        device= self.config["device"]
        # Function to infer target audio
        with torch.no_grad():
            sContent_in = data[0].to(device)
            sStyle_in=data[1].to(device)
            sTarget_gt=data[2].to(device)
            # step 1: dereverberate content sound
            sContent_in_np = sContent_in.cpu().numpy()
            # Using a list comprehension to apply fit_nara to each slice and collect the results
            results = [fit_nara(sContent_in_np[i,:,:].T) for i in range(sContent_in_np.shape[0])]
            # Extracting the first element of each tuple returned by fit_nara
            sContent_derev = np.array([result[0] for result in results])
            # Transpose back to match the original shape before processing
            sContent_derev = sContent_derev.transpose(0, 2, 1)
            # step 2: estimate RIR of the style
            # Noise for late part
            rir_length = int(self.fins_config.model.params.rir_duration * self.fins_config.model.params.sr)
            stochastic_noise = torch.randn((1, 1, rir_length), device="cuda")
            batch_stochastic_noise = stochastic_noise.repeat(1, self.fins_config.model.params.num_filters, 1)
            # Noise for decoder conditioning
            batch_noise_condition = torch.randn((1, self.fins_config.model.params.noise_condition_length), device="cuda")
            reverberated_source=sStyle_in.unsqueeze(0)
            reverberated_source = audio_fins.audio_normalize_batch(reverberated_source, "rms", self.fins_config.model.params.rms_level)
            predicted_rir = self.fins_model(reverberated_source, batch_stochastic_noise, batch_noise_condition).cpu().numpy()
            # step 3: convolve derev.signal with the prediction
            sTarget_prediction = torch.from_numpy(scipy.signal.fftconvolve(sContent_derev, predicted_rir,mode="full"))[:,:self.sig_len].to(device)
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
