from acoustics.room import t60_impulse, c50_from_file
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torchaudio
import librosa
import scipy.signal as signal
from masp import shoebox_room_sim as srs
from os.path import join as pjoin
import colorednoise
import random
import yaml
from fft_conv_pytorch import fft_conv

def place_on_circle(head_pos,r,angle_deg):
# place a source around the reference point (like head)
    angle_rad = (90-angle_deg) * (np.pi / 180)
    x_coord=head_pos[0]+r*np.sin(angle_rad)
    y_coord=head_pos[1]+r*np.cos(angle_rad)
    src_pos=np.array([x_coord, y_coord, head_pos[2]]) 
    return src_pos

def compute_ir_stats(filename,bands):    
    # Compute stats based on the RIR: 
    rt30=t60_impulse(filename, bands, rt='t30')
    rt20=t60_impulse(filename, bands, rt='t20')
    edt=t60_impulse(filename, bands, rt='edt')
    c50=c50_from_file(filename,bands)
    # average over all bands 
    rt30=np.mean(rt30)
    rt20=np.mean(rt20)
    edt=np.mean(edt)
    c50=np.mean(c50)

    return rt30,rt20,edt,c50

def cut_or_zeropad(sig_in, len_smpl):
    if sig_in.shape[1]<len_smpl:
        sig_out = torch.zeros(sig_in.shape[0], int(len_smpl))
        sig_out[:,:sig_in.shape[1]] = sig_in
    else:
        sig_out=sig_in[:,:int(len_smpl)]
    return sig_out

def cut_or_rep(sig_in, len_smpl):
    if sig_in.shape[1]<len_smpl:
        repetitions = len_smpl // sig_in.shape[1]
        remainder = len_smpl % sig_in.shape[1]
        sig_in = torch.cat((sig_in.repeat(1,repetitions),sig_in[:,:remainder]),dim=1)
    else:
        sig_in=sig_in[:,:int(len_smpl)]
    return sig_in

def torch_standardize_std(sig_in):
    mu = torch.mean(sig_in)
    sigma = torch.std(sig_in)
    sig_out=(sig_in-mu)/sigma
    return sig_out

def torch_normalize_max_abs(signal,out=False):
    # doesnt work for batch!
    max_abs_value = torch.max(torch.abs(signal))
    standardized_signal = signal / max_abs_value
    # standardized_signal -= torch.mean(standardized_signal)
    if out:
        return standardized_signal, max_abs_value
    else:
        return standardized_signal


def torch_resample_if_needed(audio,sr,sr_target):
    if sr!=sr_target:
        audio=torchaudio.transforms.Resample(sr,sr_target)(audio.cpu())
    return audio


def get_nonsilent_frame(audio,L_win_samples):
    if audio.shape[1]<L_win_samples:
        # if signal shorter than desired datapoint length - pad and take directly
        sig_out = torch.zeros(1, int(L_win_samples))
        sig_out[:,:audio.shape[1]] = audio
        chosen_frame=sig_out
    else:
        E_rms=10*torch.log10(torch.sqrt(torch.mean(torch.square(audio))))
        E_thresh=E_rms
        E_frame=-100
        tries=0
        while E_frame<E_thresh:
            idx_start= torch.randint(0, audio.shape[1]-L_win_samples, (1,))
            chosen_frame=audio[:,idx_start:idx_start+L_win_samples]
            E_frame=10*torch.log10(torch.sqrt(torch.mean(torch.square(chosen_frame))))
            tries+=1
            if tries>=100:
                print("cannot find silent frame")
    return chosen_frame


def conv_based_crop_torch(audio_orig,fs,L_win_smpl,stride_smpl,device):
    # This function computes energy of a signal using a strided convolution in pytorch 
    # and picks the window with the highest energy out of the original audio
    # L_win_smpl - desired window lenght in samples
    # stride_smpl - how often to compute energy in samples

    # if stereo - use averaged chanels to decide about cropping
    if audio_orig.shape[1]==2:
        audio=torch.mean(audio_orig,dim=1,keepdim=True)
    else: 
        audio=audio_orig
    
    if audio.shape[2]<L_win_smpl:
        # if signal shorter than desired datapoint length - pad and take directly
        sig_out = torch.zeros(1, int(L_win_smpl))
        sig_out[:,:,:audio.shape[1]] = audio
        audio_crop=sig_out
    else:
        # if signal longer than desired datapoint length - choose fragment with the highest energy
        # using strided convolution as a method for obtaining moving average with a stepsize
        L_audio_smpl=audio.shape[2] # Shape: (batch_size, in_channels, input_length)
        # kernel size = window for energy computation
        kernel= torch.ones((L_win_smpl,),dtype=torch.float32)/L_win_smpl
        kernel=kernel.unsqueeze(0).unsqueeze(0).to(device) # Shape: (batch_size, in_channels, input_length)
        # create convolutional layer with flat kernel (no need to flip the kernel)
        conv_layer=torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=L_win_smpl, stride=stride_smpl, bias=False)
        with torch.no_grad():
            conv_layer.weight.data = kernel
        # ouput layer contains energies   
        out_layer=conv_layer(audio**2)
        # link energy values to the initial audio indices
        energy=torch.zeros(L_audio_smpl)
        energy[0:-(L_win_smpl-1):stride_smpl]=out_layer[0,0,:].detach()
        # find the index with the highest energy
        idx_start=int(torch.argmax(energy))
        idx_end=idx_start+L_win_smpl
        # crop original audio 
        audio_crop=audio_orig[:,:,idx_start:idx_end]
    return audio_crop

def torch_set_level(sig_in,L_des):
    # set FS level of the signal
    sig_zeromean=torch.subtract(sig_in,torch.mean(sig_in,dim=1))
    sig_norm_en=sig_zeromean/torch.max(torch.std(sig_zeromean.reshape(-1)),torch.tensor(1e-10))
    sig_out =sig_norm_en*np.power(10,L_des/20)
    #print(20*np.log10(np.sqrt(np.mean(np.power(sig_out,2)))))
    return sig_out

def torch_mix_and_set_snr(s,n,snr):
    if snr<100:
        # set snr between signal and noise
        n=torch_set_level(n,-30)
        s=torch_set_level(s,-30+snr)
        out=s+n
    else:
        out=s
    return out


def torch_load_mono(filename,sr_target): 
    sig,sr=torchaudio.load(filename)
    if sig.shape[0]==2:
        sig=torch.mean(sig,dim=0,keepdim=True)
    if sr!=sr_target:
        sig=torchaudio.transforms.Resample(sr,sr_target)(sig)
    return sig

def unsqueezeif2D(x): 
    if len(x.shape)<3:
       x=x.unsqueeze(0)
    return x


def plotspectrogram(y, sr, n_fft,hop_length,title):
    # Compute STFT spectrogram
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # Plot the spectrogram
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram**2, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', vmin=-80, vmax=0)
    plt.title(title)

def shiftby(sig, lag):
    sig=sig.T
    sig_out = torch.zeros_like(sig)

    if lag > 0:
        # add zeros at the beginning and cut the end
        sig_cut = sig[0:-lag]
        sig_cut = F.pad(sig_cut.T, (lag,0), mode='constant',value=0).T
    elif lag < 0:
        # cut the beginning of sig2
        sig_cut = sig[-lag:]
    else:
        sig_cut=sig

    sig_out[:sig_cut.shape[0]] = sig_cut
    
    return sig_out.T

def synch_sig2(sig1, sig2):
    sig1=sig1.T
    sig2=sig2.T
    sig2_out = torch.zeros_like(sig2)
    corr = signal.correlate(sig1, sig2, 'full')
    lag = signal.correlation_lags(len(sig1), len(sig2), mode='full')[np.argmax(corr)]
    if lag > 0:
        # add zeros at the beginning and cut the end
        sig2_cut = sig2[0:-lag]
        sig2_cut = F.pad(sig2_cut.T, (lag,0), mode='constant',value=0).T
    elif lag < 0:
        # cut the beginning of sig2
        sig2_cut = sig2[-lag:]
    else:
        sig2_cut=sig2

    sig2_out[:sig2_cut.shape[0]] = sig2_cut
    
    return sig1.T, sig2_out.T, lag


def render_random_rir(room_x,room_y,room_z,rt60, fixed_mic_dist=None):
    # generate random rir from a room specified by the 
    # above 4 parameters
    # -------------------------------------------------------
    maxlim = 1.8 # Maximum reverberation time
    band_centerfreqs=np.array([16000]) #change this for multiband
    mic_specs = np.array([[0, 0, 0, 1]]) # omni-directional
    fs_rir = 48000
    # create random src-rec position in the room 
    mic_pos, src_pos= random_srcrec_in_room(room_x,room_y,room_z,fixed_mic_dist)
    # receiver position (mono mic): 
    rec = np.array([[mic_pos[0], mic_pos[1], mic_pos[2]]])
    # source position
    src = np.array([[src_pos[0], src_pos[1], src_pos[2]]])
    # room dimensions:
    room = np.array([room_x, room_y, room_z])
    # reverberation:
    rt60 = np.array([rt60])
    # Compute absorption coefficients for desired rt60 and room dimensions
    abs_walls,rt60_true = srs.find_abs_coeffs_from_rt(room, rt60)
    # Small correction for sound absorption coefficients:
    if sum(rt60_true-rt60>0.05*rt60_true)>0:
        abs_walls,rt60_true = srs.find_abs_coeffs_from_rt(room, rt60_true + abs(rt60-rt60_true))
    # Generally, we simulate up to RT60:
    limits = np.minimum(rt60, maxlim)
    # Compute echogram:
    abs_echogram= srs.compute_echograms_mic(room, src, rec, abs_walls, limits, mic_specs)
    # Render RIR: 
    mic_rir = srs.render_rirs_mic(abs_echogram, band_centerfreqs, fs_rir)
    return mic_rir

def batch_convolution(signal, filter):
    """Performs batch convolution with pytorch fft convolution.
    Args
        signal : torch.FloatTensor (batch, n_channels, num_signal_samples)
        filter : torch.FloatTensor (batch, n_channels, num_filter_samples)
    Return
        filtered_signal : torch.FloatTensor (batch, n_channels, num_signal_samples)
    """
    batch_size, n_channels, signal_length = signal.size()
    _, _, filter_length = filter.size()

    # Pad signal in the beginning by the filter size
    padded_signal = torch.nn.functional.pad(signal, (filter_length, 0), 'constant', 0)

    # Transpose : move batch to channel dim for group convolution
    padded_signal = padded_signal.transpose(0, 1)

    filtered_signal = fft_conv(padded_signal.double(), filter.double(), padding=0, groups=batch_size).transpose(0, 1)[
        :, :, :signal_length
    ]

    filtered_signal = filtered_signal.type(signal.dtype)

    return filtered_signal

def random_srcrec_in_room(room_x,room_y,room_z,fixed_mic_dist=None):
    # mic_position - position randomly placed inside the room:
    mic_pos=[]
    mic_pos.append(np.random.uniform(low = 0.35*room_x, high = 0.65*room_x))
    mic_pos.append(np.random.uniform(low = 0.35*room_y, high = 0.65*room_y))
    mic_pos.append(np.random.uniform(low = 1., high = 2.))
    np.array(mic_pos)

    # source position always the same in reference to mic (close-mic):
    if fixed_mic_dist==None:
        src_pos=[]
        src_pos.append(np.random.uniform(low = 0.35*room_x, high = 0.65*room_x))
        src_pos.append(np.random.uniform(low = 0.35*room_y, high = 0.65*room_y))
        src_pos.append(np.random.uniform(low = 1., high = 2.))
        np.array(src_pos)
    else:
        src_pos = place_on_circle(np.array([mic_pos[0],mic_pos[1],mic_pos[2]]),fixed_mic_dist,0)
    
    return mic_pos,src_pos

def gen_rand_colored_noise(p,L):

    if random.random() < p:
        beta = random.random() + 1.0
        noise = colorednoise.powerlaw_psd_gaussian(beta, L)
        noise = np.expand_dims(noise, 0)
        noise=torch.tensor(noise, dtype=torch.float32) 
    else:
        noise = torch.zeros(1,L, dtype=torch.float32) 

    return noise

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config

def load_config_fins(file_path):
    from easydict import EasyDict as ed
    with open(file_path, encoding="utf-8") as f:
        contents = yaml.load(f, Loader=yaml.FullLoader)
    return ed(contents)

def torch_deconv_W(reverberant_signal, room_impulse_response):
    reverberant_signal=reverberant_signal.squeeze()
    room_impulse_response=room_impulse_response.squeeze()
    # Determine the longer of the two signals
    max_length = max(len(reverberant_signal), len(room_impulse_response))
    # Pad the signals to the same length
    reverberant_signal_padded = torch.nn.functional.pad(reverberant_signal, (0, max_length - len(reverberant_signal)))
    room_impulse_response_padded = torch.nn.functional.pad(room_impulse_response, (0, max_length - len(room_impulse_response)))
    # Perform FFT on both signals
    Y = torch.fft.fft(reverberant_signal_padded)
    H = torch.fft.fft(room_impulse_response_padded)
    # Compute wiener filter
    wiener_filter = torch.conj(H) / (torch.abs(H)**2 + 1e-10)
   # Apply Wiener filter
    X = Y * wiener_filter
    # Perform inverse FFT to obtain deconvolved signal
    estimated_anechoic_signal = torch.fft.ifft(X)
    # Take real part to get rid of imaginary part 
    estimated_anechoic_signal = torch.real(estimated_anechoic_signal)
    return estimated_anechoic_signal[:len(reverberant_signal)].unsqueeze(0).unsqueeze(0)


def rir_split_earlylate(rir, fs, cutpoint_ms):
    rir_early=torch.zeros_like(rir)
    rir_late=torch.zeros_like(rir)
    rir_early[:,:int(1e-3*cutpoint_ms*fs)]=rir[:,:int(1e-3*cutpoint_ms*fs)]
    rir_late[:,int(1e-3*cutpoint_ms*fs):]=rir[:,int(1e-3*cutpoint_ms*fs):]
    return rir_early, rir_late


def truncate_ir_silence(ir, sample_rate, threshold_db=20):

    ir= ir.squeeze(0)
    # Normalize the data if necessary
    ir = ir / torch.max(torch.abs(ir))

    # Find the peak value
    peak_value = torch.max(torch.abs(ir))

    # Calculate the threshold (20 dB below the peak)
    threshold_value = peak_value / (10 ** (threshold_db / 20))

    # Find the peak index
    peak_index = torch.argmax(torch.abs(ir))

    # Find the last sample before the peak that is within the threshold
    above_threshold_indices = torch.where(torch.abs(ir[:peak_index]) >= threshold_value)[0]
    if len(above_threshold_indices) == 0:
        last_sample_index = 0
    else:
        last_sample_index = above_threshold_indices[-1].item()

    # Truncate the initial silence
    truncated_ir = ir[last_sample_index:]

    return truncated_ir.unsqueeze(0)


def plot_2_waveforms(audio1, audio2, fs):
    plt.rcParams.update({'font.size': 6})

    time1 = np.linspace(0, len(audio1) / fs, num=len(audio1))
    time2 = np.linspace(0, len(audio2) / fs, num=len(audio2))

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2))

    # Plot waveform of audio1
    ax1.plot(time1, audio1)
    ax1.set_title('Waveform 1')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # Plot waveform of audio2
    ax2.plot(time2, audio2)
    ax2.set_title('Waveform 2')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_2_spectrograms(audio1, audio2, fs):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    plt.rcParams.update({'font.size': 6})

    # Compute STFT for both audio signals
    f1, t1, Zxx1 = stft(audio1, fs=fs, nperseg=512, noverlap=256)
    f2, t2, Zxx2 = stft(audio2, fs=fs, nperseg=512, noverlap=256)

    # Limit the frequency to 10 kHz
    freq_limit = 10000
    freq_idx_limit1 = np.where(f1 <= freq_limit)[0]
    freq_idx_limit2 = np.where(f2 <= freq_limit)[0]

    f1_limited = f1[freq_idx_limit1]
    f2_limited = f2[freq_idx_limit2]

    Zxx1_limited = Zxx1[freq_idx_limit1, :]
    Zxx2_limited = Zxx2[freq_idx_limit2, :]

    # Convert magnitude to dB
    Zxx1_dB = 20 * np.log10(np.abs(Zxx1_limited) + 1e-10)
    Zxx2_dB = 20 * np.log10(np.abs(Zxx2_limited) + 1e-10)

    # Compute SNR mask in dB
    snr_mask_dB = Zxx1_dB - Zxx2_dB

    # Create subplots for the two spectrograms and the SNR mask
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))

    # Plot spectrogram for audio1
    pcm1 = ax1.pcolormesh(t1, f1_limited, Zxx1_dB, shading='gouraud', cmap='inferno')
    ax1.set_title('Spectrogram 1 (dB, limited to 10 kHz)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [Hz]')
    fig.colorbar(pcm1, ax=ax1, format="%+2.0f dB")

    # Plot spectrogram for audio2
    pcm2 = ax2.pcolormesh(t2, f2_limited, Zxx2_dB, shading='gouraud', cmap='inferno')
    ax2.set_title('Spectrogram 2 (dB, limited to 10 kHz)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Frequency [Hz]')
    fig.colorbar(pcm2, ax=ax2, format="%+2.0f dB")

    # Plot SNR mask
    pcm3 = ax3.pcolormesh(t1, f1_limited, snr_mask_dB, shading='gouraud', cmap='viridis')
    ax3.set_title('SNR Mask (dB)')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Frequency [Hz]')
    fig.colorbar(pcm3, ax=ax3, format="%+2.0f dB")

    # Show the plot
    plt.tight_layout()
    plt.show()

