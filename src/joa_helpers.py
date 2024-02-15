from acoustics.room import t60_impulse, c50_from_file
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import librosa
import scipy.signal as signal
from masp import shoebox_room_sim as srs
from os.path import join as pjoin

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
        sig_out = torch.zeros(1, int(len_smpl))
        sig_out[:,:sig_in.shape[1]] = sig_in
    else:
        sig_out=sig_in[:,:int(len_smpl)]
    return sig_out

def torch_standardize_std(sig_in):
    mu = torch.mean(sig_in)
    sigma = torch.std(sig_in)
    sig_out=(sig_in-mu)/sigma
    return sig_out

def torch_standardize_max_abs(signal):
    max_abs_value = torch.max(torch.abs(signal))
    standardized_signal = signal / max_abs_value
    standardized_signal -= torch.mean(standardized_signal)
    return standardized_signal

def torch_resample_if_needed(audio,sr,sr_target):
    if sr!=sr_target:
        audio=torchaudio.transforms.Resample(sr,sr_target)(audio)
    return audio


def get_nonsilent_frame(audio,L_win_samples):
    if audio.shape[1]<L_win_samples:
        # if signal shorter than desired datapoint length - pad and take directly
        sig_out = torch.zeros(1, int(L_win_samples))
        sig_out[:,:audio.shape[1]] = audio
        chosen_frame=sig_out
    else:
        E_rms=10*torch.log10(torch.sqrt(torch.mean(torch.square(audio))))
        E_thresh=E_rms-5
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
    sig_norm_en=sig_zeromean/torch.std(sig_zeromean.reshape(-1))
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


def plotspectrogram(y, sr, n_fft,hop_length,title):
    # Compute STFT spectrogram
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # Plot the spectrogram
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram**2, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', vmin=-80, vmax=0)
    plt.title(title)


def synch_sigs(sig1,sig2):
    sig1_out=np.zeros(sig1.shape)
    sig2_out=np.zeros(sig2.shape)
    corr = signal.correlate(sig1, sig2, 'full')
    lag = signal.correlation_lags(len(sig1), len(sig2), mode='full')[np.argmax(corr)]
    if lag > 0:
        sig2=sig2[0:-lag]
        sig1=sig1[lag:]
    elif lag < 0:
        sig2=sig2[-lag:]
        sig1=sig1[0:lag]

    sig1_out[:sig1.shape[0]]=sig1
    sig2_out[:sig2.shape[0]]=sig2
    return sig1_out,sig2_out, lag


def render_random_rir(room_x,room_y,room_z,rt60):
    # generate random rir from a room specified by the 
    # above 4 parameters
    # -------------------------------------------------------
    maxlim = 1.8 # Maximum reverberation time
    band_centerfreqs=np.array([16000]) #change this for multiband
    mic_specs = np.array([[0, 0, 0, 1]]) # omni-directional
    fs_rir = 48000
    # create random src-rec position in the room 
    mic_pos, src_pos= random_srcrec_in_room(room_x,room_y,room_z)
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

def random_srcrec_in_room(room_x,room_y,room_z):
    # mic_position - position randomly placed inside the room:
    mic_pos=[]
    mic_pos.append(np.random.uniform(low = 0.35*room_x, high = 0.65*room_x))
    mic_pos.append(np.random.uniform(low = 0.35*room_y, high = 0.65*room_y))
    mic_pos.append(np.random.uniform(low = 1., high = 2.))
    np.array(mic_pos)
    # source position always the same in reference to mic (close-mic):
    src_pos = place_on_circle(np.array([mic_pos[0],mic_pos[1],mic_pos[2]]),0.1,0)
    return mic_pos,src_pos


def deconvolve(reverberant,anechoic):
    reverberant_spectrum = np.fft.fft(reverberant)
    anechoic_spectrum = np.fft.fft(anechoic)

    # Perform deconvolution
    impulse_response_spectrum = np.divide(reverberant_spectrum, anechoic_spectrum)

    # Inverse Fourier Transform
    impulse_response = np.fft.ifft(impulse_response_spectrum)
    
    return impulse_response
