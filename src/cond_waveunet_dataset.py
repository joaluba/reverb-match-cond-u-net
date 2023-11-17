import torch
import scipy
import numpy as np
from torch.utils.data import Dataset
import torchaudio
import joa_helpers as hlp
import pandas as pd
import json
from datetime import datetime
import scipy.signal as signal


class DatasetReverbTransfer(Dataset):

    def __init__(self,args):

        self.df_ds = args.df_ds[args.df_ds["split"]==args.split] # pd data frame with metadata 
        # Create a custom index with consecutive pairs
        # (a datapoint will constist of a mixture of two signals)
        custom_index = np.repeat(np.arange(len(self.df_ds)//2), 2)
        self.df_ds = self.df_ds.copy() # to prevent "SettingWithCopy" warning
        self.df_ds.loc[:, "pair_idx"] = custom_index
        self.sig_len=args.sig_len # length of input waveform 
        self.fs=args.fs # sampling rate
        self.split = args.split # train/test/val
        self.device=args.device
        self.content_ir=args.content_rir
        self.style_ir=args.style_rir

    def __len__(self):
        return int(len(self.df_ds)/2)

    def __getitem__(self,index):
        # Pick pair of signals from metadata:
        df_pair=self.df_ds[self.df_ds["pair_idx"]==index]
        df_pair=df_pair.reset_index()

        # Load signals (and resample if needed)
        sContent = hlp.torch_load_mono(df_pair["speech_file_path"][0],self.fs)
        sStyle = hlp.torch_load_mono(df_pair["speech_file_path"][1],self.fs)
        nContent = hlp.torch_load_mono(df_pair["noise_file_path"][0],self.fs)
        nStyle = hlp.torch_load_mono(df_pair["noise_file_path"][1],self.fs)

        # Crop signals to a desired length
        sContent=hlp.get_nonsilent_frame(sContent,self.sig_len)
        sStyle=hlp.get_nonsilent_frame(sStyle,self.sig_len)
        nContent=hlp.get_nonsilent_frame(nContent,self.sig_len)
        nStyle=hlp.get_nonsilent_frame(nStyle,self.sig_len)

        # Load impulse responses
        # Note: If self.content_ir is not empty, it means that we want all content audios to have the same target ir,
        # and analogically for self.style_ir - we want only one target ir. Otherwise each style and each content audio
        # can have a different ir. This reflects if we want to learn one-to-one, many-to-one, one-to-many, or many-to-many. 

        if self.content_ir is None:
            irContent = hlp.torch_load_mono(df_pair["ir_file_path"][0],self.fs)
        elif self.content_ir=="anechoic":
            irContent = torch.cat((torch.tensor([[1.0]]), torch.zeros((1,self.fs-1))),1)
        else: 
            irContent = hlp.torch_load_mono(self.content_ir,self.fs)
            
        if self.style_ir is None:
            irStyle = hlp.torch_load_mono(df_pair["ir_file_path"][1],self.fs)
        else: 
            irStyle = hlp.torch_load_mono(self.style_ir,self.fs)

        # Convolve signals with impulse responses
        sContent_rev = torch.from_numpy(scipy.signal.fftconvolve(sContent, irContent,mode="full"))[:,:self.sig_len]
        sStyle_rev = torch.from_numpy(scipy.signal.fftconvolve(sStyle, irStyle,mode="full"))[:,:self.sig_len]
        sTarget_rev = torch.from_numpy(scipy.signal.fftconvolve(sContent, irStyle,mode="full"))[:,:self.sig_len]

        # Add noise to signals
        snr1=df_pair["snr"][0]
        snr2=df_pair["snr"][1]
        sContent_noisyrev=hlp.torch_mix_and_set_snr(sContent_rev,nContent,snr1)
        sStylen_noisyrev=hlp.torch_mix_and_set_snr(sStyle_rev,nStyle,snr2)

        # scale data but preserve symmetry
        sContent_in=hlp.torch_standardize_max_abs(sContent_noisyrev)
        sStyle_in=hlp.torch_standardize_max_abs(sStylen_noisyrev)
        sTarget_out=hlp.torch_standardize_max_abs(sTarget_rev)

        return sContent_in, sStyle_in, sTarget_out
