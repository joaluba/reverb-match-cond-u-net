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
        self.df_ds=pd.read_csv(args.df_metadata,index_col=0) # pd data frame with metadata 
        self.df_ds = self.df_ds[self.df_ds["split"]==args.split] 
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
        s1 = hlp.torch_load_mono(df_pair["speech_file_path"][0],self.fs)
        s2 = hlp.torch_load_mono(df_pair["speech_file_path"][1],self.fs)
        # n1 = hlp.torch_load_mono(df_pair["noise_file_path"][0],self.fs)
        # n2 = hlp.torch_load_mono(df_pair["noise_file_path"][1],self.fs)

        # Crop signals to a desired length
        s1=hlp.get_nonsilent_frame(s1,self.sig_len)
        s2=hlp.get_nonsilent_frame(s2,self.sig_len)
        # Apply phase shift or none
        s1*=torch.tensor(df_pair["aug_phase"][0])
        s2*=torch.tensor(df_pair["aug_phase"][1])
        

        # n1=hlp.get_nonsilent_frame(n1,self.sig_len)
        # n2=hlp.get_nonsilent_frame(n2,self.sig_len)

        # Load impulse responses
        # Note: If self.content_ir is not empty, it means that we want all content audios to have the same target ir,
        # and analogically for self.style_ir - we want only one target ir. Otherwise each style and each content audio
        # can have a different ir. This reflects if we want to learn one-to-one, many-to-one, one-to-many, or many-to-many. 

        if self.content_ir is None:
            r1 = hlp.torch_load_mono(df_pair["ir_file_path"][0],self.fs)
        elif self.content_ir=="anechoic":
            r1 = torch.cat((torch.tensor([[1.0]]), torch.zeros((1,self.fs-1))),1)
        else: 
            r1 = hlp.torch_load_mono(self.content_ir,self.fs)
            
        if self.style_ir is None:
            r2 = hlp.torch_load_mono(df_pair["ir_file_path"][1],self.fs)
        else: 
            r2 = hlp.torch_load_mono(self.style_ir,self.fs)


        # separate style rir into early and late 
        cutpoint_ms=50
        r2_early, r2_late = hlp.rir_split_earlylate(r2,self.fs,cutpoint_ms)

        # Convolve signals with impulse responses
        s1r1 = torch.from_numpy(scipy.signal.fftconvolve(s1, r1,mode="full"))[:,:self.sig_len]
        s2r2 = torch.from_numpy(scipy.signal.fftconvolve(s2, r2,mode="full"))[:,:self.sig_len]
        # s1r2 = torch.from_numpy(scipy.signal.fftconvolve(s1, r2,mode="full"))[:,:self.sig_len]
        s1r2_early = torch.from_numpy(scipy.signal.fftconvolve(s1, r2_early,mode="full"))[:,:self.sig_len]
        s1r2_late = torch.from_numpy(scipy.signal.fftconvolve(s1, r2_late,mode="full"))[:,:self.sig_len] 
        # s2r1 = torch.from_numpy(scipy.signal.fftconvolve(s2, r1,mode="full"))[:,:self.sig_len]

        # Add noise to signals
        snr1=df_pair["snr"][0]
        snr2=df_pair["snr"][1]
        # s1r1n1=hlp.torch_mix_and_set_snr(s1r1,n1,snr1)
        # s2r2n2=hlp.torch_mix_and_set_snr(s2r2,n2,snr2)
        s1r1n1=s1r1
        s2r2n2=s2r2

        # scale data but preserve symmetry
        s1r1n1=hlp.torch_standardize_max_abs(s1r1n1) # Reverberant content sound
        s2r2n2=hlp.torch_standardize_max_abs(s2r2n2) # Style sound
        s1r2, sc_max=hlp.torch_standardize_max_abs(s1r2_early+s1r2_late,out=True) # Target all
        s1r2_early=s1r2_early/sc_max
        s1r2_late=s1r2_late/sc_max
        # s1r2=hlp.torch_standardize_max_abs(s1r2) # Target
        # s2r1=hlp.torch_standardize_max_abs(s2r1) # "Flipped" target
        s1=hlp.torch_standardize_max_abs(s1) # Anechoic content sound


        return s1r1n1, s2r2n2, s1r2, s1, s1r2_early, s1r2_late
    
    def get_idx_with_rt60diff(self,diff_rt60_min,diff_rt60_max):
        # create column diff_rt60 to compute difference in rt60 between content and style audio
        self.df_ds["diff_rt60"] = self.df_ds["rt60_true"].diff()
        self.df_ds["diff_rt60"][0::2]=self.df_ds["diff_rt60"][1::2]
        # # check indices of datapoint where the rt60 for content is lower than rt60 for style
        selected=self.df_ds[(self.df_ds["diff_rt60"]>diff_rt60_min) & (self.df_ds["diff_rt60"]<diff_rt60_max)]
        selected=selected.iloc[::2]
        selected=selected["pair_idx"].tolist()
        return selected
    

    def get_info(self,index,id="style"):

        df_pair=self.df_ds[self.df_ds["pair_idx"]==index]

        if id=="style":
            styleorcontent_idx=1
        elif id=="content":
            styleorcontent_idx=0

        df=df_pair.iloc[styleorcontent_idx]

        return df

    def get_rirs(self,index):
        # Pick pair of signals from metadata:
        df_pair=self.df_ds[self.df_ds["pair_idx"]==index]
        df_pair=df_pair.reset_index()

        if self.content_ir is None:
            r1 = hlp.torch_load_mono(df_pair["ir_file_path"][0],self.fs)
            r1b=r1
        elif self.content_ir=="anechoic":
            r1 = torch.cat((torch.tensor([[1.0]]), torch.zeros((1,self.fs-1))),1)
            r1b=r1
        else: 
            r1 = hlp.torch_load_mono(self.content_ir,self.fs)
            r1b =hlp.render_random_rir(df_pair["room_x"],df_pair["room_y"],df_pair["room_z"],df_pair["rt60_set"])

        if self.style_ir is None:
            r2 = hlp.torch_load_mono(df_pair["ir_file_path"][1],self.fs)
        else: 
            r2 = hlp.torch_load_mono(self.style_ir,self.fs)
        
        return r1,r2,r1b

            
        


    def get_item_test(self,index):
                # Pick pair of signals from metadata:
        df_pair=self.df_ds[self.df_ds["pair_idx"]==index]
        df_pair=df_pair.reset_index()

        # Load signals (and resample if needed)
        s1 = hlp.torch_load_mono(df_pair["speech_file_path"][0],self.fs)
        s2 = hlp.torch_load_mono(df_pair["speech_file_path"][1],self.fs)
        # n1 = hlp.torch_load_mono(df_pair["noise_file_path"][0],self.fs)
        # n2 = hlp.torch_load_mono(df_pair["noise_file_path"][1],self.fs)

        # Crop signals to a desired length
        s1=hlp.get_nonsilent_frame(s1,self.sig_len)
        s2=hlp.get_nonsilent_frame(s2,self.sig_len)
        # Apply phase shift or none
        s1*=torch.tensor(df_pair["aug_phase"][0])
        s2*=torch.tensor(df_pair["aug_phase"][1])
        

        # n1=hlp.get_nonsilent_frame(n1,self.sig_len)
        # n2=hlp.get_nonsilent_frame(n2,self.sig_len)

        # Load impulse responses
        # Note: If self.content_ir is not empty, it means that we want all content audios to have the same target ir,
        # and analogically for self.style_ir - we want only one target ir. Otherwise each style and each content audio
        # can have a different ir. This reflects if we want to learn one-to-one, many-to-one, one-to-many, or many-to-many. 

        if self.content_ir is None:
            r1 = hlp.torch_load_mono(df_pair["ir_file_path"][0],self.fs)
            r1b=r1
        elif self.content_ir=="anechoic":
            r1 = torch.cat((torch.tensor([[1.0]]), torch.zeros((1,self.fs-1))),1)
            r1b=r1
        else: 
            r1 = hlp.torch_load_mono(self.content_ir,self.fs)
            r1b =hlp.render_random_rir(df_pair["room_x"],df_pair["room_y"],df_pair["room_z"],df_pair["rt60_set"])

            
        if self.style_ir is None:
            r2 = hlp.torch_load_mono(df_pair["ir_file_path"][1],self.fs)
        else: 
            r2 = hlp.torch_load_mono(self.style_ir,self.fs)

        # Convolve signals with impulse responses
        s1r1 = torch.from_numpy(scipy.signal.fftconvolve(s1, r1,mode="full"))[:,:self.sig_len]
        s2r2 = torch.from_numpy(scipy.signal.fftconvolve(s2, r2,mode="full"))[:,:self.sig_len]
        s1r2 = torch.from_numpy(scipy.signal.fftconvolve(s1, r2,mode="full"))[:,:self.sig_len]
        s2r1 = torch.from_numpy(scipy.signal.fftconvolve(s2, r1,mode="full"))[:,:self.sig_len]
        s1r1b = torch.from_numpy(scipy.signal.fftconvolve(s1, r1b,mode="full"))[:,:self.sig_len]

        # Add noise to signals
        snr1=df_pair["snr"][0]
        snr2=df_pair["snr"][1]
        # s1r1n1=hlp.torch_mix_and_set_snr(s1r1,n1,snr1)
        # s2r2n2=hlp.torch_mix_and_set_snr(s2r2,n2,snr2)
        s1r1n1=s1r1
        s2r2n2=s2r2

        # scale data but preserve symmetry
        s1r1n1=hlp.torch_standardize_max_abs(s1r1n1) # Reverberant content sound
        s2r2n2=hlp.torch_standardize_max_abs(s2r2n2) # Style sound
        s1r2=hlp.torch_standardize_max_abs(s1r2) # Target
        s2r1=hlp.torch_standardize_max_abs(s2r1) # "Flipped" target
        s1=hlp.torch_standardize_max_abs(s1) # Anechoic content sound
        s1r1b=hlp.torch_standardize_max_abs(s1r1b) # Reverberant signal from the same room as content (but different position)

        return s1r1n1, s2r2n2, s1r2, s1, s2r1, s2, s1r1b