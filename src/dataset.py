import torch
import scipy
import numpy as np
from torch.utils.data import Dataset
import torchaudio
import helpers as hlp
import pandas as pd
import json
from datetime import datetime
import scipy.signal as signal


class DatasetReverbTransfer(Dataset):

    def __init__(self,config):
        self.df_ds=pd.read_csv(config["df_metadata"],index_col=None) # pd data frame with metadata 
        self.df_ds = self.df_ds[self.df_ds["split"]==config["split"]].reset_index()
        # Create a custom index with consecutive pairs
        # (a datapoint will constist of a mixture of two signals)
        custom_index = np.repeat(np.arange(len(self.df_ds)//2), 2)
        self.df_ds = self.df_ds.copy() # to prevent "SettingWithCopy" warning
        self.df_ds.loc[:, "pair_idx"] = custom_index
        self.sig_len=config["sig_len"] # length of input waveform 
        self.fs=config["fs"] # sampling rate
        self.split = config["split"] # train/test/val
        self.device=config["device"]
        self.content_ir=config["content_rir"]
        self.style_ir=config["style_rir"]
        self.p_noise=config["p_noise"] if config["split"]=="train" else 0

    def __len__(self):
        return int(len(self.df_ds)/2)

    def __getitem__(self,index):
        # Pick pair of signals from metadata:
        df_pair=self.df_ds[self.df_ds["pair_idx"]==index]
        df_pair=df_pair.reset_index()

        # Load signals (and resample if needed)
        s1 = hlp.torch_load_mono(df_pair["speech_file_path"][0],self.fs)
        s2 = hlp.torch_load_mono(df_pair["speech_file_path"][1],self.fs)

        # Crop signals to a desired length
        s1=hlp.get_nonsilent_frame(s1,self.sig_len)
        s2=hlp.get_nonsilent_frame(s2,self.sig_len)

        # Apply phase shift or none
        s1*=np.random.choice([-1, 1])
        s2*=np.random.choice([-1, 1])

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


        # Convolve signals with impulse responses
        s1r1 = torch.from_numpy(scipy.signal.fftconvolve(s1, r1,mode="full"))[:,:self.sig_len]
        s2r2 = torch.from_numpy(scipy.signal.fftconvolve(s2, r2,mode="full"))[:,:self.sig_len]
        s1r2 = torch.from_numpy(scipy.signal.fftconvolve(s1, r2,mode="full"))[:,:self.sig_len]

        # # Synchronize all signals to anechoic signal
        _,s1r1,_ = hlp.synch_sig2(s1,s1r1)
        _,s1r2,_ = hlp.synch_sig2(s1,s1r2)
        _,s2r2,_ = hlp.synch_sig2(s2,s2r2)

        # generate background noise samples
        n1=hlp.gen_rand_colored_noise(self.p_noise,self.sig_len)
        n2=hlp.gen_rand_colored_noise(self.p_noise,self.sig_len)

        # Add noise to content and style signal
        snr1=15 + (40 - 15) * torch.rand(1)
        snr2=15 + (40 - 15) * torch.rand(1)
        s1r1n1=hlp.torch_mix_and_set_snr(s1r1,n1,snr1)
        s2r2n2=hlp.torch_mix_and_set_snr(s2r2,n2,snr2)

        # scale data but preserve symmetry
        s1r1n1=hlp.torch_standardize_max_abs(s1r1n1) # Reverberant content sound
        s2r2n2=hlp.torch_standardize_max_abs(s2r2n2) # Style sound
        s1r2=hlp.torch_standardize_max_abs(s1r2) # Target

        # s2r1=hlp.torch_standardize_max_abs(s2r1) # "Flipped" target
        s1=hlp.torch_standardize_max_abs(s1) # Anechoic content sound

        return s1r1n1, s2r2n2, s1r2, s1
    
    def get_idx_with_rt60diff(self,diff_rt60_min,diff_rt60_max):
        # create column diff_rt60 to compute difference in rt60 between content and style audio
        self.df_ds["diff_rt60"] = self.df_ds["rt60_true"].diff()
        self.df_ds.loc[0::2, 'diff_rt60'] = self.df_ds['diff_rt60'].shift(periods=-1)
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

    def get_item_test(self,index, gen_rir_b=False, fixed_mic_dist=None):
        # Pick pair of signals from metadata:
        df_pair=self.df_ds[self.df_ds["pair_idx"]==index]
        df_pair=df_pair.reset_index()

        # Load signals (and resample if needed)
        s1 = hlp.torch_load_mono(df_pair["speech_file_path"][0],self.fs)
        s2 = hlp.torch_load_mono(df_pair["speech_file_path"][1],self.fs)

        # Crop signals to a desired length
        s1=hlp.get_nonsilent_frame(s1,self.sig_len)
        s2=hlp.get_nonsilent_frame(s2,self.sig_len)

        # Apply phase shift or none
        s1*=torch.tensor(df_pair["aug_phase"][0])
        s2*=torch.tensor(df_pair["aug_phase"][1])
        
        # Load impulse responses
        # Note: If self.content_ir is not empty, it means that we want all content audios to have the same target ir,
        # and analogically for self.style_ir - we want only one target ir. Otherwise each style and each content audio
        # can have a different ir. This reflects if we want to learn one-to-one, many-to-one, one-to-many, or many-to-many. 

        if self.content_ir is None:
            if fixed_mic_dist==None:
                print("generating r1 with random src-rec distance")
                r1_array=hlp.render_random_rir(df_pair["room_x"][0],df_pair["room_y"][0],df_pair["room_z"][0],df_pair["rt60_set"][0],fixed_mic_dist=fixed_mic_dist)
                r1= torch.tensor(r1_array.T).squeeze(0).float()
            else:
                print("loading r1 with fixed src-rec distance")
                r1 = hlp.torch_load_mono(df_pair["ir_file_path"][0],self.fs)
            if gen_rir_b: # generate 
                print("generating r1b with src-rec distance:" + str(fixed_mic_dist))
                r1b_array =hlp.render_random_rir(df_pair["room_x"][0],df_pair["room_y"][0],df_pair["room_z"][0],df_pair["rt60_set"][0],fixed_mic_dist=fixed_mic_dist)
                r1b= torch.tensor(r1b_array.T).squeeze(0).float()
        elif self.content_ir=="anechoic":
            r1 = torch.cat((torch.tensor([[1.0]]), torch.zeros((1,self.fs-1))),1)
        else: 
            r1 = hlp.torch_load_mono(self.content_ir,self.fs)

            
        if self.style_ir is None:
            if fixed_mic_dist==None:
                print("generating r2 with random src-rec distance")
                r2_array=hlp.render_random_rir(df_pair["room_x"][1],df_pair["room_y"][1],df_pair["room_z"][1],df_pair["rt60_set"][1],fixed_mic_dist=fixed_mic_dist)
                r2= torch.tensor(r2_array.T).squeeze(0).float()
            else:
                print("loading r2 with fixed src-rec distance")
                r2 = hlp.torch_load_mono(df_pair["ir_file_path"][1],self.fs)
            
        else: 
            r2 = hlp.torch_load_mono(self.style_ir,self.fs)

        # separate rirs into early and late 
        cutpoint_ms=50
        r2_early, r2_late = hlp.rir_split_earlylate(r2,self.fs,cutpoint_ms)
        r1_early, r1_late = hlp.rir_split_earlylate(r1,self.fs,cutpoint_ms)
        if gen_rir_b:
            r1b_early, r1b_late = hlp.rir_split_earlylate(r1b,self.fs,cutpoint_ms)


        # content a
        s1r1_early = torch.from_numpy(scipy.signal.fftconvolve(s1, r1_early,mode="full"))[:,:self.sig_len]
        s1r1_late = torch.from_numpy(scipy.signal.fftconvolve(s1, r1_late,mode="full"))[:,:self.sig_len]
        s1r1, sc_max=hlp.torch_standardize_max_abs(s1r1_early+s1r1_late,out=True) # Target all
        s1r1_early=s1r1_early/sc_max
        s1r1_late=s1r1_late/sc_max
        # Synchronize all signals to anechoic
        _,s1r1,lag = hlp.synch_sig2(s1,s1r1)
        s1r1_early=hlp.shiftby(s1r1_early,lag)
        s1r1_late=hlp.shiftby(s1r1_late,lag)


        # content b
        if gen_rir_b:
            s1r1b_early = torch.from_numpy(scipy.signal.fftconvolve(s1, r1b_early,mode="full"))[:,:self.sig_len]
            s1r1b_late = torch.from_numpy(scipy.signal.fftconvolve(s1, r1b_late,mode="full"))[:,:self.sig_len]
            s1r1b, sc_max=hlp.torch_standardize_max_abs(s1r1b_early+s1r1b_late,out=True) # Target all
            s1r1b_early=s1r1b_early/sc_max
            s1r1b_late=s1r1b_late/sc_max
            # Synchronize all signals to anechoic
            _,s1r1b,lag = hlp.synch_sig2(s1,s1r1b)
            print(lag)
            s1r1b_early=hlp.shiftby(s1r1b_early,lag)
            s1r1b_late=hlp.shiftby(s1r1b_late,lag)

        # target
        s1r2_early = torch.from_numpy(scipy.signal.fftconvolve(s1, r2_early,mode="full"))[:,:self.sig_len]
        s1r2_late = torch.from_numpy(scipy.signal.fftconvolve(s1, r2_late,mode="full"))[:,:self.sig_len]
        s1r2, sc_max=hlp.torch_standardize_max_abs(s1r2_early+s1r2_late,out=True) # Target all
        s1r2_early=s1r2_early/sc_max
        s1r2_late=s1r2_late/sc_max
        # Synchronize all signals to anechoic
        _,s1r2,lag = hlp.synch_sig2(s1,s1r2)
        print(lag)
        s1r2_early=hlp.shiftby(s1r2_early,lag)
        s1r2_late=hlp.shiftby(s1r2_late,lag)
        

        # style
        s2r2 = torch.from_numpy(scipy.signal.fftconvolve(s2, r2,mode="full"))[:,:self.sig_len]
        s2r1 = torch.from_numpy(scipy.signal.fftconvolve(s2, r1,mode="full"))[:,:self.sig_len]
        s2r2=hlp.torch_standardize_max_abs(s2r2) # Style sound
        s2r1=hlp.torch_standardize_max_abs(s2r1) # "Flipped" target
        # Synchronize all signals to anechoic
        _,s2r1,lag = hlp.synch_sig2(s2,s2r1)
        print(lag)
        s2r1=s1r2

        # Anechoic
        s1=hlp.torch_standardize_max_abs(s1) 

        if gen_rir_b:
            signals={
                "s1r1": s1r1.to(self.device),
                "s1r1_early": s1r1_early.to(self.device),
                "s1r1_late": s1r1_late.to(self.device),
                "s1r1b": s1r1b.to(self.device),
                "s1r1b_early": s1r1b_early.to(self.device),
                "s1r1b_late": s1r1b_late.to(self.device),
                "s1r2": s1r2.to(self.device),
                "s1r2_early": s1r2_early.to(self.device),
                "s1r2_late": s1r2_late.to(self.device),
                "s2r2": s2r2.to(self.device),
                "s2r1": s2r1.to(self.device),
                "s1": s1.to(self.device),
                "s2": s2.to(self.device),
                }
            
            rirs={
                "r1": r1,
                "r1_early": r1_early,
                "r1_late": r1_late,
                "r2": r2,
                "r2_early": r2_early,
                "r2_late": r2_late,
                "r1b": r1b,
                }
        
        else:
                     
            signals={
                "s1r1": s1r1,
                "s1r1_early": s1r1_early,
                "s1r1_late": s1r1_late,
                "s1r2": s1r2,
                "s1r2_early": s1r2_early,
                "s1r2_late": s1r2_late,
                "s2r2": s2r2,
                "s2r1": s2r1,
                "s1": s1,
                "s2": s2,
                }
            
            rirs={
                "r1": r1,
                "r1_early": r1_early,
                "r1_late": r1_late,
                "r2": r2,
                "r2_early": r2_early,
                "r2_late": r2_late,
                }

        return signals, rirs