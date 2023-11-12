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

    def __init__(self,df_ds,sr=48e3,sig_len=48e3*2.73,split="train",content_ir=None,style_ir=None,device="cuda"):

        self.df_ds = df_ds[df_ds["split"]==split] # pd data frame with metadata 
        # Create a custom index with consecutive pairs
        # (a datapoint will constist of a mixture of two signals)
        custom_index = np.repeat(np.arange(len(self.df_ds)//2), 2)
        self.df_ds = self.df_ds.copy() # to prevent "SettingWithCopy" warning
        self.df_ds.loc[:, "pair_idx"] = custom_index
        self.sig_len=sig_len # length of input waveform 
        self.sr=int(sr) # sampling rate
        self.split = split # train/test/val
        self.device=device
        self.content_ir=content_ir
        self.style_ir=style_ir

    def __len__(self):
        return int(len(self.df_ds)/2)

    def __getitem__(self,index):
        # Pick pair of signals from metadata:
        df_pair=self.df_ds[self.df_ds["pair_idx"]==index]
        df_pair=df_pair.reset_index()

        # Load signals (and resample if needed)
        sContent = hlp.torch_load_mono(df_pair["speech_file_path"][0],self.sr).to(self.device)
        sStyle = hlp.torch_load_mono(df_pair["speech_file_path"][1],self.sr).to(self.device)
        nContent = hlp.torch_load_mono(df_pair["noise_file_path"][0],self.sr).to(self.device)
        nStyle = hlp.torch_load_mono(df_pair["noise_file_path"][1],self.sr).to(self.device)

        # Crop signals  
        sContent=hlp.get_nonsilent_frame(sContent,self.sig_len,self.device)
        sStyle=hlp.get_nonsilent_frame(sStyle,self.sig_len,self.device)
        nContent=hlp.get_nonsilent_frame(nContent,self.sig_len,self.device)
        nStyle=hlp.get_nonsilent_frame(nStyle,self.sig_len,self.device)

        # Load impulse responses
        # Note: If self.content_ir is not empty, it means that we want all content audios to have the same target ir,
        # and analogically for self.style_ir - we want only one target ir. Otherwise each style and each content audio
        # can have a different ir. This reflects if we want to learn one-to-one, many-to-one, one-to-many, or many-to-many. 

        if self.content_ir is None:
            irContent = hlp.torch_load_mono(df_pair["ir_file_path"][0],self.sr).to(self.device)
        elif self.content_ir=="anechoic":
            irContent = torch.cat((torch.tensor([[1.0]],device=self.device), torch.zeros((1,self.sr-1), device=self.device)),1)
        else: 
            irContent = hlp.torch_load_mono(self.content_ir,self.sr).to(self.device)
            
        if self.style_ir is None:
            irStyle = hlp.torch_load_mono(df_pair["ir_file_path"][1],self.sr).to(self.device)
        else: 
            irStyle = hlp.torch_load_mono(self.style_ir,self.sr).to(self.device)

        # Convolve signals with impulse responses
        sContent_rev=torchaudio.functional.convolve(sContent, irContent,mode="same")
        sStyle_rev=torchaudio.functional.convolve(sStyle, irStyle,mode="same")
        sTarget_rev=torchaudio.functional.convolve(sContent, irStyle,mode="same")

        # Add noise to signals
        snr1=df_pair["snr"][0]
        snr2=df_pair["snr"][1]
        sContent_noisyrev=hlp.torch_mix_and_set_snr(sContent_rev,nContent,snr1)
        sStylen_noisyrev=hlp.torch_mix_and_set_snr(sStyle_rev,nStyle,snr2)

        # Remove unnecessary from memory
        del sContent, sStyle, nContent, nStyle, irContent, irStyle
        torch.cuda.empty_cache() 

        # scale data but preserve symmetry
        sContent_in=hlp.torch_standardize_max_abs(sContent_noisyrev)
        sStyle_in=hlp.torch_standardize_max_abs(sStylen_noisyrev)
        sTarget_out=hlp.torch_standardize_max_abs(sTarget_rev)

        # return in the format (batch_size, in_channels, input_length)
        return sContent_in, sStyle_in, sTarget_out

if __name__ == "__main__":

    # # ---- check if the dataset definition is correct: ----
    STYLE_RIR_FILE ="/home/ubuntu/Data/ACE-Single/Lecture_Room_1/1/Single_508_1_RIR.wav"
    DF_METADATA="/home/ubuntu/joanna/reverb-match-cond-u-net/notebooks/data_set_check.csv" 
    SAMPLING_RATE=int(48e3)
    SIG_LEN_SEC=2 # TODO: instead of only cropping -> cut or zero-pad 

    df_ds=pd.read_csv(DF_METADATA,index_col=0)
    DEVICE="cuda"

    # create a tag for dataset info file 
    dataset=DatasetReverbTransfer(df_ds,sr=48e3,sig_len=98304,split="train",content_ir=None,style_ir=STYLE_RIR_FILE,device=DEVICE)

    import time
    start_time = time.time()
    sContent_in, sStyle_in, sTarget_out = dataset[39]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{elapsed_time=}")

    # create a tag for dataset info file
    print("Number of data points:" + str(len(dataset)))
    print("Dimensions of input data:" + str(dataset[20][0].shape))
