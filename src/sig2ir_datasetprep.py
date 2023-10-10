import torch
import scipy
import numpy as np
from torch.utils.data import Dataset
import torchaudio
import helpers
import pandas as pd
import json
from datetime import datetime


class Dataset_SpeechInSpace(Dataset):

    def __init__(self,df_audiopool,df_irs,sr=48e3, ir_len=int(48e3*2), sig_len=48e3*2.73, N_per_ir=1e4):

        self.df_irs = df_irs # pd data frame with info and paths to impulse responses
        self.N_per_ir = int(N_per_ir) # number of samples per each IR in the data set
        self.df_audiopool = df_audiopool.sample(n = self.N_per_ir) # pd data frame with paths to audio files
        self.sr=sr # sampling rate
        self.ir_len=ir_len # crop all the irs to this duration [s]
        self.sig_len=sig_len # length of the (reverberant) data point in seconds
        self.preproc="wave" # feature extraction method
        # self.ds_df=self.df_audiopool.merge(self.df_irs, how='cross')
        self.ds_df=self.create_rand_combinations()

    def __len__(self):
        return len(self.ds_df)

    def __getitem__(self,index):

        # --- room impulse response: ----
        # load
        ir, sr_ir = torchaudio.load(self.ds_df["filepath_ir"][int(index)])
        # resample
        ir=torchaudio.transforms.Resample(sr_ir,self.sr)(ir)
        # cut or zero-pad
        ir=helpers.cut_or_zeropad(ir,self.ir_len)
        # scale data but preserve symmetry
        ir=helpers.standardize_max_abs(ir)
        # ---- audio: ----
        # load
        sig, sr_sig = torchaudio.load(self.ds_df["filepath_sig"][int(index)])
        # resample
        sig=torchaudio.transforms.Resample(sr_sig,self.sr)(sig)
        # cut random excerpt or zero-pad 
        N=int(self.sig_len)
        sig_rand_excerpt=helpers.cut_or_zeropad_random(sig,N)
        # apply hanning flanks 
        sig_rand_excerpt=helpers.apply_hann_flanks(sig_rand_excerpt,0.5,self.sr)
        # scale data but preserve symmetry
        sig_rand_excerpt=helpers.standardize_max_abs(sig_rand_excerpt)
        # ---- convolve: ----
        sig_ir = torch.from_numpy(scipy.signal.fftconvolve(sig_rand_excerpt, ir))
        # crop signal to the length of the audio 
        sig_ir=sig_ir[:,:N]
        # scale data but preserve symmetry
        sig_ir = helpers.standardize_max_abs(sig_ir)
       
        if self.preproc=="wave": 
            # store signal as input data point
            data_point=sig_ir.view(1,sig_ir.shape[0],sig_ir.shape[1])
            assert data_point.shape==torch.Size([1,1,int(self.sig_len)]), f"{data_point.shape=}"

        # create label consisting of source audio and room acoustic params, 
        label={
            "filepath_ir": self.ds_df["filepath_ir"][int(index)],
            "filepath_sig": self.ds_df["filepath_sig"][int(index)],
            "database_ir":self. ds_df["database_ir"][int(index)],
            "database_sig":self. ds_df["database_sig"][int(index)],
            "mic": self.ds_df["mic"][int(index)],
            "rt": self.ds_df["rt"][int(index)],
            "drr": self.ds_df["drr"][int(index)],
            "cte": self.ds_df["cte"][int(index)],
            "edt": self.ds_df["edt"][int(index)],
        }
        return sig_rand_excerpt, ir, data_point, label
    
    def create_rand_combinations(self):
        # initialize data frame with a full data set
        df_ir_expanded=pd.concat([self.df_irs] * self.N_per_ir, ignore_index=True)
        df_audio_samples=self.df_audiopool.sample(n = len(df_ir_expanded),replace=True)
        df_ir_expanded.reset_index(drop=True, inplace=True)
        df_audio_samples.reset_index(drop=True, inplace=True)
        df_ds = pd.concat([df_ir_expanded, df_audio_samples], axis=1)
        df_ds = df_ds.sort_values(by='filepath_ir').reset_index(drop=True)
        return df_ds
        
    
    def save_dataset_info(self,dir,nametag):
        # create dictionary with parameters for training data
        train_params={"N_ir": len(self.df_irs),
                    "N_per_ir": self.N_per_ir,
                    "sr": self.sr,
                    "ir_len": self.ir_len,
                    "sig_len": self.sig_len,
                    "preproc":self.preproc}
        # combine dictionary with df containg training data list
        data = {
        'ds_df': self.ds_df.to_dict(orient='records'),
        'train_params': train_params
        }
        # Save the combined data as a JSON file
        with open(dir+'ds_info_'+nametag+'.json', 'w') as file:
            json.dump(data, file)


if __name__ == "__main__":
    # ---- check if the dataset definition is correct: ----

    # Set random seed for NumPy, Pandas, and PyTorch
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # set up sources of RIRs and audios for dataset
    AUDIO_INFO_FILE = "/home/ubuntu/joanna/VAE-IR/audio_VCTK_datura.csv"
    IR_INFO_FILE = "/home/ubuntu/joanna/VAE-IR/irstats_ARNIandBUT_datura.csv"
    SAMPLING_RATE=48e3

    df_audiopool=pd.read_csv(AUDIO_INFO_FILE,index_col=0)
    df_irs=pd.read_csv(IR_INFO_FILE,index_col=0)
    df_irs=df_irs.head(10)

    # create a tag for dataset info file 
    dataset=Dataset_SpeechInSpace(df_audiopool,df_irs,sr=SAMPLING_RATE, ir_len=SAMPLING_RATE*2, 
                                  sig_len=int(SAMPLING_RATE*2.73), N_per_ir=1e4)
    
    # create a tag for dataset info file
    current_datetime = datetime.now()
    nametag = current_datetime.strftime("%d-%m-%Y_%H-%M")
    # save info about dataset
    dataset.save_dataset_info("",nametag)

    print("Number of data points:" + str(len(dataset)))
    print("Dimensions of input data:" + str(dataset[20][0].shape))
    print("List of labels:" + str(dataset[20][3]))
  


  