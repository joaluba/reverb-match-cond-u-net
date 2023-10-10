import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import pandas as pd
import getpass
import helpers
from datetime import datetime
import time
# my modules: 
import sig2ir_datasetprep 
import sig2ir_loss 
import rev2rev_custom_model 
import sig2ir_traintest 


if __name__ == "__main__":

    # argument parser - for later 
    # parser = argparse.ArgumentParser(description='Sig2Ir Training')
    # parser.add_argument('--N_per_ir',help='number of audio samples per each impulse response', type=int)
    # args = parser.parse_args()

    # current directory
    projectdir="/home/ubuntu/joanna/VAE-IR/"

    # --------------------- Model: ---------------------
    FS=48000
    SIG_LEN=int(2.73*FS)
    L_LEN=512
    V_LEN=400 
    Z_LEN=512*2
    IR_LEN=FS
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "mps")    
    
    model=rev2rev_custom_model.sig2ir_encdec(sig_len=SIG_LEN, l_len=L_LEN, v_len=V_LEN, z_len=Z_LEN, ir_len=IR_LEN,device=DEVICE)

    # --------------------- Parameters: ---------------------

    LEARNRATE=1e-4
    N_EPOCHS=10
    BATCH_SIZE=16

    TRAINPARAMS={
        "num_epochs": N_EPOCHS, 
        "device": DEVICE,
        "batchsize": BATCH_SIZE,
        "learnrate":LEARNRATE,
        "optimizer": torch.optim.AdamW(model.parameters(), LEARNRATE),
        "criterion": sig2ir_loss.MultiResolutionSTFTLoss(),
        "datasavepath": projectdir + 'models/'
        }
    
    # --------------------- Dataset: ---------------------

    # Set random seed for NumPy, Pandas, and PyTorch
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # set up sources of RIRs and audios for dataset
    AUDIO_INFO_FILE = "/home/ubuntu/joanna/VAE-IR/audio_VCTK_datura.csv"
    IR_INFO_FILE = "/home/ubuntu/joanna/VAE-IR/irstats_ARNIandBUT_datura.csv"

    df_audiopool=pd.read_csv(AUDIO_INFO_FILE,index_col=0)
    df_irs=pd.read_csv(IR_INFO_FILE,index_col=0)
    # df_irs=df_irs[df_irs["database_ir"]=="arni"]
    # df_irs=df_irs.sample(10)

    # create a tag for dataset info file 
    dataset=sig2ir_datasetprep.Dataset_SpeechInSpace(df_audiopool,df_irs,sr=FS, ir_len=FS*2, sig_len=SIG_LEN, N_per_ir=1e2)
    # split dataset into training set, test set and validation set
    N_train = round(len(dataset) * 0.5)
    N_rest = len(dataset) - N_train
    trainset, restset = random_split(dataset, [N_train, N_rest])
    N_test = round(len(restset) * 0.5)
    N_val = len(restset) - N_test
    testset, valset = random_split(restset, [N_test, N_val])
    
    # save info about dataset
    current_datetime = datetime.now()
    nametag = current_datetime.strftime("%d-%m-%Y_%H-%M")
    dataset.save_dataset_info(TRAINPARAMS["datasavepath"],nametag)

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,pin_memory=True)

    # --------------------- Training: ---------------------
    sig2ir_traintest.train_and_test(model, trainloader, valloader, testloader, TRAINPARAMS, 1)








