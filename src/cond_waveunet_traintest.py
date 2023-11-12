import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import cond_waveunet_dataset
import cond_waveunet_loss
import cond_waveunet_model
import argparse
import pandas as pd
import getpass
import joa_helpers
from datetime import datetime
import time


def infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, data):
    # Function to infer target and compute loss - same for training, testing and validation
    # -------------------------------------------------------------------------------------
    # get datapoint
    sContent_in = data[0]
    sStyle_in=data[1]
    sTarget_gt=data[2]
    # forward pass - get prediction of the ir
    embedding_gt=model_reverbenc(sStyle_in)
    sTarget_prediction=model_waveunet(sContent_in,embedding_gt)
    # loss
    embedding_prediction=model_reverbenc(sTarget_prediction)
    L_sc, L_mag = audio_criterion(sTarget_gt.squeeze(1), sTarget_prediction.squeeze(1))
    L_emb=-torch.mean(emb_criterion(embedding_gt,embedding_prediction))
    # loss 
    loss=L_sc + L_mag +  L_emb
    return loss


def train_and_test(model_reverbenc, model_waveunet, trainloader, valloader, testloader, trainparams, store_outputs):
    
    # create training tag based on date
    tag = datetime.now().strftime("%d-%m-%Y--%H-%M")
    # training parameters
    device = trainparams["device"]
    audio_criterion = trainparams["audio_criterion"]
    emb_criterion = trainparams["emb_criterion"]
    optimizer = trainparams["optimizer"]
    num_epochs = trainparams["num_epochs"]
    datasavepath = trainparams["datasavepath"]

    # move components to device
    audio_criterion=audio_criterion.to(device)
    emb_criterion=emb_criterion.to(device)
    model_reverbenc=model_reverbenc.to(device)
    model_waveunet=model_waveunet.to(device)

    # allocate variable to track loss evolution 
    loss_evol=[]
    
    # ------------- TRAINING START: -------------
    start = time.time()
    for epoch in range(num_epochs):


        # ----- Training loop for this epoch: -----
        model_waveunet.train()
        model_reverbenc.train()
        train_loss=0
        for j,data in tqdm(enumerate(trainloader),total = len(trainloader)):
            # infer and compute loss
            loss=infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, data)
            # empty gradient
            optimizer.zero_grad()
            # compute gradients 
            loss.backward()
            # update weights
            optimizer.step()
            # compute loss for the current batch
            train_loss += loss.item()

        # ----- Validation loop for this epoch: -----
        model_waveunet.eval() 
        model_reverbenc.eval()
        with torch.no_grad():
            val_loss=0
            for j,data in tqdm(enumerate(valloader),total = len(valloader)):
                 # infer and compute loss
                loss=infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, data)
                # compute loss for the current batch
                val_loss += loss.item()
        
        # Print stats at the end of the epoch
        num_samples_train=len(trainloader.sampler)
        num_samples_val=len(valloader.sampler)
        avg_train_loss = train_loss / num_samples_train
        avg_val_loss = val_loss / num_samples_val
        loss_evol.append((avg_train_loss,avg_val_loss))
        print(f'Epoch: {epoch}, Train. Loss: {avg_train_loss:.5f}, Val. Loss: {avg_val_loss:.5f}')
  
        # Save checkpoint
        if (store_outputs==1) & (epoch % 3 ==0):
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_evol,
                        }, datasavepath+tag+'.pt')
            
    end=time.time()
    print(f"Finished training after: {(end-start)} seconds")

    # ------------- TESTING START: -------------
    model.eval() 
    with torch.no_grad():
        test_loss=0
        for j,data in tqdm(enumerate(testloader),total = len(testloader)):
            # infer and compute loss
            loss=infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, data)
            # compute loss for the current batch
            test_loss += loss.item()

    # Print stats at the end of the epoch
    num_samples_test=len(testloader.sampler)
    avg_test_loss = test_loss / num_samples_test
    print(f'Test. Loss: {avg_test_loss:.5f}')



if __name__ == "__main__":
    # ---- check if training loop is correct ----

    # current directory
    projectdir="/home/ubuntu/joanna/VAE-IR/"

    # --------------------- Model: ---------------------
    FS=22050
    Z_LEN=512
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "mps")    
    SIG_LEN_SMPL=98304
    N_LAYERS_REVENC=3
    N_LAYERS_WAVEUNET=12

    # load reverb encoder
    model_ReverbEncoder=cond_waveunet_model.ReverbEncoder(x_len=SIG_LEN_SMPL, z_len=Z_LEN, N_layers=N_LAYERS_REVENC)
    model_ReverbEncoder.to("cuda")
    model_ReverbEncoder.eval
    # check waveunet 
    model_waveunet=cond_waveunet_model.waveunet(n_layers=N_LAYERS_WAVEUNET,channels_interval=24,z_channels=Z_LEN)
    model_waveunet.to("cuda")
    model_waveunet.eval


    # --------------------- Parameters: ---------------------

    LEARNRATE=1e-4
    N_EPOCHS=4
    BATCH_SIZE=16

    TRAINPARAMS={
        "num_epochs": N_EPOCHS, 
        "device": DEVICE,
        "batchsize": BATCH_SIZE,
        "learnrate":LEARNRATE,
        "optimizer": torch.optim.Adam(model_waveunet.parameters(), LEARNRATE),
        "audio_criterion": cond_waveunet_loss.MultiResolutionSTFTLoss(),
        "emb_criterion": torch.nn.CosineSimilarity(dim=2,eps=1e-8),
        "datasavepath": projectdir + 'models/'
        }
    
    # --------------------- Dataset: ---------------------

    STYLE_RIR ="/home/ubuntu/Data/ACE-Single/Lecture_Room_1/1/Single_508_1_RIR.wav"
    CONTENT_RIR="anechoic"

    DF_METADATA="/home/ubuntu/joanna/reverb-match-cond-u-net/notebooks/data_set_check.csv" 
  
    df_ds=pd.read_csv(DF_METADATA,index_col=0)

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    trainset=cond_waveunet_dataset.DatasetReverbTransfer(df_ds,sr=48e3,sig_len=SIG_LEN_SMPL,split="train",content_ir=CONTENT_RIR,style_ir=STYLE_RIR,device=DEVICE)
    testset=cond_waveunet_dataset.DatasetReverbTransfer(df_ds,sr=48e3,sig_len=SIG_LEN_SMPL,split="test",content_ir=CONTENT_RIR,style_ir=STYLE_RIR,device=DEVICE)
    valset=cond_waveunet_dataset.DatasetReverbTransfer(df_ds,sr=48e3,sig_len=SIG_LEN_SMPL,split="val",content_ir=CONTENT_RIR,style_ir=STYLE_RIR,device=DEVICE)

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)



    # --------------------- Training: ---------------------
    train_and_test(model_ReverbEncoder, model_waveunet, trainloader, valloader, testloader, TRAINPARAMS, 1)








