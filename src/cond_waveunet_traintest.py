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
from cond_waveunet_options import Options
import argparse
import pandas as pd
import getpass
import joa_helpers
from datetime import datetime
import time


def infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, data,device):
    # Function to infer target and compute loss - same for training, testing and validation
    # -------------------------------------------------------------------------------------
    # get datapoint
    sContent_in = data[0].to(device)
    sStyle_in=data[1].to(device)
    sTarget_gt=data[2].to(device)
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


def train_and_test(model_reverbenc, model_waveunet, trainloader, valloader, testloader, args):
    
    # create training tag based on date
    tag = datetime.now().strftime("%d-%m-%Y--%H-%M")
    # training parameters
    device = args.device
    if args.audio_criterion=="multi_stft_loss":
        audio_criterion = cond_waveunet_loss.MultiResolutionSTFTLoss()
    if args.emb_criterion=="cosine_similarity":
        emb_criterion = torch.nn.CosineSimilarity(dim=2,eps=1e-8)
    if args.optimizer=="adam":
        optimizer_waveunet =  torch.optim.AdamW(model_waveunet.parameters(), args.learn_rate)
        optimizer_reverbenc =  torch.optim.AdamW(model_reverbenc.parameters(), args.learn_rate)
    num_epochs = args.num_epochs
    savedir = args.savedir
    store_outputs = args.store_outputs
    

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
            loss=infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, data,device)
            # empty gradient
            optimizer_waveunet.zero_grad()
            optimizer_reverbenc.zero_grad()
            # compute gradients 
            loss.backward()
            # update weights
            optimizer_waveunet.step()
            optimizer_reverbenc.step()
            # compute loss for the current batch
            train_loss += loss.item()

        # ----- Validation loop for this epoch: -----
        model_waveunet.eval() 
        model_reverbenc.eval()
        with torch.no_grad():
            val_loss=0
            for j,data in tqdm(enumerate(valloader),total = len(valloader)):
                 # infer and compute loss
                loss=infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, data,device)
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
        if (store_outputs) & (epoch % 3 ==0):
            torch.save({
                        'epoch': epoch,
                        'model_waveunet_state_dict': model_waveunet.state_dict(),
                        'model_reverbenc_state_dict': model_reverbenc.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_evol,
                        }, savedir+tag+'.pt')
  
    end=time.time()
    print(f"Finished training after: {(end-start)} seconds")
    # Save parameters
    if (store_outputs):
        torch.save(args, savedir+tag+'_args.pt')    

    # ------------- TESTING START: -------------
    model_waveunet.eval() 
    model_reverbenc.eval()
    with torch.no_grad():
        test_loss=0
        for j,data in tqdm(enumerate(testloader),total = len(testloader)):
            # infer and compute loss
            loss=infer_and_compute_loss(model_reverbenc, model_waveunet, emb_criterion, audio_criterion, data,device)
            # compute loss for the current batch
            test_loss += loss.item()

    # Print stats at the end of the epoch
    num_samples_test=len(testloader.sampler)
    avg_test_loss = test_loss / num_samples_test
    print(f'Test. Loss: {avg_test_loss:.5f}')



if __name__ == "__main__":
    # ---- test training loop ----

    args = Options().parse()

    # ---- MODEL: ----
    # load reverb encoder
    model_ReverbEncoder=cond_waveunet_model.ReverbEncoder(args)
    model_ReverbEncoder.to("cuda")
    # check waveunet 
    model_waveunet=cond_waveunet_model.waveunet(args)
    model_waveunet.to("cuda")

    # ---- DATASET: ----
    args.split="train"
    trainset=cond_waveunet_dataset.DatasetReverbTransfer(args)
    args.split="test"
    testset=cond_waveunet_dataset.DatasetReverbTransfer(args)
    args.split="val"
    valset=cond_waveunet_dataset.DatasetReverbTransfer(args)

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)

    # --------------------- Training: ---------------------
    train_and_test(model_ReverbEncoder, model_waveunet, trainloader, valloader, testloader, args)








