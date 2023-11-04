import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import sig2ir_datasetprep as sig2ir_datasetprep
import sig2ir_loss
import rev2rev_custom_model
import argparse
import pandas as pd
import getpass
import helpers
from datetime import datetime
import time

def train_and_test(model, trainloader, valloader, testloader, trainparams, store_outputs):
    
    # create training tag based on date
    tag = datetime.now().strftime("%d-%m-%Y--%H-%M")
    # training parameters
    criterion = trainparams["criterion"]
    optimizer = trainparams["optimizer"]
    num_epochs = trainparams["num_epochs"]
    device = trainparams["device"]
    datasavepath = trainparams["datasavepath"]

    criterion=criterion.to(device)
    model=model.to(device)

    # allocate variable to track loss evolution 
    loss_evol=[]

    
    # ------------- TRAINING START: -------------
    start = time.time()
    for epoch in range(num_epochs):


        # ----- Training loop for this epoch: -----
        model.train()
        train_loss=0
        for j,data in tqdm(enumerate(trainloader),total = len(trainloader)):
            # get datapoint
            x_orig = data[0].to(device)
            ir_target=data[1].to(device)
            # forward pass - get prediction of the ir
            ir_predict = model(x_orig)
            # loss 
            sc_loss, mag_loss = criterion(ir_predict.squeeze(1), ir_target[:,:,:ir_predict.shape[2]].squeeze(1))
            loss= sc_loss+mag_loss
            # empty gradient
            optimizer.zero_grad()
            # compute gradients 
            loss.backward()
            # update weights
            optimizer.step()
            # compute loss for the current batch
            train_loss += loss.item()


        # ----- Validation loop for this epoch: -----
        model.eval() 
        with torch.no_grad():
            val_loss=0
            for j,data in tqdm(enumerate(valloader),total = len(valloader)):
                # get datapoint
                x_orig = data[0].to(device)
                ir_target=data[1].to(device)
                # forward pass - get prediction of the ir
                ir_predict = model(x_orig)
                # loss 
                sc_loss, mag_loss = criterion(ir_predict.squeeze(1), ir_target[:,:,:ir_predict.shape[2]].squeeze(1))
                loss= sc_loss+mag_loss
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
    print(f"Finished training after: {(end-start)} seconds" )

    # ------------- TESTING START: -------------
    model.eval() 
    with torch.no_grad():
        test_loss=0
        for j,data in tqdm(enumerate(testloader),total = len(testloader)):
            # get datapoint
            x_orig = data[0].to(device)
            ir_target=data[1].to(device)
            # forward pass - get prediction of the ir
            ir_predict = model(x_orig)
            # loss 
            sc_loss, mag_loss = criterion(ir_predict.squeeze(1), ir_target[:,:,:ir_predict.shape[2]].squeeze(1))
            loss= sc_loss+mag_loss
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
    N_EPOCHS=4
    BATCH_SIZE=16

    TRAINPARAMS={
        "num_epochs": N_EPOCHS, 
        "device": DEVICE,
        "batchsize": BATCH_SIZE,
        "learnrate":LEARNRATE,
        "optimizer": torch.optim.Adam(model.parameters(), LEARNRATE),
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
    torch.cuda.empty_cache()

    # set up sources of RIRs and audios for dataset
    AUDIO_INFO_FILE = "/home/ubuntu/joanna/VAE-IR/audio_VCTK_datura.csv"
    IR_INFO_FILE = "/home/ubuntu/joanna/VAE-IR/irstats_ARNIandBUT_datura.csv"

    df_audiopool=pd.read_csv(AUDIO_INFO_FILE,index_col=0)
    df_irs=pd.read_csv(IR_INFO_FILE,index_col=0)
    df_irs=df_irs.head(10)

    # create a tag for dataset info file 
    dataset=sig2ir_datasetprep.Dataset_SpeechInSpace(df_audiopool,df_irs,sr=FS, ir_len=FS*2, 
                                  sig_len=SIG_LEN, N_per_ir=1e1)
    
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
    train_and_test(model, trainloader, valloader, testloader, TRAINPARAMS, 1)








