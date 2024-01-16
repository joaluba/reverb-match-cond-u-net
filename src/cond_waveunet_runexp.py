import torch
from tqdm import tqdm
from datetime import datetime
import time
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import json
# my modules
import cond_waveunet_dataset
import cond_waveunet_model
from cond_waveunet_traintest import train_and_test
from cond_waveunet_options import Options

def exp_combinations_to_file(combinations, file_path): 
    i=1
    # save combinations to text file
    with open(file_path, 'w') as file:
        for transf_type, loss in combinations:
            file.write(f"Scheduled combination: {i},Transform. Type: {transf_type}, Loss: {loss}\n")
            i+=1
        file.write(f" \n")
    file.close()
    # get lines of a text file (to track progress during experiment)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def note_down_finished_cond(file_path,line,wheresaved): 
    # Open the file again in write mode to overwrite its content with the appended word
    with open(file_path, 'a') as file:
            # Append the word to the end of each line and write it back to the file
            new_line = line.strip().replace("Scheduled", "Finished")
            file.write(new_line + ". \n"+ "-> Saved as:" + wheresaved + " \n" )


def setup_and_train(args):
    # ---- MODEL: ----
    # load reverb encoder
    model_ReverbEncoder=cond_waveunet_model.ReverbEncoder(args)
    # laod waveunet 
    model_waveunet=cond_waveunet_model.waveunet(args)
    # ---- DATASET: ----
    args.split="train"
    trainset=cond_waveunet_dataset.DatasetReverbTransfer(args)
    args.split="test"
    testset=cond_waveunet_dataset.DatasetReverbTransfer(args)
    args.split="val"
    valset=cond_waveunet_dataset.DatasetReverbTransfer(args)
    # create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=6,pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=6,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=6,pin_memory=True)
    # move models to gpu
    model_ReverbEncoder.to(args.device)
    model_waveunet.to(args.device)
    # ---- TRAINING ----
    train_and_test(model_ReverbEncoder, model_waveunet, trainloader, valloader, testloader, args)

if __name__ == "__main__":

    # load default arguments
    args = Options().parse()

    # Conditions of the experiment
    cond_trasf_type = ["one-to-many", "many-to-many"]
    cond_losses=["stft","stft+rev","stft+emb","stft+rev+emb"]

    # Conditions combinations list
    cond_combinations = list(product(cond_trasf_type, cond_losses))

    # Create folder for storing results of this experiment
    date_tag = datetime.now().strftime("%d-%m-%Y")
    runexp_savepath=os.path.join(args.savedir,"runs-exp-"+date_tag)
    if not os.path.exists(runexp_savepath):
        os.makedirs(runexp_savepath)

    # Save parameter combinations list to file (to track progress of training)
    condfilepath = os.path.join(runexp_savepath,"expconds_" + date_tag +".txt")
    lines=exp_combinations_to_file(cond_combinations,condfilepath)

    cond_count=0
    for transf_type, loss in cond_combinations:

        # prepare params for this combination

        if transf_type=="one-to-many":
            args.content_rir="anechoic"
            args.style_rir=None
            # best params for one-to-many
            args.learn_rate = 1e-4
            args.batch_size = 8
        elif transf_type=="many-to-many":
            args.content_rir=None
            args.style_rir=None
            # best params for many-to-many
            args.learn_rate = 1e-4
            args.batch_size = 24

        args.losstype=loss

        # create training tag based on date and params
        date_tag = datetime.now().strftime("%d-%m-%Y--%H-%M")
        loss_tag = "_"+ loss
        transf_tag="_"+transf_type
        tag=date_tag+transf_tag+loss_tag

        # prepare diectory for this training combination
        args.savedir=os.path.join(runexp_savepath,tag) 

        # train with current parameters
        setup_and_train(args)

        # note down finished condition
        note_down_finished_cond(condfilepath,lines[cond_count], args.savedir)
        cond_count+=1
        






    








