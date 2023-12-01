import torch
from tqdm import tqdm
from datetime import datetime
import time
import sys
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
        for transf_type, learn_rate, batch_size in combinations:
            file.write(f"Scheduled combination: {i},Transform. Type: {transf_type}, Learning Rate: {learn_rate}, Batch Size: {batch_size}\n")
            i+=1
        file.write(f" \n")
    file.close()
    # get lines of a text file (to track progress during experiment)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def note_down_finished_cond(file_path,line): 
    # Open the file again in write mode to overwrite its content with the appended word
    with open(file_path, 'a') as file:
            # Append the word to the end of each line and write it back to the file
            file.write("Finished combination: " + line.strip() + " \n" )


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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=6,pin_memory=True)
    # move models to gpu
    model_ReverbEncoder.to(args.device)
    model_waveunet.to(args.device)
    # ---- TRAINING ----
    train_and_test(model_ReverbEncoder, model_waveunet, trainloader, valloader, testloader, args)

if __name__ == "__main__":

    # load default arguments
    args = Options().parse()

    # Conditions of the experiment
    cond_trasf_type = ["many-to-many","one-to-many"]
    cond_learn_rate = [1e-3, 1e-4, 1e-5]
    cond_batch_size = [8, 24]

    # Conditions combinations list
    cond_combinations = list(product(cond_trasf_type, cond_learn_rate, cond_batch_size))

    # Save parameter combinations list to file
    date_tag = datetime.now().strftime("%d-%m-%Y--%H-%M")
    condfilepath = "runs/expconds_" + date_tag +".txt"
    lines=exp_combinations_to_file(cond_combinations,condfilepath)

    cond_count=0
    for transf_type, learn_rate, batch_size in cond_combinations:

        # prepare params for this combination
        if transf_type=="one-to-many":
            args.content_rir="anechoic"
            args.style_rir=None
        elif transf_type=="many-to-many":
            args.content_rir=None
            args.style_rir=None

        args.learn_rate=learn_rate
        args.batch_size=batch_size

        # create training tag based on date and params
        date_tag = datetime.now().strftime("%d-%m-%Y--%H-%M")
        lr_tag = "_lr-"+str(args.learn_rate)
        bs_tag = "_bs-"+str(args.batch_size)
        transf_tag="_"+transf_type
        tag=date_tag+transf_tag+lr_tag+bs_tag

        # prepare diectory for this training combination
        # runs/one-to-many14-11-2023--11-49_lr-0.001_bs-8/
        args.savedir="runs/" + tag +"/" 

        # train with current parameters
        setup_and_train(args)

        # note down finished condition
        note_down_finished_cond(condfilepath,lines[cond_count])
        cond_count+=1
        









    








