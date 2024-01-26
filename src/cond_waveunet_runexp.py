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
import cond_waveunet_trainer
from cond_waveunet_options import Options


def exp_combinations_to_file(combinations, file_path):
    # Save combinations to text file
    with open(file_path, 'w') as file:
        for i, combination in enumerate(combinations, start=1):
            line = f"Scheduled combination: {i},"
            for condition_name, condition_value in combination.items():
                line += f" {condition_name}: {condition_value},"
            # Remove the trailing comma
            line = line.rstrip(',')
            file.write(line + '\n')
        file.write("\n")

    # Get lines of a text file (to track progress during experiment)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    return lines

def note_down_finished_cond(file_path,line,wheresaved): 
    # Open the file again in write mode to overwrite its content with the appended word
    with open(file_path, 'a') as file:
            # Append the word to the end of each line and write it back to the file
            new_line = line.strip().replace("Scheduled", "Finished")
            file.write(new_line + ". \n"+ "-> Saved as:" + wheresaved + " \n" )

if __name__ == "__main__":

    # load default arguments
    args = Options().parse()

    # Conditions of the experiment
    cond_trasf_type = ["one-to-many","many-to-many"]
    cond_losses=["stft"]

    # Conditions combinations list
    from itertools import product

    # Generate all combinations
    cond_combinations = []
    for combo in product(cond_trasf_type, cond_losses):
        combination_dict = {
            'cond_trasf_type': combo[0],
            'cond_losses': combo[1]
        }
        cond_combinations.append(combination_dict)

    # Create folder for storing results of this experiment
    date_tag = datetime.now().strftime("%d-%m-%Y")
    runexp_savepath=os.path.join(args.savedir,"runs-exp-"+date_tag)
    if not os.path.exists(runexp_savepath):
        os.makedirs(runexp_savepath)

    # Save parameter combinations list to file (to track progress of training)
    condfilepath = os.path.join(runexp_savepath,"expconds_" + date_tag +".txt")
    lines=exp_combinations_to_file(cond_combinations,condfilepath)

    cond_count=0
    for i, combination in enumerate(cond_combinations, start=1):

        # prepare params for this combination
        loss=combination["cond_losses"]
        transf_type=combination["cond_trasf_type"]

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
        transf_tag="_"+ transf_type
        tag=date_tag+transf_tag+loss_tag

        # prepare diectory for this training combination
        args.savedir=os.path.join(runexp_savepath,tag) 

        # train with current parameters
        new_experiment=cond_waveunet_trainer.Trainer(args)
        new_experiment.train()

        # note down finished condition
        note_down_finished_cond(condfilepath,lines[cond_count], args.savedir)
        cond_count+=1
        






    








