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


def get_msg_for_exp_log(message):
    # Get current date and time
    current_date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    # Get user input
    user_input = input(message + ": ")
    # Format message with current date
    message_with_date = f"[{current_date}] {user_input}\n"
    return message_with_date


if __name__ == "__main__":


    # Prompt user for input with a message
    user_message = get_msg_for_exp_log("Enter info for experiment log")

    # Write the message to a file
    with open("/home/ubuntu/joanna/reverb-match-cond-u-net/experiment_log.txt", "a") as file:
        file.write(user_message)

    # load default arguments
    args = Options().parse()


    # Conditions of the experiment to permute
    perm_cond_trasf_type = ["many-to-many"]
    perm_cond_losses=["stft+vae", "logmel+vae", "stft","logmel","stft+emb"]
    # Additional conditions
    cond_alphas=[[1,1],[1,1],[1],[1],[1,1]]
    cond_is_vae=[1,1,0,0,0]

    # Conditions combinations list
    from itertools import product

    # Generate all combinations
    cond_combinations = []
    for combo in product(perm_cond_trasf_type, perm_cond_losses):
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
    for i, combination in enumerate(cond_combinations, start=0):

        # prepare params for this combination
        loss=combination["cond_losses"]
        transf_type=combination["cond_trasf_type"]
        alphas=cond_alphas[i]

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
            args.batch_size = 8

        args.is_vae=cond_is_vae[i]
        args.losstype=loss
        args.loss_alphas=alphas

        # create training tag based on date and params
        date_tag = datetime.now().strftime("%d-%m-%Y--%H-%M")
        loss_tag = "_"+ loss
        transf_tag="_"+ transf_type
        alpha_tag="_"+ '_'.join(map(str, alphas))
        tag=date_tag+transf_tag+loss_tag+alpha_tag
        # prepare diectory for this training combination
        args.savedir=os.path.join(runexp_savepath,tag) 

        # # resume training 
        # args.savedir="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-19-02-2024/19-02-2024--14-58_many-to-many_stft+rev+emb_1_1_1"
        # args.resume_checkpoint="checkpointbest.pt"
        # args.resume_tboard=args.savedir

        # train with current parameters
        new_experiment=cond_waveunet_trainer.Trainer(args)
        new_experiment.train()

        # note down finished condition
        note_down_finished_cond(condfilepath,lines[cond_count], args.savedir)
        cond_count+=1
        






    








