import argparse
import torch
import os
import random
import numpy as np
import pandas as pd

def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def list_float_flag(s):
    return [float(_) for _ in list(s)]

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        
        parser = self.parser

        # general arguments
        parser.add_argument('--projectdir', default="/home/ubuntu/joanna/reverb-match-cond-u-net/", type=str)
        parser.add_argument('--savedir', default="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/", type=str)
        parser.add_argument('--device', default="cuda", type=str)
        parser.add_argument('--fs', default=48000, type=int)

        # data and model parameters
        parser.add_argument('--sig_len', default=98304, type=int)
        parser.add_argument('--enc_len', default=512, type=int)
        parser.add_argument('--n_layers_revenc', default=8, type=int)
        parser.add_argument('--n_layers_waveunet', default=12, type=int)

        # dataset parameters
        parser.add_argument('--style_rir', default="/home/ubuntu/Data/ACE-Single/Lecture_Room_1/1/Single_508_1_RIR.wav", type=str)
        parser.add_argument('--content_rir', default=None, type=str)
        parser.add_argument('--df_metadata', 
                            default="/home/ubuntu/joanna/reverb-match-cond-u-net/notebooks/nonoise2_data_set.csv" , type=str)
        # training arguments
        parser.add_argument('--num_epochs', default=300, type=int)
        parser.add_argument('--checkpoint_step', default=30, type=int)
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--learn_rate', default=1e-4, type=float)
        parser.add_argument('--optimizer', default="adam", type=str) # see below
        parser.add_argument('--audio_criterion', default="multi_stft_loss", type=str)
        parser.add_argument('--emb_criterion', default="cosine_similarity", type=str)
        parser.add_argument('--store_outputs', default=True, type=bool)
        parser.add_argument('--split', default="test", type=str)


    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.df_ds=pd.read_csv(self.opt.df_metadata,index_col=0)
        torch.manual_seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print()
        return self.opt
