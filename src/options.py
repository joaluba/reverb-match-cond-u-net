import argparse
import torch
import os
import random
import numpy as np
import pandas as pd


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
        # parser.add_argument('--content_sig_len', default=98304, type=int)
        parser.add_argument('--enc_len', default=512, type=int)
        parser.add_argument('--btl_len', default=512, type=int)
        parser.add_argument('--gauss_len', default=3, type=int)
        parser.add_argument('--n_layers_revenc', default=8, type=int)
        parser.add_argument('--n_layers_enc', default=12, type=int)
        parser.add_argument('--n_layers_dec', default=7, type=int)
        parser.add_argument('--symmetric_film', default=1, type=int)


        # dataset parameters
        parser.add_argument('--style_rir', default="/home/ubuntu/Data/ACE-Single/Lecture_Room_1/1/Single_508_1_RIR.wav", type=str)
        parser.add_argument('--content_rir', default=None, type=str)
        parser.add_argument('--df_metadata', 
                            default="/home/ubuntu/joanna/reverb-match-cond-u-net/dataset-metadata/nonoise_48khz_guestxr.csv" , type=str)

        # training arguments
        parser.add_argument('--modeltype', default="c_fins", type=str)
        parser.add_argument('--num_epochs', default=30, type=int)
        parser.add_argument('--checkpoint_step', default=5, type=int)
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--learn_rate', default=1e-4, type=float)
        parser.add_argument('--optimizer', default="adam", type=str) # see below
        parser.add_argument('--losstype', default="stft", type=str)
        parser.add_argument('--loss_alphas', nargs='+', type=int, default=[1])
        parser.add_argument('--store_outputs', default=1, type=int)
        parser.add_argument('--split', default="train", type=str)

        # arguments to resume training
        parser.add_argument('--resume_checkpoint', default=None, type=str)
        parser.add_argument('--resume_tboard', default=None, type=str)



    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        torch.manual_seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        # print('| options')
        # for k, v in args.items():
        #     print('%s: %s' % (str(k), str(v)))
        # print()
        return self.opt


class OptionsEval():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        
        parser = self.parser

        # training arguments
        parser.add_argument('--device', default="cuda", type=str)
        parser.add_argument('--batch_size_eval', default=16, type=int)
        parser.add_argument('--checkpoint_development', default=False, type=bool)
        parser.add_argument('--eval_file_name', default="eval_all.csv", type=str)
        parser.add_argument('--eval_dir', default="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-15-01-2024/", type=str)
        parser.add_argument('--eval_split', default="test", type=str)
        parser.add_argument('--rt60diffmin', default=-3, type=float)
        parser.add_argument('--rt60diffmax', default=3, type=float)
        parser.add_argument('--N_datapoints', default=1000, type=int)
        parser.add_argument('--train_results_file', 
                            default="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-15-01-2024/18-01-2024--00-56_many-to-many_stft/checkpoint_best.pt", type=str)
        parser.add_argument('--eval_tag', default="18-01-2024--00-56_many-to-many_stft", type=str)
        parser.add_argument('--train_args_file', 
                    default="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-15-01-2024/18-01-2024--00-56_many-to-many_stft/trainargs.pt", type=str)
        parser.add_argument('--savedir', default="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-15-01-2024/", type=str)

    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        torch.manual_seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        # print('| options')
        # for k, v in args.items():
        #     print('%s: %s' % (str(k), str(v)))
        # print()
        return self.opt
