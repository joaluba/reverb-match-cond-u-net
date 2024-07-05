import torch
from tqdm import tqdm
from datetime import datetime
import time
import sys
import os
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
import speechmetrics
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SpeechReverberationModulationEnergyRatio, PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import numpy as np
import pandas as pd
import soundfile as sf
import random
# my modules
import dataset
import baselines
import loss_mel, loss_stft, loss_waveform
import trainer
import evaluation
import helpers as hlp
from torch.utils.data import Subset


if __name__ == "__main__":

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    config=hlp.load_config(pjoin("/home/ubuntu/joanna/reverb-match-cond-u-net/config/basic.yaml"))
    config["eval_dir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/"
    config["N_datapoints"] = 0 # if 0 - whole test set included

    # Compute for a baseline 1
    config["baseline"]= "wpe+fins"

    config["eval_file_name"] = "wpe+fins_eval_rereverb.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/rereverb/"
    config["rt60diffmin"] = 0.2
    config["rt60diffmax"] = 2
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")

    config["eval_file_name"] = "wpe+fins_eval_dereverb.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/dereverb/"
    config["rt60diffmin"] = -2
    config["rt60diffmax"] = -0.2
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")


    config["eval_file_name"] = "wpe+fins_eval_all_batches.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/all_batches/"
    config["rt60diffmin"] = -3
    config["rt60diffmax"] = 3
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")

    # Compute for a baseline 2
    config["baseline"]= "anecho+fins"

    config["eval_file_name"] = "anecho+fins_eval_rereverb.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/rereverb/"
    config["rt60diffmin"] = 0.2
    config["rt60diffmax"] = 2
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")

    config["eval_file_name"] = "anecho+fins_eval_dereverb.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/dereverb/"
    config["rt60diffmin"] = -2
    config["rt60diffmax"] = -0.2
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")

    config["eval_file_name"] = "anecho+fins_eval_all_batches.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/all_batches/"
    config["rt60diffmin"] = -3
    config["rt60diffmax"] = 3
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")


    # Compute for a baseline 3
    config["baseline"]= "dfnet+fins"

    config["eval_file_name"] = "dfnet+fins_eval_rereverb.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/rereverb/"
    config["rt60diffmin"] = 0.2
    config["rt60diffmax"] = 2
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")

    config["eval_file_name"] = "dfnet+fins_eval_dereverb.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/dereverb/"
    config["rt60diffmin"] = -2
    config["rt60diffmax"] = -0.2
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")

    config["eval_file_name"] = "dfnet+fins_eval_all_batches.csv"
    config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/all_batches/"
    config["rt60diffmin"] = -3
    config["rt60diffmax"] = 3
    eval_dict=evaluation.eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")


           


   








