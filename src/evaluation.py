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
import numpy as np
import pandas as pd
import soundfile as sf
import random
# my modules
import dataset
import loss 
import models
import helpers as hlp
from options import OptionsEval
from torch.utils.data import Subset


class Evaluator(torch.nn.Module):
    def __init__(self,args_test):
        super().__init__()
        self.args_test=args_test
        self.args_train=torch.load(self.args_test.train_args_file)
        self.args_train.symmetric_film=self.args_test.symmetric_film
        # if we are on dacom we need to change the path of the dataset metadata (so far only this data was used)
        # self.args_train.df_metadata="/home/Imatge/projects/reverb-match-cond-u-net/dataset-metadata/nonoise2_dacom.csv"
        self.args_train.df_metadata="/home/ubuntu/joanna/reverb-match-cond-u-net/dataset-metadata/nonoise2_guestxr2.csv"
        self.load_eval_objects()
        self.failcount=0

    def load_eval_objects(self):
        # ---- MODELS: ----
        self.model=self.load_chosen_model(self.args_train.modeltype).to(self.args_test.device)
        # ---- LOSS CRITERIA: ----
        self.criterion_stft_loss = loss.MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024, 2048,4096],
            hop_sizes=[64, 128, 256,512,1024],
            win_lengths=[256, 512, 1024, 2048,4096],
            window="hann_window").to(self.args_test.device)
        self.criterion_logmel=loss.LogMelSpectrogramLoss().to(self.args_test.device)
        self.criterion_si_sdr = ScaleInvariantSignalDistortionRatio().to(self.args_test.device)
        self.criterion_srmr = SpeechReverberationModulationEnergyRatio(16000).to(self.args_test.device)
        self.criterion_mse = torch.nn.MSELoss().to(self.args_test.device)
        self.criterion_cosine = torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(self.args_test.device)
        self.criterion_pesq=PerceptualEvaluationSpeechQuality(16000, 'wb').to(self.args_test.device)
        self.criterion_stoi=ShortTimeObjectiveIntelligibility(16000).to(self.args_test.device)
        # ---- TRAINING RESULTS (WEIGHTS): ----
        self.train_results=torch.load(args_test.train_results_file,map_location=self.args_test.device)
        self.model.load_state_dict(self.train_results["model_state_dict"])
        # ---- DATASETS: ----
        self.args_train.split=self.args_test.eval_split
        self.testset_orig=dataset.DatasetReverbTransfer(self.args_train)
        indices_chosen=self.testset_orig.get_idx_with_rt60diff(self.args_test.rt60diffmin,self.args_test.rt60diffmax)
        if self.args_test.N_datapoints>0:
            # indices_chosen=random.sample(indices_chosen, self.args_test.N_datapoints)
            indices_chosen=range(0,self.args_test.N_datapoints)
        self.testset=Subset(self.testset_orig,indices_chosen)
        print(f"Preparing to evaluate {len(self.testset)} test datapoints")
        # ---- DATA LOADER: ----
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args_test.batch_size_eval, shuffle=True, num_workers=6,pin_memory=True)

    
    def infer(self,data):
        # Function to infer target audio
        with torch.no_grad():
            sContent_in = data[0].to(self.args_test.device)
            sStyle_in=data[1].to(self.args_test.device)
            sTarget_gt=data[2].to(self.args_test.device)
            sTarget_prediction=self.model(sContent_in,sStyle_in)
            return sContent_in, sStyle_in, sTarget_gt, sTarget_prediction
        
    def eval_losses_batchproc(config):

        df_subdir_losses=pd.DataFrame({'label':[],'idx':[], 'compared': [],
                    'L_stft2': [], 'L_logmel': [],'L_wav_L2': [], 
                    'L_si_sdr': [], 'L_srmr_x1': [],  'L_srmr_x2': [], 
                    'L_pesq': [], 'L_stoi': [],   
                    'L_emb_cosine': [], 'L_emb_euc': []})

        subdir_path = os.path.join(args_test.eval_dir, args_test.eval_subdir)

        print(f"Comparing losses for {subdir_path}")

        # load results from checkpoints in the directory
        for filename in os.listdir(subdir_path):
            if filename.startswith("checkpoint") & filename.endswith("best.pt"): # only computing measures for the best checkpoint
                with torch.no_grad():
                    # specify training params file
                    args_test.train_args_file=pjoin(subdir_path,"trainargs.pt")
                    # load checkpoint file
                    args_test.train_results_file=pjoin(subdir_path,filename)
                    # create tag for this evalauation
                    args_test.eval_tag=args_test.train_results_file.split('/')[-2]
                    # create evaluator object
                    tmp_evaluation=Evaluator(args_test)

                    for j, data in tqdm(enumerate(tmp_evaluation.testloader),total = len(tmp_evaluation.testloader)):
                        # get signals
                        sContent, sStyle, sTarget, sPrediction=tmp_evaluation.infer(data)
                        if bool(tmp_evaluation.args_train.is_vae):      
                            sPrediction, mu, log_var = sPrediction
                        # predicion : target
                        df_subdir_losses=pd.concat([df_subdir_losses, pd.DataFrame(tmp_evaluation.add_all_losses(j,"prediction:target",sPrediction,sTarget),index=[0])],ignore_index=True)
                        # content : target 
                        df_subdir_losses=pd.concat([df_subdir_losses, pd.DataFrame(tmp_evaluation.add_all_losses(j,"content:target",sContent,sTarget),index=[0])], ignore_index=True)
                        # predicion: content
                        df_subdir_losses=pd.concat([df_subdir_losses, pd.DataFrame(tmp_evaluation.add_all_losses(j,"prediction:content",sPrediction,sContent),index=[0])],ignore_index=True)

        df_subdir_losses["label"]=args_test.eval_tag
        df_subdir_losses.to_csv(pjoin(subdir_path, args_test.eval_tag + "_eval_batchproc.csv"), index=False)
        print(f"Saved subdir results")
        return df_subdir_losses
        
    

    def add_all_losses(self,idx,comp_name,x1,x2):

        if len(x1.shape)<3:
            x1=x1.unsqueeze(0)
        if len(x2.shape)<3:
            x2=x2.unsqueeze(0)
        x1_emb=self.model.conditioning_network(x1)
        x2_emb=self.model.conditioning_network(x2)

        x1=hlp.torch_resample_if_needed(x1,48000,16000).to(self.args_test.device)
        x2=hlp.torch_resample_if_needed(x2,48000,16000).to(self.args_test.device)


        # ----- Compute audio losses -----
        L_sc, L_mag = self.criterion_stft_loss(x1,x2)
        L_stft = L_sc + L_mag
        L_logmel=self.criterion_logmel(x1,x2)
        L_wav_L2=self.criterion_mse(x1,x2)
        L_si_sdr=self.criterion_si_sdr(x1,x2)
        if comp_name=="prediction:target":
            L_srmr=self.criterion_srmr(x1)
            L_srmr_name="L_srmr_x1"
        elif comp_name=="content:target":
            L_srmr=self.criterion_srmr(x2)
            L_srmr_name="L_srmr_x2"
        elif comp_name=="prediction:content":
            L_srmr=self.criterion_srmr(x2)
            L_srmr_name="L_srmr_x2"

        # ----- Compute embedding losses -----
        L_emb_cosine=(1-((torch.mean(self.criterion_cosine(x1_emb,x2_emb))+ 1) / 2))
        L_emb_euc=torch.dist(x1_emb,x2_emb)
        
        # ----- Compute perceptual losses -----
        L_pesq=torch.tensor([float('nan')])
        try:
            L_pesq=self.criterion_pesq(x1,x2)
        except: 
            self.failcount+=1
            print("Could not compute pesq for this signal for " + str(self.failcount) + "times")

        L_stoi=torch.tensor([float('nan')])
        try:
            L_stoi=self.criterion_stoi(x1,x2)
        except: 
            self.failcount+=1
            print("Could not compute stoi for this signal for " + str(self.failcount) + "times")


        # if torch.isnan(L_stoi) or torch.isnan(L_pesq):
            # for i in range(self.args_test.batch_size_eval):
            #     x1i=x1[i,0].detach().cpu().numpy()
            #     x2i=x2[i,0].detach().cpu().numpy()
            #     sf.write(f"input_sig_{self.failcount}_{i}.wav", x1i, 16000)
            #     sf.write(f"target_sig_{self.failcount}_{i}.wav", x2i, 16000)
            # print("Saved faulty audios")

        df_row={'idx':idx, 'compared': comp_name,
                'L_stft2': L_stft.item(), 'L_logmel': L_logmel.item(),'L_wav_L2': L_wav_L2.item(), 
                'L_si_sdr': L_si_sdr.item(), L_srmr_name: L_srmr.item(),
                'L_pesq': L_pesq.item(), 'L_stoi': L_stoi.item(),  
                'L_emb_cosine': L_emb_cosine.item(), 'L_emb_euc': L_emb_euc.item()}
        
        return df_row
        

    
    
def eval_condition(config,checkpoint_name):
        
    subdir_path=config["subdir_path"]
    eval_tag=pjoin(subdir_path).split('/')[-1]

    # load results from checkpoints in the directory
    for filename in os.listdir(subdir_path):
        if filename==checkpoint_name: 

            # specify training params file
            config_train=hlp.load_config(pjoin(subdir_path,"train_config.yaml"))

            # create evaluator object
            tmp_eval = Evaluator(config,config_train)
        
            # load checkpoint file
            checkpoint=pjoin(subdir_path,filename)

            # create evaluator object
            tmp_eval = Evaluator(config,config_train)

            # create evaluator object
            df_subdir_losses=tmp_eval.eval_losses_batchproc(checkpoint,eval_tag)


def eval_experiment(config):

    eval_dir=config["eval_dir"]

    for subdir in os.listdir(eval_dir):
        subdir_path = os.path.join(eval_dir, subdir)
        if os.path.isdir(subdir_path):

            print(f"Processing trainig results: {subdir_path}")
            config["subdir_path"]=subdir_path

             
            df_all_losses.to_csv(args_test.eval_dir+args_test.eval_file_name, index=False)
            print(f"Saved final results")


if __name__ == "__main__":

    config=hlp.load_config("basic.yaml")

    # Compute for all examples
    config["eval_dir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-19-04-2024/"
    config["evalsavedir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-19-04-2024/"
    config["N_datapoints"] = 0
    config["eval_file_name"] = "eval_all_batches.csv"
    config["rt60diffmin"] = -3
    config["rt60diffmax"] = 3
    config["N_datapoints"] = 0 # if 0 - whole test set included
    eval_experiment(config)

    # Compute for re-reverberation
    config["eval_file_name"] = "eval_rereverb.csv"
    config["rt60diffmin"] = -2
    config["rt60diffmax"] = -0.2
    config["N_datapoints"] = 0 # if 0 - whole test set included
    eval_experiment(config)

    # Compute for difficult de-reverberation
    config["eval_file_name"] = "eval_dereverb.csv"
    config["rt60diffmin"] = 0.2
    config["rt60diffmax"] = 2
    config["N_datapoints"] = 0 # if 0 - whole test set included
    eval_experiment(config)





    
            

            

           


   








