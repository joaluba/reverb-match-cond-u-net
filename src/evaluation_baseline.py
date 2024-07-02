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
import baselines
import loss_mel, loss_stft, loss_waveform
import trainer
import helpers as hlp
from torch.utils.data import Subset

class Evaluator(torch.nn.Module):
    def __init__(self,config,config_train):
        super().__init__()
        self.config=config
        self.config_train=config_train
        self.model =trainer.load_chosen_model(config_train,config_train["modeltype"]).to(config["device"])
        self.baseline=baselines.Baseline(config)
        self.load_measures()
        self.load_eval_dataset()
        self.failcount=0

    
    def load_measures(self):
        device=self.config["device"]

        self.measures= {
            "multi-stft" : loss_stft.MultiResolutionSTFTLoss().to(device), 
            "multi-mel"  : loss_mel.MultiMelSpectrogramLoss().to(device),
            "multi-wave" : loss_waveform.MultiWindowShapeLoss().to(device),
            "stft" : loss_stft.STFTLoss().to(device), 
            "logmel"  : loss_mel.LogMelSpectrogramLoss().to(device),
            "wave" : loss_waveform.WaveformShapeLoss(winlen=960).to(device),
            "mse" : torch.nn.MSELoss().to(device),
            "sisdr" : ScaleInvariantSignalDistortionRatio().to(device),
            "srmr" : SpeechReverberationModulationEnergyRatio(16000).to(device),
            "pesq" : PerceptualEvaluationSpeechQuality(16000, 'wb').to(device),
            "stoi" : ShortTimeObjectiveIntelligibility(16000).to(device),
            "emb_cos": torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(device)
        }

    def load_eval_dataset(self):

        rt60diffmin = self.config["rt60diffmin"]
        rt60diffmax = self.config["rt60diffmax"]
        N_datapoints = self.config["N_datapoints"]
        batch_size_eval = self.config["batch_size_eval"]

        # load a test split from the dataset used for training
        self.config_train["split"] = "test" 
        self.config_train["p_noise"] = 0 # for evaluation, we do not want noise 
        self.testset_orig = dataset.DatasetReverbTransfer(self.config_train)
        # choose a subset of the original test split
        indices_chosen = self.testset_orig.get_idx_with_rt60diff(rt60diffmin,rt60diffmax)
        if N_datapoints>0:
            indices_chosen = range(0,N_datapoints)
        self.testset = Subset(self.testset_orig,indices_chosen)
        print(f"Preparing to evaluate {len(self.testset)} test datapoints")
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size_eval, shuffle=True, num_workers=6,pin_memory=True)

    def infer(self,data):
        device= self.config["device"]
        # Function to infer target audio
        with torch.no_grad():
            sContent_in = data[0].to(device)
            sStyle_in=data[1].to(device)
            sTarget_gt=data[2].to(device)
            sTarget_prediction=self.model(sContent_in,sStyle_in)
            return sContent_in, sStyle_in, sTarget_gt, sTarget_prediction
    
    def compute_losses(self,checkpointpath,eval_tag):

        train_results=torch.load(os.path.join(checkpointpath),map_location=self.config["device"])
        self.model.load_state_dict(train_results["model_state_dict"])

        eval_dict_list=[]
        for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):
            # get signals
            sContent, sStyle, sTarget, sPrediction=self.infer(data)
            if bool(self.config_train["is_vae"]):      
                sPrediction, mu, log_var = sPrediction
             
            # predicion : target
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"prediction:target",sPrediction,sTarget))
            # content : target 
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"content:target",sContent,sTarget))
            # predicion: content
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"prediction:content",sPrediction,sContent))

        return eval_dict_list
    
    def compute_losses_baseline(self,eval_tag):

        eval_dict_list=[]
        for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):
            # get signals
            sContent, sStyle, sTarget, sPrediction=self.baseline.infer_baseline(data)
             
            # predicion : target
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"prediction:target",sPrediction,sTarget))
            # content : target 
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"content:target",sContent,sTarget))
            # predicion: content
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"prediction:content",sPrediction,sContent))

        return eval_dict_list
    
            
    def compute_losses_detection(self,checkpointpath,eval_tag,fixed_mic_dist=None):

        evalsavedir=self.config["evalsavedir"]
        N_datapoints=self.config["N_datapoints"]
        device=self.config["device"]
        train_results=torch.load(os.path.join(checkpointpath),map_location=device)
        self.model.load_state_dict(train_results["model_state_dict"])

        eval_dict_list=[]
        for j in range(0,N_datapoints):
            # get signals: 
            print(j)
            signals, rirs =self.testset_orig.get_item_test(j, gen_rir_b=True, fixed_mic_dist=fixed_mic_dist)
            with torch.no_grad():
                sPrediction=self.model(signals["s1r1"].unsqueeze(0),signals["s1r2"].unsqueeze(0))
                if bool(self.config_train["is_vae"]):      
                    sPrediction, mu, log_var = sPrediction
            # same source, RIRs from two different rooms
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1:room2",signals["s1r1"],signals["s1r2"]))
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1:room2 (early)",signals["s1r1_early"],signals["s1r2_early"]))
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1:room2 (late)",signals["s1r1_late"],signals["s1r2_late"]))
            # same source, RIRs from the same room but different position 
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1a:room1b",signals["s1r1"],signals["s1r1b"]))
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1a:room1b (early)",signals["s1r1_early"],signals["s1r1b_early"]))
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1a:room1b (late)",signals["s1r1_late"],signals["s1r1b_late"]))
            # predicion : target
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"prediction:target",sPrediction,signals["s1r2"]))
            # predicion: content
            eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"prediction:content",sPrediction,signals["s1r1"]))

            if j<10:
                if not os.path.exists(pjoin(evalsavedir,'eval_extras')):
                    os.mkdir(pjoin(evalsavedir,'eval_extras'))
                torch.save(rirs, pjoin(evalsavedir,'eval_extras',"rirs_testsample"+ str(j)+ ".pt"))

        return eval_dict_list
        
    def compute_losses_batch(self, idx, label, comp_name, x1, x2):

        device= self.config["device"]

        if len(x1.shape)<3:
            x1=x1.unsqueeze(0)
        if len(x2.shape)<3:
            x2=x2.unsqueeze(0)
        x1_emb=self.model.conditioning_network(x1)
        x2_emb=self.model.conditioning_network(x2)

        x1=x1.squeeze(1)
        x2=x2.squeeze(1)
        # Resample - losses operate on 16kHz sampling rate
        x1=hlp.torch_resample_if_needed(x1,48000,16000).to(device)
        x2=hlp.torch_resample_if_needed(x2,48000,16000).to(device)

        # ----- Compute perceptual losses -----
        L_pesq=torch.tensor([float('nan')])
        try:
            L_pesq=self.measures["pesq"](x1,x2)
        except: 
            self.failcount+=1
            print("Could not compute pesq for this signal for " + str(self.failcount) + "times")

        L_stoi=torch.tensor([float('nan')])
        try:
            L_stoi=self.measures["stoi"](x1,x2)
        except: 
            self.failcount+=1
            print("Could not compute stoi for this signal for " + str(self.failcount) + "times")

        # ----- Create a dictionary with loss values -----
        dict_row={ 
        'label': label,
        'idx':idx, 
        'compared': comp_name,
        'L_multi-stft-sc': self.measures["multi-stft"](x1,x2)[0].item(),
        'L_multi-stft-mag': self.measures["multi-stft"](x1,x2)[1].item(), 
        'L_multi-mel': self.measures["multi-mel"](x1,x2).item(),
        'L_multi-wave': self.measures["multi-wave"](x1,x2).item(),
        'L_stft-sc': self.measures["stft"](x1,x2)[0].item(),
        'L_stft-mag': self.measures["stft"](x1,x2)[1].item(),
        'L_logmel': self.measures["logmel"](x1,x2).item(),
        'L_wave': self.measures["wave"](x1,x2).item(),
        'L_sisdr': self.measures["sisdr"](x1,x2).item(),
        'L_srmr_1': self.measures["srmr"](x1).item(),
        'L_srmr_2': self.measures["srmr"](x2).item(),
        'L_pesq': L_pesq.item(), 
        'L_stoi': L_stoi.item(),  
        'L_emb_cos': (1-((torch.mean(self.measures["emb_cos"](x1_emb,x2_emb))+ 1) / 2)).item(), 
        'L_emb_euc': torch.dist(x1_emb,x2_emb).item()
        }
        
        return dict_row
        
    
def eval_condition(config,exp_subdir,checkpoint_name):
        
    eval_tag=pjoin(exp_subdir).split('/')[-1]

    # load results from checkpoints in the directory
    for filename in os.listdir(exp_subdir):
        if filename==checkpoint_name: 

            # specify training params file
            config_train=hlp.load_config(pjoin(exp_subdir,"train_config.yaml"))

            # create evaluator object
            tmp_eval = Evaluator(config,config_train)
        
            # checkpoint file path
            checkpointpath=pjoin(exp_subdir,filename)

            # compute losses for a test dataset
            if config["evalscript"]=="basic":
                eval_dict_list_condition=tmp_eval.compute_losses(checkpointpath,eval_tag)
            elif config["evalscript"]=="detection":
                eval_dict_list_condition=tmp_eval.compute_losses_detection(checkpointpath,eval_tag,fixed_mic_dist=config["fixed_mic_dist"])

            return eval_dict_list_condition
        
def eval_baseline(config):
        
    eval_tag=config["baseline"]
    config_train=config
    # create evaluator object
    tmp_eval = Evaluator(config,config_train)
    eval_dict_list=tmp_eval.compute_losses_baseline(eval_tag)
    return eval_dict_list


def eval_experiment(config):

    eval_dir=config["eval_dir"]
    eval_file_name=config["eval_file_name"]

    eval_dict_list_exp=[]
    for subdir in os.listdir(eval_dir):
        subdir_path = os.path.join(eval_dir, subdir)
        if os.path.isdir(subdir_path) & (subdir!="eval_extras"):

            print(f"Processing trainig results: {subdir_path}")
            eval_dict_list_exp.extend(eval_condition(config,subdir_path,"checkpointbest.pt"))
            print(f"Updated results to contain measures for {subdir_path}")

            pd.DataFrame(eval_dict_list_exp).to_csv(eval_dir+eval_file_name, index=False)
            print(f"Saved final results")


if __name__ == "__main__":

    config=hlp.load_config(pjoin("/home/ubuntu/joanna/reverb-match-cond-u-net/config/basic.yaml"))

    # Compute for a baseline 
    config["eval_dir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/"
    config["eval_file_name"] = "eval_wpe+fins.csv"
    config["rt60diffmin"] = -3
    config["rt60diffmax"] = 3
    config["N_datapoints"] = 0 # if 0 - whole test set included
    config["baseline"]= "wpe+fins"
    eval_dict=eval_baseline(config)
    pd.DataFrame(eval_dict).to_csv(pjoin(config["eval_dir"])+config["eval_file_name"], index=False)
    print(f"Saved condition results")



    # # Compute for all examples
    # config["eval_dir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-09-05-2024/"
    # config["evalsavedir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-09-05-2024/"
    # config["eval_file_name"] = "eval_detection_fixedmicpos.csv"
    # config["rt60diffmin"] = -3
    # config["rt60diffmax"] = 3
    # config["N_datapoints"] = 1000 # if 0 - whole test set included
    # config["evalscript"]="detection"
    # config["fixed_mic_dist"]=0.1
    # eval_dict=eval_condition(config,pjoin(config["eval_dir"], "14-05-2024--23-12_c_wunet_stft_1"),"checkpointbest.pt")
    # pd.DataFrame(eval_dict).to_csv(config["eval_dir"]+config["eval_file_name"], index=False)
    # print(f"Saved detection analysis")

    # # Compute for all examples
    # config["eval_dir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-09-05-2024/"
    # config["evalsavedir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-09-05-2024/"
    # config["eval_file_name"] = "eval_detection_randmicpos.csv"
    # config["rt60diffmin"] = -3
    # config["rt60diffmax"] = 3
    # config["N_datapoints"] = 1000 # if 0 - whole test set included
    # config["evalscript"]="detection"
    # config["fixed_mic_dist"]=None
    # eval_dict=eval_condition(config,pjoin(config["eval_dir"], "14-05-2024--23-12_c_wunet_stft_1"),"checkpointbest.pt")
    # pd.DataFrame(eval_dict).to_csv(config["eval_dir"]+config["eval_file_name"], index=False)
    # print(f"Saved detection analysis")







    
            

            

           


   








