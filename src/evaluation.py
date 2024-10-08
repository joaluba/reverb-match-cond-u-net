import torch
from tqdm import tqdm
from datetime import datetime
import time
import sys
import os
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio, SpeechReverberationModulationEnergyRatio, PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
from pysepm import fwSNRseg, bsd, wss
import numpy as np
import pandas as pd
import soundfile as sf
import random
sys.path.insert(0,'/home/ubuntu/joanna/reverb-match-cond-u-net/urgent2024_challenge/evaluation_metrics')
from calculate_intrusive_se_metrics import lsd_metric, mcd_metric
# my modules
import dataset
import baselines
import loss_mel, loss_stft, loss_waveform, loss_embedd
import trainer
import helpers as hlp
from torch.utils.data import Subset
from torchaudio import save as audiosave

class Evaluator(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.baselines=baselines.Baselines(config)
        self.load_metrics()
        self.load_eval_dataset()
        self.failcount=0

    def load_metrics(self):
        device=self.config["device"]

        my_best_checkpoint_path="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/10-06-2024--15-02_c_wunet_stft+wave_0.8_0.2/checkpointbest.pt"

        self.measures= {
            "multi-stft" : loss_stft.MultiResolutionSTFTLoss().to(device), 
            "multi-mel"  : loss_mel.MultiMelSpectrogramLoss().to(device),
            "multi-wave" : loss_waveform.MultiWindowShapeLoss().to(device),
            "stft" : loss_stft.STFTLoss().to(device), 
            "logmel"  : loss_mel.LogMelSpectrogramLoss().to(device),
            "wave" : loss_waveform.WaveformShapeLoss(winlen=960).to(device),
            "mse" : torch.nn.MSELoss().to(device),
            "sisdr" : ScaleInvariantSignalDistortionRatio().to(device),
            "sdr" : SignalDistortionRatio().to(device),
            "srmr" : SpeechReverberationModulationEnergyRatio(16000).to(device),
            "pesq" : PerceptualEvaluationSpeechQuality(16000, 'wb').to(device),
            "stoi" : ShortTimeObjectiveIntelligibility(16000).to(device),
            "emb_cos": loss_embedd.EmbeddingLossCosine(my_best_checkpoint_path,device="cuda"),
            "emb_euc": loss_embedd.EmbeddingLossEuclidean(my_best_checkpoint_path,device="cuda"),
            "squim_obj" : SQUIM_OBJECTIVE.get_model().to(device),
            "squim_subj" : SQUIM_SUBJECTIVE.get_model().to(device)
        }

    def load_eval_dataset(self):

        rt60diffmin = self.config["rt60diffmin"]
        rt60diffmax = self.config["rt60diffmax"]
        N_datapoints = self.config["N_datapoints"]
        batch_size_eval = self.config["batch_size_eval"]

        # load a test split from the dataset used for training
        self.config["split"] = "test" 
        self.config["p_noise"] = 0 # for evaluation, we do not want noise 
        self.testset_orig = dataset.DatasetReverbTransfer(self.config)
        # choose a subset of the original test split
        indices_chosen = self.testset_orig.get_idx_with_rt60diff(rt60diffmin,rt60diffmax)
        if N_datapoints>0:
            indices_chosen = range(0,N_datapoints)
        self.testset = Subset(self.testset_orig,indices_chosen)
        print(f"Preparing to evaluate {len(self.testset)} test datapoints")
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size_eval, shuffle=False, num_workers=6,pin_memory=True)


    def compute_metrics_oracle(self):
                
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        # get speech reference for non-intrusive metrics:
        speechref = hlp.torch_load_mono("/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/speech_VCTK_4_sentences.wav",48000)[:,:4*48000].unsqueeze(1)
        
        eval_dict_list=[]
        for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):
            # get datapoint 
            sContent, sStyle, sTarget, sAnecho, sStyle_anecho = data
            sTargetClone=self.testset_orig.get_target_clone(j,sAnecho)
            # get metrics
            # -> content : target 
            eval_dict_list.append(self.metrics4batch(j,"oracle","content:target",sContent,sTarget,sAnecho,nmref=speechref))
            # -> content : anechoic
            eval_dict_list.append(self.metrics4batch(j,"oracle","content:anecho",sContent,sAnecho,sAnecho, nmref=speechref))
            # -> content : style
            eval_dict_list.append(self.metrics4batch(j,"oracle","content:style",sContent,sStyle,sAnecho, nmref=speechref))
            # -> targetclone : target
            eval_dict_list.append(self.metrics4batch(j,"oracle","targetclone:target",sTargetClone,sTarget,sAnecho, nmref=speechref))
            # # -> r(content) : r(target) 
            # eval_dict_list.append(self.metrics4batch(j,"oracle","r(content):r(target)",sContent-sAnecho,sTarget-sAnecho, sAnecho, nmref=speechref))

        return eval_dict_list
    
    def compute_metrics_checkpoint(self,checkpointpath,eval_tag):
                
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        # get speech reference for non-intrusive metrics:
        speechref = hlp.torch_load_mono("/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/speech_VCTK_4_sentences.wav",48000)[:,:4*48000].unsqueeze(1)

        # load training configuration
        device=self.config["device"]
        config_train=hlp.load_config(pjoin(os.path.dirname(checkpointpath),"train_config.yaml"))
        # load model architecture
        model=trainer.load_chosen_model(config_train,config_train["modeltype"]).to(device)
        # load weights from checkpoint
        train_results=torch.load(os.path.join(checkpointpath),map_location=device, weights_only=True)
        model.load_state_dict(train_results["model_state_dict"])

        eval_dict_list=[]
        for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):
            # get datapoint 
            sContent, sStyle, sTarget, sAnecho, sStyle_anecho = data
            # get prediction
            _, _, _, sPrediction=trainer.infer(model, data, device)
            if bool(config_train["is_vae"]):      
                sPrediction, mu, log_var = sPrediction
            # get metrics
            # -> predicion : target
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"prediction:target", sPrediction, sTarget, sAnecho, nmref=speechref))
            # -> predicion : content
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"prediction:content",sPrediction, sContent, sAnecho,nmref=speechref))
            # # -> predicion : style
            # eval_dict_list.append(self.metrics4batch(j,eval_tag,"r(prediction):r(style)",sPrediction,sStyle,sAnecho,nmref=speechref))
            # # -> reverb(predicion) : reverb(target)
            # eval_dict_list.append(self.metrics4batch(j,eval_tag,"r(prediction):r(target)",sPrediction-sAnecho,sTarget-sAnecho,sAnecho,nmref=speechref))
            # # -> reverb(predicion) : reverb(content)
            # eval_dict_list.append(self.metrics4batch(j,eval_tag,"r(prediction):r(content)",sPrediction-sAnecho,sContent-sAnecho,sAnecho,nmref=speechref))

        return eval_dict_list
    
    def compute_metrics_baselines(self,baseline):
        
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        # get speech reference for non-intrusive metrics:
        speechref = hlp.torch_load_mono("/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/speech_VCTK_4_sentences.wav",48000)[:,:4*48000].unsqueeze(1)

        eval_dict_list=[]
        for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):
            # get datapoint 
            sContent, sStyle, sTarget, sAnecho, sStyle_anecho = data
            # get prediction
            _, _, _, sPrediction=self.baselines.infer_baseline(data,baseline)
            # get metrics
            # -> predicion : target
            eval_dict_list.append(self.metrics4batch(j,baseline,"prediction:target",sPrediction,sTarget,sAnecho,nmref=speechref))
            # -> predicion : content
            eval_dict_list.append(self.metrics4batch(j,baseline,"prediction:content",sPrediction,sContent,sAnecho,nmref=speechref))
            # # -> predicion : style
            # eval_dict_list.append(self.metrics4batch(j,baseline,"prediction:style",sPrediction,sStyle,sAnecho,nmref=speechref))
            # # -> r(predicion) : r(target)
            # eval_dict_list.append(self.metrics4batch(j,baseline,"r(prediction):r(target)",sPrediction-sAnecho,sTarget-sAnecho,sAnecho,nmref=speechref))
            # # -> r(predicion) : r(content)
            # eval_dict_list.append(self.metrics4batch(j,baseline,"r(prediction):r(content)",sPrediction-sAnecho,sContent-sAnecho,sAnecho,nmref=speechref))

        return eval_dict_list
    

    def metrics4batch(self, idx, label, comp_name, x1, x2, x_anecho, nmref):

        device= self.config["device"]
        # prepare dimensions (B,C,N) -> (B,N)
        x1=x1.squeeze(1) 
        x2=x2.squeeze(1)
        x_anecho=x_anecho.squeeze(1)
        nmref=nmref.squeeze(1)
        # Resample - losses operate on 16kHz sampling rate
        x1=hlp.torch_resample_if_needed(x1,48000,16000).to(device)
        x2=hlp.torch_resample_if_needed(x2,48000,16000).to(device)
        x_anecho=hlp.torch_resample_if_needed(x_anecho,48000,16000).to(device)
        nmref=hlp.torch_resample_if_needed(nmref,48000,16000).to(device)

        # ----- Compute perceptual losses -----
        L_pesq_x1=torch.tensor([float('nan')])
        L_pesq_x2=torch.tensor([float('nan')])
        try:
            L_pesq_x1=self.measures["pesq"](x1,x_anecho)
            L_pesq_x2=self.measures["pesq"](x2,x_anecho)
            L_pesq_ab=self.measures["pesq"](x1,x2)
            L_pesq_ba=self.measures["pesq"](x2,x1)
        except: 
            self.failcount+=1
            print("Could not compute pesq for this signal for " + str(self.failcount) + "times")

        L_stoi_x1=torch.tensor([float('nan')])
        L_stoi_x2=torch.tensor([float('nan')])
        L_stoi_ab=torch.tensor([float('nan')])
        L_stoi_ba=torch.tensor([float('nan')])
        try:
            L_stoi_x1=self.measures["stoi"](x1,x_anecho)
            L_stoi_x2=self.measures["stoi"](x2,x_anecho)
            L_stoi_ab=self.measures["stoi"](x1,x2)
            L_stoi_ba=self.measures["stoi"](x2,x1)
        except: 
            self.failcount+=1
            print("Could not compute stoi for this signal for " + str(self.failcount) + "times")

        L_sisdr_x1=torch.tensor([float('nan')])
        L_sisdr_x2=torch.tensor([float('nan')])
        L_sisdr_x1=self.measures["sisdr"](x1,x_anecho)
        L_sisdr_x2=self.measures["sisdr"](x2,x_anecho)

        # ----- Compute non-intrusive metrics -----
        try:
            L_pesqni_x1, L_stoini_x1, L_sisdrni_x1=self.measures["squim_obj"](x1)
            L_pesqni_x2, L_stoini_x2, L_sisdrni_x2=self.measures["squim_obj"](x2)
        except:
            print("Could not compute squim objective predictions")

        L_mos_x1=torch.tensor([float('nan')])
        L_mos_x2=torch.tensor([float('nan')])
        try:
            L_mos_x1=self.measures["squim_subj"](x1,nmref) 
            L_mos_x2=self.measures["squim_subj"](x2,nmref) 
        except:
            print("Could not compute squim subjective predictions")

        # L_srmr_x1=self.measures["srmr"](x1)
        # L_srmr_x2=self.measures["srmr"](x2)

        # ----- Compute metrics from urgent -----

        L_mcd_ab=mcd_metric(x1.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(), 16000, eps=1.0e-08)
        L_mcd_ba=mcd_metric(x2.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(), 16000, eps=1.0e-08)

        L_lsd_ab=lsd_metric(x1.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(), 16000, nfft=0.032, hop=0.016, p=2, eps=1.0e-08)
        L_lsd_ba=lsd_metric(x2.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(), 16000, nfft=0.032, hop=0.016, p=2, eps=1.0e-08)

        # ----- Compute metrics from spear -----
        L_fwsnr_ab=fwSNRseg(x1.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(),16000)
        L_fwsnr_ba=fwSNRseg(x2.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(),16000)

        L_multi_stft_ab=self.measures["multi-stft"](x1,x2)[0].item() + self.measures["multi-stft"](x1,x2)[1].item()
        L_multi_stft_ba=self.measures["multi-stft"](x2,x1)[0].item() + self.measures["multi-stft"](x2,x1)[1].item()

        L_stft_ab=self.measures["stft"](x1,x2)[0].item() + self.measures["stft"](x1,x2)[1].item()
        L_stft_ba=self.measures["stft"](x2,x1)[0].item() + self.measures["stft"](x2,x1)[1].item()

        # ----- Create a dictionary with loss values -----
        dict_row={ 
        'label': label,
        'idx':idx, 
        'compared': comp_name,
        # Type 1: similarity measured using symmetric metric
        # M = m(a,b) = m(b,a)
        '1L_multi-stft-mag': self.measures["multi-stft"](x1,x2)[1].item(), 
        '1L_stft-mag': self.measures["stft"](x1,x2)[1].item(),
        '1L_multi-wave': self.measures["multi-wave"](x1,x2).item(),
        '1L_wave': self.measures["wave"](x1,x2).item(),
        '1L_logmel': self.measures["logmel"](x1,x2).item(),
        '1L_multi-mel': self.measures["multi-mel"](x1,x2).item(),
        '1S_sisdr': self.measures["sisdr"](x1,x2).item(),
        '1L_emb_euc': self.measures["emb_euc"](x1,x2).item(),
        # 'D_emb_cos': self.measures["emb_cos"](x1,x2).item(), 
        
        # Type 2: similarity measured using non-symmetric metric 
        # M = (m(a,b)+m(b,a))/2
        '2L_lsd' : (L_lsd_ab+L_lsd_ba)/2, 
        '2L_mcd' : (L_mcd_ab+L_mcd_ba)/2, 
        '2S_fwsnr': (L_fwsnr_ab + L_fwsnr_ba)/2,
        '2L_multi-stft': ((L_multi_stft_ab+L_multi_stft_ba)/2),
        '2L_stft': ((L_stft_ab+L_stft_ba)/2),
        '2S_pesq': ((L_pesq_ab+L_pesq_ba)/2).item(),
        '2S_stoi': ((L_stoi_ab+L_stoi_ba)/2).item(),

        # Type 3: similarity measured as distance in intrusive metric
        # M = abs(m(a,ref) - m(b,ref))
        '3D_pesq': torch.mean(torch.abs(L_pesq_x1-L_pesq_x2)).item(), 
        '3D_stoi': torch.mean(torch.abs(L_stoi_x1-L_stoi_x2)).item(),
        '3D_sisdr': torch.mean(torch.abs(L_sisdr_x1-L_sisdr_x2)).item(),
        '3D_mos_nidiff': torch.mean(torch.abs(L_mos_x1-L_mos_x2)).item(),
        '3D_pesq_nidiff': torch.mean(torch.abs(L_pesqni_x1-L_pesqni_x2)).item(),
        '3D_stoi_nidiff': torch.mean(torch.abs(L_stoini_x1-L_stoini_x2)).item(), 
        '3D_sisdr_nidiff': torch.mean(torch.abs(L_sisdrni_x1-L_sisdrni_x2)).item(),  

        }
        
        return dict_row
        
    def save_audios_sample(self,checkpointpath,idx,savedir,savefiles=True):
                        
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        filenames=[]
        sigs=[]

        fs=self.testset_orig.fs

        if savefiles==True:
            data=self.testset_orig[idx]
            sContent, sStyle, sTarget, sAnecho, sStyle_anecho = data

        if checkpointpath=="groundtruth":
            
            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + '--sAnecho.wav'))
            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + '--sTarget.wav'))
            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + '--sTargetClone.wav'))
            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + '--sContent.wav'))
            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + '--sStyle.wav'))

            if savefiles==True:
                print("getting ground truth signals")
                sTargetClone=self.testset_orig.get_target_clone(idx,sAnecho)
            else:
                print("loading ground truth signals from file")
                sAnecho=hlp.torch_load_mono(filenames[0],fs)
                sTarget=hlp.torch_load_mono(filenames[1],fs)
                sTargetClone=hlp.torch_load_mono(filenames[2],fs)
                sContent=hlp.torch_load_mono(filenames[3],fs)
                sStyle=hlp.torch_load_mono(filenames[4],fs)
                
            sigs.append(sAnecho.cpu())
            sigs.append(sTarget.cpu())
            sigs.append(sTargetClone.cpu())
            sigs.append(sContent.cpu())
            sigs.append(sStyle.cpu())

        elif checkpointpath=="baselines":

            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + '--sPred_anecho_fins.wav'))
            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + '--sPred_dfnet_fins.wav'))
            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + '--sPred_wpe_fins.wav'))

            if savefiles==True:
                print("getting baseline signals")
                sPred_anecho_fins=self.baselines.infer_baseline(data,"anecho+fins")[3].cpu().squeeze(1)
                sPred_dfnet_fins=self.baselines.infer_baseline(data,"dfnet+fins")[3].cpu().squeeze(1)
                sPred_wpe_fins=self.baselines.infer_baseline(data,"wpe+fins")[3].cpu().squeeze(1)
            else: 
                print("loading baseline signals from file")
                sPred_anecho_fins=hlp.torch_load_mono(filenames[0],fs)
                sPred_dfnet_fins=hlp.torch_load_mono(filenames[1],fs)
                sPred_wpe_fins=hlp.torch_load_mono(filenames[2],fs)

            sigs.append(sPred_anecho_fins)
            sigs.append(sPred_dfnet_fins)
            sigs.append(sPred_wpe_fins)

        else: 
            model_tag=checkpointpath.split('/')[-2] + "_" + os.path.basename(checkpointpath).split(".")[0]
            filenames.append(pjoin(savedir,"testset_idx" + str(idx) + "--" + model_tag + '_sPred_model.wav'))

            if savefiles==True:
                print("getting checkpoint signals")
                # load training configuration
                device=self.config["device"]
                config_train=hlp.load_config(pjoin(os.path.dirname(checkpointpath),"train_config.yaml"))
                # load model architecture
                model=trainer.load_chosen_model(config_train,config_train["modeltype"]).to(device)
                # load weights from checkpoint
                train_results=torch.load(os.path.join(checkpointpath),map_location=device,weights_only=True)
                model.load_state_dict(train_results["model_state_dict"])
                sPred_model=trainer.infer(model, data, device)[3].cpu().squeeze(1)
                if bool(config_train["is_vae"]):      
                    sPred_model, mu, log_var = sPred_model
            else:
                print("loading checkpoint signals from file")
                sPred_model=hlp.torch_load_mono(filenames[0],fs)
            
            sigs.append(sPred_model)

        if savefiles==True:
            [audiosave(filenames[i], sigs[i], 48000) for i,_ in enumerate(sigs)]

        return  sigs, filenames


def eval_experiment(config,checkpoints_list=None):

    eval_dict_list=[]
    eval_dir=config["eval_dir"]
    eval_file_name=config["eval_file_name"]

    # ---- initialize evaluator ----
    # (metrics, baselines, dataset, dataloader)
    myeval = Evaluator(config)

    # ---- evaluate oracle data  -----
    print(f"Computing metrics -> oracle ")
    eval_dict_list.extend(myeval.compute_metrics_oracle())
    # list -> df and save 
    pd.DataFrame(eval_dict_list).to_csv(eval_dir+eval_file_name, index=False)

    
    # --- evaluate baselines ----
    print(f"Computing metrics -> baselines ")
    eval_dict_list.extend(myeval.compute_metrics_baselines("anecho+fins"))
    eval_dict_list.extend(myeval.compute_metrics_baselines("dfnet+fins"))
    eval_dict_list.extend(myeval.compute_metrics_baselines("wpe+fins"))
    # list -> df and save 
    pd.DataFrame(eval_dict_list).to_csv(eval_dir+eval_file_name, index=False)


    # ---- evaluate my trained models ----
    print(f"Computing metrics -> models ")
    # a loop over all conditions of the considered experiment 
    # each condition contains a checkpoint for a different model version
    if checkpoints_list==None:
  
        for subdir in os.listdir(eval_dir):
            exp_subdir = os.path.join(eval_dir, subdir)
            if os.path.isdir(exp_subdir):
                print(f"...using training results {exp_subdir}")
                # checkpoint file path
                tmp_checkpointpath=pjoin(exp_subdir,"checkpointbest.pt")
                # name of this experiment condition
                tmp_eval_tag=pjoin(exp_subdir).split('/')[-1]
                # compute metrics for this model version
                tmp_dict_eval=myeval.compute_metrics_checkpoint(tmp_checkpointpath,tmp_eval_tag)
                # add the metrics to the main list containing all eval results
                eval_dict_list.extend(tmp_dict_eval)
                # list -> df and save 
                pd.DataFrame(eval_dict_list).to_csv(eval_dir+eval_file_name, index=False)
    else: 
        for checkpoint in checkpoints_list:
            print(f"...using training results {checkpoint}")
            # name of this experiment condition
            tmp_eval_tag=checkpoint.split('/')[-2] + "_" + os.path.basename(checkpoint).split(".")[0]
            # compute metrics for this model version
            tmp_dict_eval=myeval.compute_metrics_checkpoint(checkpoint,tmp_eval_tag)
            # add the metrics to the main list containing all eval results
            eval_dict_list.extend(tmp_dict_eval)
            # list -> df and save 
            pd.DataFrame(eval_dict_list).to_csv(eval_dir+eval_file_name, index=False)


if __name__ == "__main__":

    # make a list of checkpoints to compare (in addition to ground truth sounds and baselines)
    checkpoint_paths=[
                    "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/10-06-2024--15-02_c_wunet_stft+wave_0.8_0.2/checkpointbest.pt",
                    "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/20-05-2024--22-48_c_wunet_logmel+wave_0.8_0.2/checkpointbest.pt",
                    "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/29-05-2024--05-47_c_wunet_logmel_1/checkpointbest.pt",
                    "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/18-06-2024--18-37_c_wunet_stft_1/checkpointbest.pt",
                    "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/18-06-2024--18-37_c_wunet_stft_1/checkpoint50.pt",
                    "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/18-06-2024--18-37_c_wunet_stft_1/checkpoint10.pt",
                    "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/18-06-2024--18-37_c_wunet_stft_1/checkpoint0.pt"
                    ]

    # load default config
    config=hlp.load_config(pjoin("/home/ubuntu/joanna/reverb-match-cond-u-net/config/basic.yaml"))

    # set parameters for this experiment
    config["eval_dir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/"
    config["eval_file_name"] = "100924_compare_percept.csv"
    config["rt60diffmin"] = -3
    config["rt60diffmax"] = 3
    config["N_datapoints"] = 5 # if 0 - whole test set included
    config["batch_size_eval"] = 1

    eval_experiment(config,checkpoints_list=checkpoint_paths)