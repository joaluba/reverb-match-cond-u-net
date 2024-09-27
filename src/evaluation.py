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
        speechref = hlp.torch_load_mono("../sounds/speech_VCTK_4_sentences.wav",self.fs)

        eval_dict_list=[]
        for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):
            # get datapoint 
            sContent, sStyle, sTarget, sAnecho, sStyle_anecho = data
            # get metrics
            # -> content : target 
            eval_dict_list.append(self.metrics4batch(j,"oracle","content:target",sContent,sTarget,sAnecho,nmref=speechref))
            # -> content : anechoic
            eval_dict_list.append(self.metrics4batch(j,"oracle","content:anecho",sContent,sAnecho,sAnecho, nmref=speechref))
            # -> content : style
            eval_dict_list.append(self.metrics4batch(j,"oracle","content:style",sContent,sStyle,sAnecho, nmref=speechref))
            # -> r(content) : r(target) 
            eval_dict_list.append(self.metrics4batch(j,"oracle","r(content):r(target)",sContent-sAnecho,sTarget-sAnecho, sAnecho, nmref=speechref))

        return eval_dict_list
    
    def compute_metrics_checkpoint(self,checkpointpath,eval_tag):
                
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        # get speech reference for non-intrusive metrics:
        speechref = hlp.torch_load_mono("../sounds/speech_VCTK_4_sentences.wav",self.fs)

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
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"prediction:target",sPrediction,sTarget,sAnecho,nmref=speechref))
            # -> predicion : content
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"prediction:content",sPrediction,sContent,sAnecho,nmref=speechref))
            # -> predicion : style
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"r(prediction):r(style)",sPrediction,sStyle,sAnecho,nmref=speechref))
            # -> reverb(predicion) : reverb(target)
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"r(prediction):r(target)",sPrediction-sAnecho,sTarget-sAnecho,sAnecho,nmref=speechref))
            # -> reverb(predicion) : reverb(content)
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"r(prediction):r(content)",sPrediction-sAnecho,sContent-sAnecho,sAnecho,nmref=speechref))

        return eval_dict_list
    
    def compute_metrics_baselines(self,baseline):
        
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        # get speech reference for non-intrusive metrics:
        speechref = hlp.torch_load_mono("../sounds/speech_VCTK_4_sentences.wav",self.fs)

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
            # -> predicion : style
            eval_dict_list.append(self.metrics4batch(j,baseline,"prediction:style",sPrediction,sStyle,sAnecho,nmref=speechref))
            # -> r(predicion) : r(target)
            eval_dict_list.append(self.metrics4batch(j,baseline,"r(prediction):r(target)",sPrediction-sAnecho,sTarget-sAnecho,sAnecho,nmref=speechref))
            # -> r(predicion) : r(content)
            eval_dict_list.append(self.metrics4batch(j,baseline,"r(prediction):r(content)",sPrediction-sAnecho,sContent-sAnecho,sAnecho,nmref=speechref))

        return eval_dict_list
    
            
    # def compute_losses_checkpoint_rocs(self,checkpointpath,eval_tag):

    #     np.random.seed(0)
    #     random.seed(0)
    #     torch.manual_seed(0)

    #     train_results=torch.load(os.path.join(checkpointpath),map_location=self.config["device"],weights_only=True))
    #     self.model.load_state_dict(train_results["model_state_dict"])
    #     N_datapoints=self.config["N_datapoints"]


    #     eval_dict_list=[]
    #     for j in range(0,N_datapoints):
    #         # get signals: 
    #         print(j)
    #         signals, rirs =self.testset_orig.get_item_test(j, gen_rir_b=True, fixed_mic_dist=fixed_mic_dist)

    #             sPrediction=self.model(signals["s1r1"].unsqueeze(0),signals["s1r2"].unsqueeze(0))
    #             if bool(self.config_train["is_vae"]):      
    #                 sPrediction, mu, log_var = sPrediction
    #         # same source, RIRs from two different rooms
    #         eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1:room2",signals["s1r1"],signals["s1r2"]))
    #         eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1:room2 (early)",signals["s1r1_early"],signals["s1r2_early"]))
    #         eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1:room2 (late)",signals["s1r1_late"],signals["s1r2_late"]))
    #         # same source, RIRs from the same room but different position 
    #         eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1a:room1b",signals["s1r1"],signals["s1r1b"]))
    #         eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1a:room1b (early)",signals["s1r1_early"],signals["s1r1b_early"]))
    #         eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"room1a:room1b (late)",signals["s1r1_late"],signals["s1r1b_late"]))
    #         # predicion : target
    #         eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"prediction:target",sPrediction,signals["s1r2"]))
    #         # predicion: content
    #         eval_dict_list.append(self.compute_losses_batch(j,eval_tag,"prediction:content",sPrediction,signals["s1r1"]))

    #     return eval_dict_list
        
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
        L_pesq_ab=torch.tensor([float('nan')])
        L_pesq_ba=torch.tensor([float('nan')])
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

        L_srmr_x1=self.measures["srmr"](x1)
        L_srmr_x2=self.measures["srmr"](x2)

        # ----- Compute metrics from urgent -----
        L_mcd_x1=mcd_metric(x_anecho.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(),16000,eps=1.0e-08)
        L_mcd_x2=mcd_metric(x_anecho.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(),16000,eps=1.0e-08)

        L_lsd_ab=lsd_metric(x1.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(), 16000, nfft=0.032, hop=0.016, p=2, eps=1.0e-08)
        L_lsd_ba=lsd_metric(x2.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(), 16000, nfft=0.032, hop=0.016, p=2, eps=1.0e-08)

        # ----- Compute metrics from spear -----
        L_fwsnr_x1=fwSNRseg(x_anecho.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(),16000)
        L_fwsnr_x2=fwSNRseg(x_anecho.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(),16000)

        L_bsd_x1=bsd(x_anecho.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(),16000)
        L_bsd_x2=bsd(x_anecho.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(),16000)
        
        L_wss_x1=wss(x_anecho.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(),16000)
        L_wss_x2=wss(x_anecho.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(),16000)



        # ----- Create a dictionary with loss values -----
        dict_row={ 
        'label': label,
        'idx':idx, 
        'compared': comp_name,
        'L_mse': self.measures["mse"](x1,x2).item(),
        'L_multi-stft-sc': self.measures["multi-stft"](x1,x2)[0].item(),
        'L_multi-stft-mag': self.measures["multi-stft"](x1,x2)[1].item(), 
        'L_multi-mel': self.measures["multi-mel"](x1,x2).item(),
        'L_multi-wave': self.measures["multi-wave"](x1,x2).item(),
        'L_stft-sc': self.measures["stft"](x1,x2)[0].item(),
        'L_stft-mag': self.measures["stft"](x1,x2)[1].item(),
        'L_logmel': self.measures["logmel"](x1,x2).item(),
        'L_wave': self.measures["wave"](x1,x2).item(),
        'S_sisdr': self.measures["sisdr"](x1,x2).item(),
        'D_srmr_nidiff': torch.mean(torch.abs(L_srmr_x1-L_srmr_x2)).item(),
        'D_pesq': torch.mean(torch.abs(L_pesq_x1-L_pesq_x2)).item(), 
        'D_stoi': torch.mean(torch.abs(L_stoi_x1-L_stoi_x2)).item(),
        'D_sisdr': torch.mean(torch.abs(L_sisdr_x1-L_sisdr_x2)).item(),
        # 'S_pesq': ((L_pesq_ab+L_pesq_ba)/2).item(),
        # 'S_stoi': ((L_stoi_ab+L_stoi_ba)/2).item(),          
        'D_mos_nidiff': torch.mean(torch.abs(L_mos_x1-L_mos_x2)).item(),
        'D_pesq_nidiff': torch.mean(torch.abs(L_pesqni_x1-L_pesqni_x2)).item(),
        'D_stoi_nidiff': torch.mean(torch.abs(L_stoini_x1-L_stoini_x2)).item(), 
        'D_sisdr_nidiff': torch.mean(torch.abs(L_sisdrni_x1-L_sisdrni_x2)).item(),  
        # 'D_emb_cos': self.measures["emb_cos"](x1,x2).item(), 
        'D_emb_euc': self.measures["emb_euc"](x1,x2).item(),
        'D_mcd': np.mean(np.abs(L_mcd_x1-L_mcd_x2)).item(),
        'D_lsd' : (L_lsd_ab+L_lsd_ba)/2, 
        'D_fwsnr': np.mean(np.abs(L_fwsnr_x1-L_fwsnr_x2)).item(),
        'D_wss': np.mean(np.abs(L_wss_x1-L_wss_x2)).item(),
        'D_bsd': np.mean(np.abs(L_bsd_x1-L_bsd_x2)).item()
        }
        
        return dict_row
        

    def compare_audios_checkpoint(self,checkpointpath,idx,allsigs=False):
                    
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        model_tag=checkpointpath.split('/')[-2] + "_" + os.path.basename(checkpointpath).split(".")[0]
        # load training configuration
        device=self.config["device"]
        config_train=hlp.load_config(pjoin(os.path.dirname(checkpointpath),"train_config.yaml"))
        # load model architecture
        model=trainer.load_chosen_model(config_train,config_train["modeltype"]).to(device)
        # load weights from checkpoint
        train_results=torch.load(os.path.join(checkpointpath),map_location=device,weights_only=True)
        model.load_state_dict(train_results["model_state_dict"])
        # get datapoint
        data=self.testset_orig[idx]
        sContent, sStyle, sTarget, sAnecho, sStyle_anecho = data
        # get prediction of the model
        _, _, _, sPred_model=trainer.infer(model, data, device)
        if bool(config_train["is_vae"]):      
            sPred_model, mu, log_var = sPred_model

        if allsigs==True:

            # get the target signal with a cloned RIR (same room, but different position)
            df_info=self.testset_orig.get_info(idx,id="style")
            original_rir_path=df_info["ir_file_path"]
            dir_name = os.path.dirname(original_rir_path)
            file_name = os.path.basename(original_rir_path)
            clone_file_name = "clone_" + file_name
            # cloned impulse response
            rir_clone = hlp.torch_load_mono(pjoin(dir_name,clone_file_name),self.testset_orig.fs)
            import scipy.signal as signal
            sTargetClone = torch.from_numpy(signal.fftconvolve(sAnecho.numpy(), rir_clone,mode="full"))[:,:self.testset_orig.sig_len]
            # Synchronize to anechoic signal
            _,sTargetClone,_ = hlp.synch_sig2(sAnecho,sTargetClone)
            sTargetClone=hlp.torch_normalize_max_abs(sTargetClone)

            # get prediction of the baselines
            _,_,_,sPred_anecho_fins=self.baselines.infer_baseline(data,"anecho+fins")
            _,_,_,sPred_dfnet_fins=self.baselines.infer_baseline(data,"dfnet+fins")
            _,_,_,sPred_wpe_fins=self.baselines.infer_baseline(data,"wpe+fins")
            
        
            audios = {  "model_tag": model_tag,
                        "testset_idx": idx,
                        "sContent" : sContent,
                        "sTarget" : sTarget,
                        "sTargetClone" : sTargetClone,
                        "sAnecho" : sAnecho,
                        "sPred_model" : sPred_model.squeeze(0).cpu(),
                        "sPred_anecho_fins" : sPred_anecho_fins.squeeze(0).cpu(),
                        "sPred_dfnet_fins" : sPred_dfnet_fins.squeeze(0).cpu(),
                        "sPred_wpe_fins" : sPred_wpe_fins.squeeze(0).cpu()
                        }
        
        else: 
            audios = {"model_tag": model_tag,
                        "testset_idx": idx,
                        "sContent" : sContent,
                        "sTarget" : sTarget,
                        "sAnecho" : sAnecho,
                        "sPred_model" : sPred_model.squeeze(0).cpu()
                        }
        return audios


def save_compare_audios(audios,savedir,allsigs=False):
    filenames=[]
    if allsigs==True:

        filenames.append("testset_idx" + str(audios["testset_idx"]) + '--sContent.wav')#0
        filenames.append("testset_idx" + str(audios["testset_idx"]) + '--sAnecho.wav')#1
        filenames.append("testset_idx" + str(audios["testset_idx"]) + '--sTarget.wav')#2
        filenames.append("testset_idx" + str(audios["testset_idx"]) + '--sTargetClone.wav')#3
        filenames.append("testset_idx" + str(audios["testset_idx"]) + '--sPred_anecho_fins.wav')#4
        filenames.append("testset_idx" + str(audios["testset_idx"]) + '--sPred_dfnet_fins.wav')#5
        filenames.append("testset_idx" + str(audios["testset_idx"]) + '--sPred_wpe_fins.wav')#6
        filenames.append("testset_idx" + str(audios["testset_idx"]) + "--" + audios["model_tag"] + '_sPred_model.wav')#7

        print(f"Saving ground truth signals:\n{filenames[0]}\n{filenames[1]}\n{filenames[2]}\n{filenames[3]}\n")
        audiosave(pjoin(savedir,filenames[0]), audios["sContent"].cpu(), 48000)
        audiosave(pjoin(savedir,filenames[1]), audios["sAnecho"].cpu(), 48000)
        audiosave(pjoin(savedir,filenames[2]), audios["sTarget"].cpu(), 48000)
        audiosave(pjoin(savedir,filenames[3]), audios["sTargetClone"].cpu(), 48000)

        print(f"Saving baselines predictions:\n{filenames[4]}\n{filenames[5]}\n{filenames[6]}\n")
        audiosave(pjoin(savedir,filenames[4]), audios["sPred_anecho_fins"].cpu(), 48000)
        audiosave(pjoin(savedir,filenames[5]), audios["sPred_dfnet_fins"].cpu(), 48000)
        audiosave(pjoin(savedir,filenames[6]), audios["sPred_wpe_fins"].cpu(), 48000)

        print(f"Saving prediction of our model:\n{filenames[7]}\n")
        audiosave(pjoin(savedir,filenames[7]), audios["sPred_model"].cpu(), 48000)
        
    else:
        filenames.append("testset_idx" + str(audios["testset_idx"]) + "--" + audios["model_tag"] + '_sPred_model.wav')#0
        print(f"Saving prediction of our model:\n{filenames[0]}\n")
        audiosave(pjoin(savedir,filenames[0]), audios["sPred_model"].cpu(), 48000)
    
    return filenames


# def save_batch_audios(sContent,sStyle,sTarget,sPrediction,sAnecho,eval_tag,batch_nr,dirsounds):
#     batch_size=sContent.shape[0]
#     [audiosave(pjoin(dirsounds,eval_tag + "_b"+ str(batch_nr)+ "_i"+ str(i) +'_content.wav'), sContent[i,:,:].cpu(), 48000) for i in range(batch_size)]
#     [audiosave(pjoin(dirsounds,eval_tag + "_b"+ str(batch_nr)+ "_i"+ str(i) +'_style.wav'), sStyle[i,:,:].cpu(), 48000) for i in range(batch_size)]
#     [audiosave(pjoin(dirsounds,eval_tag + "_b"+ str(batch_nr)+ "_i"+ str(i) +'_target.wav'), sTarget[i,:,:].cpu(), 48000) for i in range(batch_size)]
#     [audiosave(pjoin(dirsounds,eval_tag + "_b"+ str(batch_nr)+ "_i"+ str(i) +'_prediction.wav'), sPrediction[i,:,:].cpu(), 48000) for i in range(batch_size)]
#     [audiosave(pjoin(dirsounds,eval_tag + "_b"+ str(batch_nr)+ "_i"+ str(i) +'_anecho.wav'), sAnecho[i,:,:].cpu(), 48000) for i in range(batch_size)]


def eval_experiment(config):

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
    for subdir in os.listdir(eval_dir):
        exp_subdir = os.path.join(eval_dir, subdir)
        if os.path.isdir(exp_subdir):
            print(f"...using training results {exp_subdir}")
            # specify training params file
            tmp_config_train=hlp.load_config(pjoin(exp_subdir,"train_config.yaml"))
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


if __name__ == "__main__":

    config=hlp.load_config(pjoin("/home/ubuntu/joanna/reverb-match-cond-u-net/config/basic.yaml"))
    
    myeval = Evaluator(config)

    my_best_checkpoint_path="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/10-06-2024--15-02_c_wunet_stft+wave_0.8_0.2/checkpointbest.pt"

    audios=myeval.compare_audios_checkpoint(my_best_checkpoint_path,0)


    # config["eval_dir"] = "/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/"
    # config["N_datapoints"] = 10 # if 0 - whole test set included

    # config["eval_file_name"] = "eval_dereverb.csv"
    # config["rt60diffmin"] = -2
    # config["rt60diffmax"] = -0.2
    # eval_experiment(config)

    # # Compute for difficult de-reverberation
    # config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/rereverb/"
    # config["eval_file_name"] = "eval_rereverb.csv"
    # config["rt60diffmin"] = 0.2
    # config["rt60diffmax"] = 2
    # eval_experiment(config)

    # # Compute for all examples
    # config["savedir_sounds"]="/home/ubuntu/joanna/reverb-match-cond-u-net/sounds/all_batches/"
    # config["eval_file_name"] = "eval_all_batches.csv"
    # config["rt60diffmin"] = -3
    # config["rt60diffmax"] = 3
    # eval_experiment(config)
