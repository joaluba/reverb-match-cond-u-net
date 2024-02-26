import torch
from tqdm import tqdm
from datetime import datetime
import time
import sys
import os
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
import speechmetrics
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SpeechReverberationModulationEnergyRatio
import numpy as np
import pandas as pd
import soundfile as sf
# my modules
import cond_waveunet_dataset
import cond_waveunet_loss 
import cond_waveunet_model
import joa_helpers as hlp
from cond_waveunet_options import OptionsEval
from torch.utils.data import Subset



class Evaluator(torch.nn.Module):
    def __init__(self,args_test):
        super().__init__()
        self.args_test=args_test
        self.args_train=torch.load(self.args_test.train_args_file)
        # if we are on dacom we need to change the path of the dataset metadata (so far only this data was used)
        self.args_train.df_metadata="/home/Imatge/projects/reverb-match-cond-u-net/dataset-metadata/nonoise2_dacom.csv"
        self.load_eval_objects()
        self.scores = {'label': [],
                'nb_pesq_input': [], 'pesq_input': [], 'stoi_input': [],  'stftloss_input': [],
                'nb_pesq_predict': [], 'pesq_predict': [], 'stoi_predict': [],  'stftloss_predict': [],
                'styleloss_input': [],  'styleloss_predict': []}
        self.speechmetrics=speechmetrics.load(['stoi', 'pesq'], 2)
        
    def load_eval_objects(self):
        # ---- MODELS: ----
        # load reverb encoder
        self.model_reverbenc=cond_waveunet_model.ReverbEncoder(self.args_train).to(self.args_test.device).eval()
        # laod waveunet 
        self.model_waveunet=cond_waveunet_model.waveunet(self.args_train).to(self.args_test.device).eval()
        # ---- LOSS CRITERION: ----
        self.criterion_audio=cond_waveunet_loss.MultiResolutionSTFTLoss().to(self.args_test.device)
        self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(self.args_test.device)
        # ---- TRAINING RESULTS (WEIGHTS): ----
        self.train_results=torch.load(args_test.train_results_file,map_location=self.args_test.device)
        self.model_reverbenc.load_state_dict(self.train_results["model_reverbenc_state_dict"])           
        self.model_waveunet.load_state_dict(self.train_results["model_waveunet_state_dict"])
        # ---- DATASETS: ----
        self.args_train.split=self.args_test.eval_split
        self.testset_orig=cond_waveunet_dataset.DatasetReverbTransfer(self.args_train)
        indices_chosen=self.testset_orig.get_idx_with_rt60diff(self.args_test.rt60diffmin,self.args_test.rt60diffmax)
        self.testset=Subset(self.testset_orig,indices_chosen)
        # ---- DATA LOADER: ----
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args_test.batch_size_eval, shuffle=True, num_workers=6,pin_memory=True)


    def infer(self,data):
        with torch.no_grad():
            # Function to infer target audio
            # ------------------------------
            # get datapoint
            sContent = data[0].to(self.args_test.device)
            sStyle=data[1].to(self.args_test.device)
            sTarget=data[2].to(self.args_test.device)
            # forward pass - get prediction of the ir
            embStyle=self.model_reverbenc(sStyle)
            sPrediction=self.model_waveunet(sContent,embStyle)
            return sContent, sStyle, sTarget, sPrediction
        
    def evaluate(self):
        with torch.no_grad():
            for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):
                # get signals
                sContent, _, sTarget, sPrediction=self.infer(data)
                # get embeddings
                sContent_emb=self.model_reverbenc(sContent)
                sTarget_emb=self.model_reverbenc(sTarget)
                sPrediction_emb=self.model_reverbenc(sPrediction)
                self.add_speech_metrics_batch(sContent,sTarget,sPrediction)
                self.add_stftloss_metrics_batch(sContent,sTarget,sPrediction)
                self.add_styleloss_metrics_batch(sContent_emb,sTarget_emb,sPrediction_emb)
                self.add_label()
        
        self.scores=pd.DataFrame(self.scores)
    
    def add_label(self):
        self.scores["label"]=self.args_test.eval_tag

    def add_speech_metrics_batch(self,input_batch,target_batch,prediction_batch):
        # Function to compute speech metrics for a batch 
        # It has no output and it modifies the dictionary "scores"
        # --------------------------------------------------------
        failcount=0
        metrics_batch = {'label': [],'nb_pesq_input': [], 'pesq_input': [], 'stoi_input': [],
                        'nb_pesq_predict': [], 'pesq_predict': [], 'stoi_predict': []}
        batch_size=input_batch.shape[0]
        for i in range(batch_size):

            target_sig=target_batch[i,0].detach().cpu().numpy()
            prediction_sig=prediction_batch[i,0].detach().cpu().numpy()
            input_sig=input_batch[i,0].detach().cpu().numpy()

            # how close the signal before transformation is to the target:
            try: 
                scores_input=self.speechmetrics(input_sig, target_sig, rate=self.args_train.fs)
            except:
                failcount+=1
                print("Could not compute metrics for this signal for " + str(failcount) + "times")
                sf.write(f"input_sig_{failcount}.wav", input_sig, self.args_train.fs)
                sf.write(f"target_sig_{failcount}.wav", target_sig, self.args_train.fs)
                print("Saved faulty audios")
                metrics_batch["nb_pesq_input"].append(np.nan)
                metrics_batch["pesq_input"].append(np.nan)
                metrics_batch["stoi_input"].append(np.nan)
            else:
                metrics_batch["nb_pesq_input"].append(scores_input["nb_pesq"][0])
                metrics_batch["pesq_input"].append(scores_input["pesq"][0])
                metrics_batch["stoi_input"].append(scores_input["stoi"][0])

            # how close the signal after transformation is to the target:
            try:
                scores_prediction=self.speechmetrics(prediction_sig, target_sig, rate=self.args_train.fs)
            except:
                failcount+=1
                print("Could not compute metrics for this signal for " + str(failcount) + "times")
                sf.write(f"prediction_sig_{failcount}.wav", prediction_sig, self.args_train.fs)
                sf.write(f"target_sig_{failcount}.wav", target_sig, self.args_train.fs)
                print("Saved faulty audios")
                metrics_batch["nb_pesq_predict"].append(np.nan)
                metrics_batch["pesq_predict"].append(np.nan)
                metrics_batch["stoi_predict"].append(np.nan)
            else: 
                metrics_batch["nb_pesq_predict"].append(scores_prediction["nb_pesq"][0])
                metrics_batch["pesq_predict"].append(scores_prediction["pesq"][0])
                metrics_batch["stoi_predict"].append(scores_prediction["stoi"][0])
        
                
        # compute mean score of the batch:
        self.scores["nb_pesq_input"].append(np.nanmean(metrics_batch["nb_pesq_input"]))
        self.scores["pesq_input"].append(np.nanmean(metrics_batch["pesq_input"]))
        self.scores["stoi_input"].append(np.nanmean(metrics_batch["stoi_input"]))
        self.scores["nb_pesq_predict"].append(np.nanmean(metrics_batch["nb_pesq_predict"]))
        self.scores["pesq_predict"].append(np.nanmean(metrics_batch["pesq_predict"]))
        self.scores["stoi_predict"].append(np.nanmean(metrics_batch["stoi_predict"]))

    def add_stftloss_metrics_batch(self,input_batch,target_batch,prediction_batch):
        # Function to compute stft loss for a batch (as a metric) 
        # It has no output and it modifies the dictionary "scores"
        # --------------------------------------------------------
        # how close the signal before transformation is to the target:
        L_sc_i, L_mag_i=self.criterion_audio(input_batch, target_batch)
        self.scores["stftloss_input"].append(float((L_sc_i+L_mag_i).cpu().numpy()))
        # how close the signal after transformation is to the target:
        L_sc_p, L_mag_p=self.criterion_audio(prediction_batch, target_batch)
        self.scores["stftloss_predict"].append(float((L_sc_p+L_mag_p).cpu().numpy()))

    def add_all_losses(self,idx,comp_name,x1,x2):

        if len(x1.shape)<3:
            x1=x1.unsqueeze(0)
        if len(x2.shape)<3:
            x2=x2.unsqueeze(0)
        x1_emb=self.model_reverbenc(x1)
        x2_emb=self.model_reverbenc(x2)
        
        # ----- Load criteria -----
        # stft loss with 4 resolutions
        criterion_stft_loss1 = cond_waveunet_loss.MultiResolutionSTFTLoss(
            fft_sizes=[64, 512, 2048,8192],
            hop_sizes=[32, 256, 1024,4096],
            win_lengths=[64, 512, 2048, 8192],
            window="hann_window")
        
        # stft loss with 5 resolutions
        criterion_stft_loss2 = cond_waveunet_loss.MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024, 2048,4096],
            hop_sizes=[64, 128, 256,512,1024],
            win_lengths=[256, 512, 1024, 2048,4096],
            window="hann_window")

        criterion_logmel=cond_waveunet_loss.LogMelSpectrogramLoss()
        criterion_si_sdr = ScaleInvariantSignalDistortionRatio()
        criterion_srmr = SpeechReverberationModulationEnergyRatio(48000)
        criterion_mse = torch.nn.MSELoss()
        criterion_cosine = torch.nn.CosineSimilarity(dim=2,eps=1e-8)

        # ----- Compute audio losses -----
        L_sc, L_mag = criterion_stft_loss1(x1,x2)
        L_stft1 = L_sc + L_mag
        L_sc, L_mag = criterion_stft_loss2(x1,x2)
        L_stft2 = L_sc + L_mag
        L_logmel=criterion_logmel(x1,x2)
        L_wav_L2=criterion_mse(x1,x2)
        L_si_sdr=criterion_si_sdr(x1,x2)
        L_srmr=criterion_srmr(x2)

        # ----- Compute embedding losses -----
        L_emb_cosine=(1-((torch.mean(criterion_cosine(x1_emb,x2_emb))+ 1) / 2))
        L_emb_euc=torch.dist(x1_emb,x2_emb)

        df_row={'idx':idx, 'compared': comp_name,
                'L_stft1': L_stft1,'L_stft2': L_stft2, 'L_logmel': L_logmel,'L_wav_L2': L_wav_L2, 
                'L_si_sdr': L_si_sdr, 'L_srmr': L_srmr,  'L_emb_cosine': L_emb_cosine, 'L_emb_euc': L_emb_euc}
        
        return df_row
        

    def add_styleloss_metrics_batch(self,input_batch,target_batch,prediction_batch):
        # Function to compute style loss for a batch (as a metric) 
        # It has no output and it modifies the dictionary "scores"
        # --------------------------------------------------------
        # how close the signal before transformation is to the target:
        scores_input=self.criterion_emb(input_batch, target_batch)
        self.scores["styleloss_input"].append(float(scores_input[0].cpu().numpy()))
        # how close the signal after transformation is to the target:
        scores_prediction=self.criterion_emb(prediction_batch, target_batch)
        self.scores["styleloss_predict"].append(float(scores_prediction[0].cpu().numpy()))


def eval_losses(args_test):

    all_losses=pd.DataFrame({'idx':[], 'compared': [],
                'L_stft1': [],'L_stft2': [], 'L_logmel': [],'L_wav_L2': [], 
                'L_si_sdr': [], 'L_srmr': [],  'L_emb_cosine': [], 'L_emb_euc': []})

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

                for i in range(0, len(tmp_evaluation.testset_orig)):
                    print(i)
                    s1r1, s2r2, s1r2_gt, _, s1, _, s1r1b = tmp_evaluation.testset_orig.get_item_test(i)
                    _,r2,_ = tmp_evaluation.testset_orig.get_rirs(i)
                    s2r2_emb=tmp_evaluation.model_reverbenc(s2r2.unsqueeze(0))
                    s1r2_pred=tmp_evaluation.model_waveunet(s1r1.unsqueeze(0),s2r2_emb) 

                    # target : predicion
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"s1r2_gt:s1r2_pred",s1r2_gt,s1r2_pred),index=[0])],ignore_index=True)
                    # target : content
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"s1r2_gt:s1r1",s1r2_gt,s1r1),index=[0])], ignore_index=True)
                    # content_a : content_b
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"s1r1:s1r1b",s1r1,s1r1b),index=[0])], ignore_index=True)

                    # target-anechoic : predicion-anechoic
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"s1r2_gt-a:s1r2_pred-a",s1r2_gt-s1,s1r2_pred-s1),index=[0])], ignore_index=True)
                    # target-anechoic: content-anechoic
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"s1r2_gt-a:s1r1-a",s1r2_gt-s1,s1r1-s1),index=[0])], ignore_index=True)
                    # content_a-anechoic: content_b-anechoic
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"s1r1-a:s1r1b-a",s1r1-s1,s1r1b-s1),index=[0])], ignore_index=True)

                    # deconv(target) : deconv(predicion)
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"d(s1r2_gt):d(s1r2_pred)",hlp.torch_deconv_W(s1r2_gt,r2),hlp.torch_deconv_W(s1r2_pred,r2)),index=[0])], ignore_index=True)
                    # deconv(target): deconv(content)
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"d(s1r2_gt):d(s1r1)",hlp.torch_deconv_W(s1r2_gt,r2),hlp.torch_deconv_W(s1r1,r2)),index=[0])], ignore_index=True)
                    # deconv(content_a): deconv(content_b)
                    all_losses=pd.concat([all_losses, pd.DataFrame(tmp_evaluation.add_all_losses(i,"d(s1r1):d(s1r1b)",hlp.torch_deconv_W(s1r1,r2),hlp.torch_deconv_W(s1r1b,r2)),index=[0])], ignore_index=True)
                    
    all_losses.to_csv(args_test.eval_dir+args_test.eval_file_name, index=False)
    print(f"Saved final results")


def eval_directory(args_test):

    scores_directory=pd.DataFrame({'label': [],
            'nb_pesq_input': [], 'pesq_input': [], 'stoi_input': [],  'stftloss_input': [],
            'nb_pesq_predict': [], 'pesq_predict': [], 'stoi_predict': [],  'stftloss_predict': []})

    for subdir in os.listdir(args_test.eval_dir):
        subdir_path = os.path.join(args_test.eval_dir, subdir)
        # Check if it's a directory
        # if os.path.isdir(subdir_path) & (subdir_path.find("many-to-many") != -1):
        if os.path.isdir(subdir_path):

            print(f"Processing trainig results: {subdir_path}")

             # load results from checkpoints in the directory
            for filename in os.listdir(subdir_path):
                if filename.startswith("checkpoint") & filename.endswith("best.pt"): # only computing measures for the best checkpoint
                # if filename.startswith("checkpoint"): # computing measures for all checkpoints
                    # specify training params file
                    args_test.train_args_file=pjoin(subdir_path,"trainargs.pt")
                    # load checkpoint file
                    args_test.train_results_file=pjoin(subdir_path,filename)
                    # create tag for this evalauation
                    args_test.eval_tag=args_test.train_results_file.split('/')[-2]
                    # create evaluator object
                    tmp_evaluation=Evaluator(args_test)
                    tmp_evaluation.evaluate()
                    # concatenate results for this condition with results for the whole experiment (directory)
                    scores_directory = pd.concat([scores_directory, pd.DataFrame(tmp_evaluation.scores)], ignore_index=True)
                    scores_directory.to_csv(args_test.eval_dir+args_test.eval_file_name, index=False)
                    print(f"Saved intermediate results")
            
            scores_directory.to_csv(args_test.eval_dir+args_test.eval_file_name, index=False)
            print(f"Saved final results")


if __name__ == "__main__":

    args_test=OptionsEval().parse()

    args_test.eval_dir="/media/ssd2/RESULTS-reverb-match-cond-u-net/runs-exp-26-01-2024/"
    args_test.eval_subdir="28-01-2024--15-34_many-to-many_stft"

    args_test.device="cpu"
    args_test.eval_file_name="all_losses.csv"
    eval_losses(args_test)

    # Compute for all
    # args_test.rt60diffmin=-3
    # args_test.rt60diffmax=3
    # args_test.eval_file_name="eval_all.csv"

    # eval_directory(args_test)

    # # Compute for difficult re-reverberation
    # args_test.rt60diffmin=-2
    # args_test.rt60diffmax=-0.5
    # args_test.eval_file_name="eval_rt60diff-50ms.csv"

    # eval_directory(args_test)

    # # Compute for difficult de-reverberation
    # args_test.rt60diffmin=0.5
    # args_test.rt60diffmax=2
    # args_test.eval_file_name="eval_rt60diff+50ms.csv"

    # eval_directory(args_test)





    
            

            

           


   








