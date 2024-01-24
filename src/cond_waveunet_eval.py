import torch
from tqdm import tqdm
from datetime import datetime
import time
import sys
import os
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
import speechmetrics
import numpy as np
import pandas as pd
import soundfile as sf
# my modules
import cond_waveunet_dataset
import cond_waveunet_loss 
import cond_waveunet_model
from cond_waveunet_options import OptionsEval


class Evaluator(torch.nn.Module):
    def __init__(self,args_test):
        super().__init__()
        self.args_test=args_test
        self.args_train=torch.load(args_test.train_args_file)
        self.load_objects(self.args_test,self.args_train)
        self.df_scores = pd.DataFrame({'label': [],
                'nb_pesq_input': [], 'pesq_input': [], 'stoi_input': [],  'stftloss_input': [],
                'nb_pesq_predict': [], 'pesq_predict': [], 'stoi_predict': [],  'stftloss_predict': []})
        self.speechmetrics=speechmetrics.load(['stoi', 'pesq'], 2)
        
    def load_objects(self,args_test,args_train):
        # ---- MODELS: ----
        # load reverb encoder
        self.model_reverbenc=cond_waveunet_model.ReverbEncoder(args_train).to(args_train.device).eval()
        # laod waveunet 
        self.model_waveunet=cond_waveunet_model.waveunet(args_train).to(args_train.device).eval()
        # ---- LOSS CRITERION: ----
        self.criterion_audio=cond_waveunet_loss.MultiResolutionSTFTLoss().to(args_train.device)
        self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(args_train.device)
        # ---- TRAINING RESULTS (WEIGHTS): ----
        self.train_results=torch.load(args_test.train_results_file,map_location=args_train.device)
        self.model_reverbenc.load_state_dict(self.train_results["model_reverbenc_state_dict"])           
        self.model_waveunet.load_state_dict(self.train_results["model_waveunet_state_dict"])
        # ---- DATASETS: ----
        self.testset=cond_waveunet_dataset.DatasetReverbTransfer(args_test.eval_split)
        # ---- DATA LOADER: ----
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=args_test.batch_size_eval, shuffle=True, num_workers=6,pin_memory=True)


    def infer(self,data):
        with torch.no_grad():
            # Function to infer target audio
            # ------------------------------
            # get datapoint
            sContent_in = data[0].to(self.args_train.device)
            sStyle_in=data[1].to(self.args_train.device)
            sTarget_gt=data[2].to(self.args_train.device)
            # forward pass - get prediction of the ir
            embedding_gt=self.model_reverbenc(sStyle_in)
            sTarget_prediction=self.model_waveunet(sContent_in,embedding_gt)
            return sContent_in, sStyle_in, sTarget_gt, sTarget_prediction
        
    def evaluate(self):
        with torch.no_grad():
            for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):
                # get signals
                sContent, _, sTarget, sPrediction=self.infer(self.model_reverbenc, self.model_waveunet, data, self.args_train.device)
                # get embeddings
                sContent_emb=self.model_reverbenc(sContent)
                sTarget_emb=self.model_reverbenc(sTarget)
                sPrediction_emb=self.model_reverbenc(sPrediction)
                self.add_speech_metrics_batch(sContent,sTarget,sPrediction)
                self.add_stftloss_metrics_batch(sContent,sTarget,sPrediction)
                self.add_styleloss_metrics_batch(sContent_emb,sTarget_emb,sPrediction_emb)
                self.add_label()
    
    def add_label(self):
        self.df_scores["label"]=self.args_test.tag*len(self.testloader)

    def add_speech_metrics_batch(self,input_batch,target_batch,prediction_batch):
        # Function to compute speech metrics for a batch 
        # It has no output and it modifies the dictionary "scores"
        # --------------------------------------------------------
        failcount=0
        metrics_batch = {'nb_pesq_input': [], 'pesq_input': [], 'stoi_input': [],
                        'nb_pesq_predict': [], 'pesq_predict': [], 'stoi_predict': []}
        batch_size=input_batch.shape[0]
        for i in range(batch_size):
            target_sig=target_batch[i,0].detach().cpu().numpy()
            prediction_sig=prediction_batch[i,0].detach().cpu().numpy()
            input_sig=input_batch[i,0].detach().cpu().numpy()

            # how close the signal before transformation is to the target:
            try: 
                scores_input=self.metrics(input_sig, target_sig, rate=self.args_train.fs)
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
                scores_prediction=self.metrics(prediction_sig, target_sig, rate=self.args_train.fs)
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
        self.df_scores["nb_pesq_input"].append(np.nanmean(metrics_batch["nb_pesq_input"]))
        self.df_scores["pesq_input"].append(np.nanmean(metrics_batch["pesq_input"]))
        self.df_scores["stoi_input"].append(np.nanmean(metrics_batch["stoi_input"]))
        self.df_scores["nb_pesq_predict"].append(np.nanmean(metrics_batch["nb_pesq_predict"]))
        self.df_scores["pesq_predict"].append(np.nanmean(metrics_batch["pesq_predict"]))
        self.df_scores["stoi_predict"].append(np.nanmean(metrics_batch["stoi_predict"]))

    def add_stftloss_metrics_batch(self,input_batch,target_batch,prediction_batch):
        # Function to compute stft loss for a batch (as a metric) 
        # It has no output and it modifies the dictionary "scores"
        # --------------------------------------------------------
        # how close the signal before transformation is to the target:
        L_sc_i, L_mag_i=self.criterion_audio(input_batch, target_batch)
        self.df_scores["stftloss_input"].append(float((L_sc_i+L_mag_i).cpu().numpy()))
        # how close the signal after transformation is to the target:
        L_sc_p, L_mag_p=self.criterion_audio(prediction_batch, target_batch)
        self.df_scores["stftloss_predict"].append(float((L_sc_p+L_mag_p).cpu().numpy()))

    def add_styleloss_metrics_batch(self,input_batch,target_batch,prediction_batch):
        # Function to compute style loss for a batch (as a metric) 
        # It has no output and it modifies the dictionary "scores"
        # --------------------------------------------------------
        # how close the signal before transformation is to the target:
        scores_input=self.criterion_emb(input_batch, target_batch)
        self.df_scores["styleloss_input"].append(float(scores_input[0].cpu().numpy()))
        # how close the signal after transformation is to the target:
        scores_prediction=self.criterion_emb(prediction_batch, target_batch)
        self.df_scores["styleloss_predict"].append(float(scores_prediction[0].cpu().numpy()))


def eval_directory(args_test):

    scores_directory=pd.DataFrame({'label': [],
            'nb_pesq_input': [], 'pesq_input': [], 'stoi_input': [],  'stftloss_input': [],
            'nb_pesq_predict': [], 'pesq_predict': [], 'stoi_predict': [],  'stftloss_predict': []})

    for subdir in os.listdir(args_test.eval_dir):
        subdir_path = os.path.join(args_test.eval_dir, subdir)
        # Check if it's a directory
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
                    tmp_evaluation.eval()
                    # concatenate results for this condition with results for the whole experiment (directory)
                    scores_directory = pd.concat([scores_directory, pd.DataFrame(tmp_evaluation.scores)], ignore_index=True)
                    scores_directory.to_csv(args_test.resultsdir+args_test.evaluation_name, index=False)
                    print(f"Saved intermediate results")
            
            scores_directory.to_csv(args_test.resultsdir+args_test.evaluation_name, index=False)
            print(f"Saved final results")


if __name__ == "__main__":

    args_test=OptionsEval().parse()

    eval_directory(args_test)



    
            

            

           


   








