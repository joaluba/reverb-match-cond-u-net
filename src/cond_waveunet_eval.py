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
from cond_waveunet_options import Options

def add_speech_metrics_batch(input_batch,target_batch,prediction_batch,sr,metrics,scores):
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
            scores_input=metrics(input_sig, target_sig, rate=sr)
        except:
            failcount+=1
            print("Could not compute metrics for this signal for " + str(failcount) + "times")
            sf.write(f"input_sig_{failcount}.wav", input_sig, sr)
            sf.write(f"target_sig_{failcount}.wav", target_sig, sr)
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
            scores_prediction=metrics(prediction_sig, target_sig, rate=sr)
        except:
            failcount+=1
            print("Could not compute metrics for this signal for " + str(failcount) + "times")
            sf.write(f"prediction_sig_{failcount}.wav", prediction_sig, sr)
            sf.write(f"target_sig_{failcount}.wav", target_sig, sr)
            print("Saved faulty audios")
            metrics_batch["nb_pesq_predict"].append(np.nan)
            metrics_batch["pesq_predict"].append(np.nan)
            metrics_batch["stoi_predict"].append(np.nan)
        else: 
            metrics_batch["nb_pesq_predict"].append(scores_prediction["nb_pesq"][0])
            metrics_batch["pesq_predict"].append(scores_prediction["pesq"][0])
            metrics_batch["stoi_predict"].append(scores_prediction["stoi"][0])

    # compute mean score of the batch:
    scores["nb_pesq_input"].append(np.nanmean(metrics_batch["nb_pesq_input"]))
    scores["pesq_input"].append(np.nanmean(metrics_batch["pesq_input"]))
    scores["stoi_input"].append(np.nanmean(metrics_batch["stoi_input"]))
    scores["nb_pesq_predict"].append(np.nanmean(metrics_batch["nb_pesq_predict"]))
    scores["pesq_predict"].append(np.nanmean(metrics_batch["pesq_predict"]))
    scores["stoi_predict"].append(np.nanmean(metrics_batch["stoi_predict"]))

def add_stftloss_metrics_batch(input_batch,target_batch,prediction_batch,sr,criterion,scores):
    # Function to compute stft loss for a batch (as a metric) 
    # It has no output and it modifies the dictionary "scores"
    # --------------------------------------------------------
    # how close the signal before transformation is to the target:
    L_sc_i, L_mag_i=criterion(input_batch, target_batch)
    scores["stftloss_input"].append(float((L_sc_i+L_mag_i).cpu().numpy()))
    # how close the signal after transformation is to the target:
    L_sc_p, L_mag_p=criterion(prediction_batch, target_batch)
    scores["stftloss_predict"].append(float((L_sc_p+L_mag_p).cpu().numpy()))

def add_styleloss_metrics_batch(input_batch,target_batch,prediction_batch,sr,criterion,scores):
    # Function to compute style loss for a batch (as a metric) 
    # It has no output and it modifies the dictionary "scores"
    # --------------------------------------------------------
    # how close the signal before transformation is to the target:
    scores_input=criterion(input_batch, target_batch)
    scores["styleloss_input"].append(float(scores_input[0].cpu().numpy()))
    # how close the signal after transformation is to the target:
    scores_prediction=criterion(prediction_batch, target_batch)
    scores["styleloss_predict"].append(float(scores_prediction[0].cpu().numpy()))



def infer(model_reverbenc, model_waveunet, data, device):
    # Function to infer target audio
    # ------------------------------
    # get datapoint
    sContent_in = data[0].to(device)
    sStyle_in=data[1].to(device)
    sTarget_gt=data[2].to(device)
    # forward pass - get prediction of the ir
    embedding_gt=model_reverbenc(sStyle_in)
    sTarget_prediction=model_waveunet(sContent_in,embedding_gt)
    return sContent_in, sStyle_in, sTarget_gt, sTarget_prediction
    

def eval_speechmetrics(model_reverbenc, model_waveunet,  testloader, args):

    device=args.device

    if args.audio_criterion=="multi_stft_loss":
        audio_criterion = cond_waveunet_loss.MultiResolutionSTFTLoss()
    if args.emb_criterion=="cosine_similarity":
        emb_criterion = torch.nn.CosineSimilarity(dim=2,eps=1e-8)

    # move components to device
    model_reverbenc=model_reverbenc.to(device)
    model_waveunet=model_waveunet.to(device)
    audio_criterion=audio_criterion.to(device)
    emb_criterion=emb_criterion.to(device)
    

    # ------------- EVALUATION START: -------------
    model_waveunet.eval() 
    model_reverbenc.eval()
    with torch.no_grad():
        
        metrics = speechmetrics.load(['stoi', 'pesq'], 2)
        scores_1_model = {'label': [],
                    'nb_pesq_input': [], 'pesq_input': [], 'stoi_input': [],  'stftloss_input': [], 'styleloss_input': [],
                    'nb_pesq_predict': [], 'pesq_predict': [], 'stoi_predict': [],  'stftloss_predict': [], 'styleloss_predict': []}

        for j, data in tqdm(enumerate(testloader),total = len(testloader)):

            # get signals
            sContent, _, sTarget, sPrediction=infer(model_reverbenc, model_waveunet, data, device)
            # get embeddings
            sContent_emb=model_reverbenc(sContent)
            sTarget_emb=model_reverbenc(sTarget)
            sPrediction_emb=model_reverbenc(sPrediction)

            add_speech_metrics_batch(sContent,sTarget,sPrediction,args.fs,metrics,scores_1_model)
            add_stftloss_metrics_batch(sContent,sTarget,sPrediction,args.fs,audio_criterion,scores_1_model)
            add_styleloss_metrics_batch(sContent_emb,sTarget_emb,sPrediction_emb,args.fs,emb_criterion,scores_1_model)
        
        return scores_1_model



if __name__ == "__main__":

    from torch.utils.data import Subset

    resultsdir="/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-11-12-2023/"

    scores_all_models = pd.DataFrame({'label': [],
                'nb_pesq_input': [], 'pesq_input': [], 'stoi_input': [],  'stftloss_input': [],
                'nb_pesq_predict': [], 'pesq_predict': [], 'stoi_predict': [],  'stftloss_predict': []})
    
    BATCH_SIZE_EVAL=24


    for i, subdir in enumerate(os.listdir(resultsdir)):
        subdir_path = os.path.join(resultsdir, subdir)
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            print(f"Processing trainig results: {subdir_path}")
            current_model_tag=subdir.split("_", 1)[-1]
            # load params 
            args=torch.load(pjoin(subdir_path,"trainargs.pt"))
            # load training results
            # train_results=torch.load(pjoin(subdir_path,"checkpoint27.pt"),map_location=args.device)
            for filename in os.listdir(subdir_path):
                if filename.startswith("checkpoint"):
                    print(f"Processing checkpoint results: {filename}")
                    current_checkpoint_tag=current_model_tag+filename
                    train_results=torch.load(pjoin(subdir_path,filename),map_location=args.device)
                    # load reverb encoder
                    model_ReverbEncoder=cond_waveunet_model.ReverbEncoder(args)
                    model_ReverbEncoder.load_state_dict(train_results["model_reverbenc_state_dict"])
                    model_ReverbEncoder.to("cuda")
                    model_ReverbEncoder.eval()
                    # load waveunet
                    model_waveunet=cond_waveunet_model.waveunet(args)
                    model_waveunet.load_state_dict(train_results["model_waveunet_state_dict"])
                    model_waveunet.to("cuda")
                    model_waveunet.eval()
                    # load test dataset
                    args.split="test"
                    testset=cond_waveunet_dataset.DatasetReverbTransfer(args)
                    # pick only first 1000 data points from the test set
                    subset_indices = range(1000)
                    testset = Subset(testset, subset_indices)
                    # create dataloader
                    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=6)
                    scores_1_model = eval_speechmetrics(model_ReverbEncoder, model_waveunet, testloader, args)
                    scores_1_model["label"]=[current_checkpoint_tag]*len(testloader)
                    scores_all_models = pd.concat([scores_all_models, pd.DataFrame(scores_1_model)], ignore_index=True)
                    scores_all_models.to_csv(resultsdir+'evaluation_metrics_evol.csv', index=False)
                    print(f"Saved intermediate results")


    scores_all_models.to_csv(resultsdir+'evaluation_metrics_evol.csv', index=False)
    print(f"Saved final results")









