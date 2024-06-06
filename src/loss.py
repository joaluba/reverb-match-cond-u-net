# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

from distutils.version import LooseVersion

import torch
import torchaudio
import torch.nn.functional as F
import loss_mel
import loss_stft
import loss_waveform



class load_chosen_loss(torch.nn.Module):
    def __init__(self,config,losstype):
        super().__init__()
        self.config=config
        self.losstype=losstype
        self.load_criterions(config["device"])

    def load_criterions(self,device):
        if self.losstype=="stft":
            self.criterion_audio=loss_stft.MultiResolutionSTFTLoss().to(device)

        elif self.losstype=="stft+vae":
            self.criterion_audio=loss_stft.MultiResolutionSTFTLoss().to(device)
            # self.beta_schedule= [(i / (self.config["num_epochs"]/2)) if i < self.config["num_epochs"]/2 else 1 for i in range(self.config["num_epochs"])]
            self.beta_schedule= [1] * self.config["num_epochs"]

        elif self.losstype=="logmel+vae":
            self.criterion_audio=loss_mel.MultiMelSpectrogramLoss().to(device)
            # self.beta_schedule= [(i / (self.config["num_epochs"]/2)) if i < self.config["num_epochs"]/2 else 1 for i in range(self.config["num_epochs"])]
            self.beta_schedule= [1] * self.config["num_epochs"]

        elif self.losstype=="logmel":
            self.criterion_audio=loss_mel.MultiMelSpectrogramLoss().to(device)
        
        elif self.losstype=="wave":
            self.criterion_audio=loss_waveform.MultiWindowShapeLoss().to(device)

        elif self.losstype=="logmel+wave":
            self.criterion_audio1=loss_mel.MultiMelSpectrogramLoss().to(device)
            self.criterion_audio2=loss_waveform.MultiWindowShapeLoss().to(device)

        elif self.losstype=="stft+emb":
            self.criterion_audio=loss_stft.MultiResolutionSTFTLoss().to(device)
            self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(device)

        else:
            print("this loss is not implemented")

    def forward(self, epoch, data, model_combined, device):
        # get datapoint
        sContent_in = data[0].to(device) # s1r1 - content
        sStyle_in=data[1].to(device) # s2r2 - style
        sTarget=data[2].to(device) # s1r2 - target

        # forward pass - get prediction 
        embStyle=model_combined.conditioning_network(sStyle_in)
        sPrediction=model_combined(sContent_in,sStyle_in)
        if bool(self.config["is_vae"]):
            sPrediction, mu, log_var = sPrediction

        if self.losstype=="stft":
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft = L_sc+ L_mag 
            L = [L_stft]
            L_names = ["L_stft"]
        
        elif self.losstype=="logmel":
            L_logmel = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L = [L_logmel]
            L_names =["L_logmel"]

        elif self.losstype=="logmel+wave":
            L_logmel = self.criterion_audio1(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_wave = self.criterion_audio2(sTarget.squeeze(1), sPrediction.squeeze(1))
            L = [L_logmel,L_wave]
            L_names =["L_logmel","L_wave"]

        elif self.losstype=="wave":
            L_wave = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L = [L_wave]
            L_names =["L_wave"]

        elif self.losstype=="stft+vae":
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft = L_sc + L_mag 
            L_vae = self.beta_schedule[epoch]*(-torch.sum(1+ log_var - mu.pow(2)- log_var.exp()))
            L = [L_stft,L_vae]
            L_names = ["L_stft","L_vae"]

        elif self.losstype=="logmel+vae":
            L_logmel = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_vae = self.beta_schedule[epoch]*(-torch.sum(1+ log_var - mu.pow(2)- log_var.exp()))
            L = [L_logmel,L_vae]
            L_names = ["L_logmel","L_vae"]

        elif self.losstype=="stft+emb":
            # get the embedding of the prediction
            embTarget = model_combined.conditioning_network(sTarget)
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft = L_sc + L_mag
            L_emb = (1-((torch.mean(self.criterion_emb(embStyle,embTarget))+ 1) / 2))
            L = [L_stft, L_emb]
            L_names = ["L_stft", "L_emb"]

        else:
            print("the forward for this loss is not implemented")

        return L, L_names
    

if __name__ == "__main__":

    # ---- check if loss definition is correct: ----
    model = loss_mel.MultiMelSpectrogramLoss()
    x = torch.randn(2, 16000)
    y = torch.randn(2, 16000)

    loss = model(x, y)
    print(loss)

