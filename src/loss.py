# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

from distutils.version import LooseVersion

import torch
import torchaudio
import torch.nn.functional as F
import helpers as hlp

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


class LogMelSpectrogramLoss(torch.nn.Module):
    def __init__(self, sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128):
        super(LogMelSpectrogramLoss, self).__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, predicted_audio, target_audio):
        # Compute the log mel spectrogram for predicted and target audio
        predicted_mel_spec = torch.log(self.mel_spectrogram(predicted_audio) + 1e-9)  # Add epsilon to avoid log(0)
        target_mel_spec = torch.log(self.mel_spectrogram(target_audio) + 1e-9)

        # Compute the MSE loss between log mel spectrograms of predicted and target audio
        loss = self.mse_loss(predicted_mel_spec, target_mel_spec)
        return loss
    

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        real = x_stft.real
        imag = x_stft.imag
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[256, 512, 1024, 2048,4096],
        hop_sizes=[64, 128, 256,512,1024],
        win_lengths=[256, 512, 1024, 2048,4096],
        window="hann_window",
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T)
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss

class LossOfChoice(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.losstype=args.losstype
        self.load_criterions(args.device)


    def load_criterions(self,device):
        if self.losstype=="stft" or self.losstype=="early" or self.losstype=="late"  or self.losstype=="stft+rev" or self.losstype=="rev" or self.losstype=="early+late" or self.losstype=="stft+early+late":
            self.criterion_audio=MultiResolutionSTFTLoss().to(device)

        elif self.losstype=="stft+vae":
            self.criterion_audio=MultiResolutionSTFTLoss().to(device)
            # self.beta_schedule= [(i / (self.args.num_epochs/2)) if i < self.args.num_epochs/2 else 1 for i in range(self.args.num_epochs)]
            self.beta_schedule= [1] * self.args.num_epochs

        elif self.losstype=="logmel+vae":
            self.criterion_audio=LogMelSpectrogramLoss().to(device)
            # self.beta_schedule= [(i / (self.args.num_epochs/2)) if i < self.args.num_epochs/2 else 1 for i in range(self.args.num_epochs)]
            self.beta_schedule= [1] * self.args.num_epochs

        elif self.losstype=="logmel":
            self.criterion_audio=LogMelSpectrogramLoss().to(device)

        elif self.losstype=="stft+logmel":
            self.criterion1_audio=MultiResolutionSTFTLoss().to(device)
            self.criterion2_audio=LogMelSpectrogramLoss().to(device)

        elif self.losstype=="stft+emb":
            self.criterion_audio=MultiResolutionSTFTLoss().to(device)
            self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(device)

        elif self.losstype=="stft+rev+emb" or self.losstype=="stft+early+late+emb":
            self.criterion_audio=MultiResolutionSTFTLoss().to(device)
            self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(device)
        
        elif self.losstype=="stft+rev+emb+trip":
            self.criterion_audio=MultiResolutionSTFTLoss().to(device)
            self.criterion_trip=torch.nn.TripletMarginLoss().to(device)
            self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(device)
        else:
            print("this loss is not implemented")

    def forward(self, epoch, data, model_combined, device):
        # get datapoint
        sContent_in = data[0].to(device) # s1r1 - content
        sStyle_in=data[1].to(device) # s2r2 - style
        sTarget=data[2].to(device) # s1r2 - target
        sAnecho=data[3].to(device) # s1 - anechoic
        sEarly=data[4].to(device) # s1r2_early - early reflections
        sLate=data[5].to(device) # s1r2_late - late reverb
        # sPositive=data[4].to(device) # s2r1
        # forward pass - get prediction of the ir
        embStyle=model_combined.conditioning_network(sStyle_in)
        sPrediction=model_combined(sContent_in,sStyle_in)
        if bool(self.args.is_vae):
            sPrediction, mu, log_var = sPrediction

        if self.losstype=="stft":
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft= L_sc+ L_mag 
            L=[L_stft]
            L_names=["L_stft"]

        elif self.losstype=="rev":
            L_sc_rev, L_mag_rev = self.criterion_audio(sTarget.squeeze(1)-sAnecho.squeeze(1), sPrediction.squeeze(1)-sAnecho.squeeze(1))
            L_rev= L_sc_rev + L_mag_rev 
            L=[L_rev]
            L_names=["L_rev"]

        elif self.losstype=="late":
            L_sc_late, L_mag_late = self.criterion_audio(sTarget.squeeze(1)-sAnecho.squeeze(1), sPrediction.squeeze(1)-sAnecho.squeeze(1))
            L_late= L_sc_late + L_mag_late 
            L=[L_late]
            L_names=["L_late"]

        elif self.losstype=="early":
            L_sc_early, L_mag_early= self.criterion_audio(sTarget.squeeze(1)-sAnecho.squeeze(1), sPrediction.squeeze(1)-sAnecho.squeeze(1))
            L_early= L_sc_early + L_mag_early
            L=[L_early]
            L_names=["L_early"]

        elif self.losstype=="stft+vae":
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft= L_sc+ L_mag 
            L_vae = self.beta_schedule[epoch]*(-torch.sum(1+ log_var - mu.pow(2)- log_var.exp()))

            L=[L_stft,L_vae]
            L_names=["L_stft","L_vae"]

        
        elif self.losstype=="logmel+vae":
            L_logmel = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_vae = self.beta_schedule[epoch]*(-torch.sum(1+ log_var - mu.pow(2)- log_var.exp()))
            L=[L_logmel,L_vae]
            L_names=["L_logmel","L_vae"]


        elif self.losstype=="stft+rev":
            L_sc_rev, L_mag_rev = self.criterion_audio(sTarget.squeeze(1)-sAnecho.squeeze(1), sPrediction.squeeze(1)-sAnecho.squeeze(1))
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft=L_sc+L_mag
            L_rev=L_sc_rev+L_mag_rev
            L= [L_stft,L_rev]
            L_names=["L_stft", "L_rev"]

        elif self.losstype=="stft+emb":
            # get the embedding of the prediction
            embTarget=model_combined.conditioning_network(sTarget)
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft=L_sc + L_mag
            L_emb=(1-((torch.mean(self.criterion_emb(embStyle,embTarget))+ 1) / 2))
            L=[L_stft, L_emb]
            L_names=["L_stft", "L_emb"]

        elif self.losstype=="stft+logmel":
            L_sc, L_mag = self.criterion1_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft=L_sc + L_mag
            L_logmel = self.criterion2_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L=[L_stft, L_logmel]
            L_names=["L_stft", "L_logmel"]

        elif self.losstype=="logmel":
            L_logmel = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L=[L_logmel]
            L_names=["L_logmel"]

        elif self.losstype=="stft+rev+emb":
            # get the embedding of the prediction
            embTarget=model_combined.conditioning_network(sTarget)
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_sc_rev, L_mag_rev = self.criterion_audio(sTarget.squeeze(1)-sAnecho.squeeze(1), sPrediction.squeeze(1)-sAnecho.squeeze(1))
            L_stft=L_sc+L_mag
            L_rev=L_sc_rev+L_mag_rev
            L_emb=(1-((torch.mean(self.criterion_emb(embStyle,embTarget))+ 1) / 2))
            L = [L_stft, L_rev, L_emb]
            L_names=["L_stft", "L_rev", "L_emb"]
            # Append the losses to the text file
            with open("losses.txt", 'a') as file:
                file.write(f"{L_stft} {L_rev} {L_emb}\n")

        elif self.losstype=="stft+early+late+emb":
            # get the embedding of the prediction
            embTarget=model_combined.conditioning_network(sTarget)
            L_sc_stft, L_mag_stft = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_sc_early, L_mag_early = self.criterion_audio(hlp.torch_standardize_max_abs(sTarget.squeeze(1)-sLate.squeeze(1)), hlp.torch_standardize_max_abs(sPrediction.squeeze(1)-sLate.squeeze(1)))
            L_sc_late, L_mag_late = self.criterion_audio(hlp.torch_standardize_max_abs(sTarget.squeeze(1)-sEarly.squeeze(1)), hlp.torch_standardize_max_abs(sPrediction.squeeze(1)-sEarly.squeeze(1)))
            L_stft = L_sc_stft + L_mag_stft
            L_early = L_sc_early+L_mag_early
            L_late = L_sc_late+L_mag_late
            L_emb=(1-((torch.mean(self.criterion_emb(embStyle,embTarget))+ 1) / 2))
            L = [L_stft, L_early, L_late, L_emb]
            L_names=["L_stft", "L_early", "L_late", "L_emb"]
        
        elif self.losstype=="early+late":
            L_sc_early, L_mag_early = self.criterion_audio(hlp.torch_standardize_max_abs(sTarget.squeeze(1)-sLate.squeeze(1)), hlp.torch_standardize_max_abs(sPrediction.squeeze(1)-sLate.squeeze(1)))
            L_sc_late, L_mag_late = self.criterion_audio(hlp.torch_standardize_max_abs(sTarget.squeeze(1)-sEarly.squeeze(1)), hlp.torch_standardize_max_abs(sPrediction.squeeze(1)-sEarly.squeeze(1)))
            L_early=L_sc_early+L_mag_early
            L_late=L_sc_late+L_mag_late
            L = [L_early, L_late]
            L_names=["L_early", "L_late"]

        elif self.losstype=="stft+early+late":
            L_sc_stft, L_mag_stft = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_sc_early, L_mag_early = self.criterion_audio(hlp.torch_standardize_max_abs(sTarget.squeeze(1)-sLate.squeeze(1)), hlp.torch_standardize_max_abs(sPrediction.squeeze(1)-sLate.squeeze(1)))
            L_sc_late, L_mag_late = self.criterion_audio(hlp.torch_standardize_max_abs(sTarget.squeeze(1)-sEarly.squeeze(1)), hlp.torch_standardize_max_abs(sPrediction.squeeze(1)-sEarly.squeeze(1)))
            L_stft = L_sc_stft + L_mag_stft
            L_early = L_sc_early+L_mag_early
            L_late = L_sc_late+L_mag_late
            L = [L_stft,L_early, L_late]
            L_names=["L_stft","L_early", "L_late"]
        
        # elif self.losstype=="stft+rev+emb+trip":
        #     # get the embedding of the prediction
        #     embTarget=model_reverbenc(sTarget)
        #     embPositive=model_reverbenc(sPositive)
        #     L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
        #     L_sc_rev, L_mag_rev = self.criterion_audio(sTarget.squeeze(1)-sAnecho.squeeze(1), sPrediction.squeeze(1)-sAnecho.squeeze(1))
        #     L_stft=L_sc+L_mag
        #     L_rev=L_sc_rev+L_mag_rev
        #     L_emb=(1-((torch.mean(self.criterion_emb(embStyle,embTarget))+ 1) / 2))
        #     L_trip=self.criterion_trip(embStyle,embPositive,embTarget)
        #     L = [L_stft, L_rev, L_emb, L_trip]
        #     L_names=["L_stft", "L_rev", "L_emb", "L_trip"]


        else:
            print("this loss is not implemented")

        return L,L_names
    

if __name__ == "__main__":

    # ---- check if loss definition is correct: ----
    model = MultiResolutionSTFTLoss()
    x = torch.randn(2, 16000)
    y = torch.randn(2, 16000)

    loss = model(x, y)
    print(loss)

