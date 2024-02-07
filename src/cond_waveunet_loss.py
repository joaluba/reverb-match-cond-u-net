# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

from distutils.version import LooseVersion

import torch
import torch.nn.functional as F

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


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
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False
        )
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
        fft_sizes=[64, 512, 2048,8192],
        hop_sizes=[32, 256, 1024,4096],
        win_lengths=[64, 512, 2048, 8192],
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
        self.losstype=args.losstype
        self.load_criterions(args.device)

    def load_criterions(self,device):
        if self.losstype=="stft" or self.losstype=="stft+rev" or self.losstype=="rev":
            self.criterion_audio=MultiResolutionSTFTLoss().to(device)

        elif self.losstype=="stft+emb":
            self.criterion_audio=MultiResolutionSTFTLoss().to(device)
            self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(device)

        elif self.losstype=="stft+rev+emb":
            self.criterion_audio=MultiResolutionSTFTLoss().to(device)
            self.criterion_emb=torch.nn.CosineSimilarity(dim=2,eps=1e-8).to(device)

        else:
            print("this loss is not implemented")

    def forward(self, data, model_waveunet, model_reverbenc, device):
        # get datapoint
        sContent_in = data[0].to(device)
        sStyle_in=data[1].to(device)
        sTarget=data[2].to(device)
        sAnecho=data[3].to(device)
        # forward pass - get prediction of the ir
        embStyle=model_reverbenc(sStyle_in)
        sPrediction=model_waveunet(sContent_in,embStyle)

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

        elif self.losstype=="stft+rev":
            L_sc_rev, L_mag_rev = self.criterion_audio(sTarget.squeeze(1)-sAnecho.squeeze(1), sPrediction.squeeze(1)-sAnecho.squeeze(1))
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft=L_sc+L_mag
            L_rev=L_sc_rev+L_mag_rev
            L= [L_stft,L_rev]
            L_names=["L_stft", "L_rev"]

        elif self.losstype=="stft+emb":
            # get the embedding of the prediction
            embTarget=model_reverbenc(sTarget)
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft=L_sc + L_mag
            L_emb=(1-((torch.mean(self.criterion_emb(embStyle,embTarget))+ 1) / 2))
            L=[L_stft, L_emb]
            L_names=["L_stft", "L_emb"]

        elif self.losstype=="stft+rev+emb":
            # get the embedding of the prediction
            embTarget=model_reverbenc(sTarget)
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

