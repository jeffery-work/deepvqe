"""
# File       : SETrainer.py
# Time       : 2024/3/14 21:39
# Author     : fei jie
# version    : 1.0
# Description: 
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import torch.nn.functional as functional

from Train.TrainerBase import TrainerBase
plt.switch_backend("agg")

# speech enhancement trainer
class SETrainer(TrainerBase):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            optimizer,
            loss_function,
            train_dataloader,
            validation_dataloader,
    ):
        super(SETrainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        # self.nfft = config["trainer"]["feature"]["nfft"]
        # self.hop_length = config["trainer"]["feature"]["hop_len"]
        # self.win_len = config["trainer"]["feature"]["win_len"]
        self.nfft = 320
        self.hop_length = 160
        self.win_len = 320
        self.uloss = torch.nn.MSELoss()


    def _Forward(self, noisy, clean):
        noisy = torch.squeeze(noisy.to(self.device))  # [B, T]
        clean = torch.squeeze(clean.to(self.device))  # [B, T]
        noisy_spec = torch.stft(
            noisy,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.win_len,
            window=torch.hann_window(self.win_len).to(self.device),
            return_complex=True)  # complex[B, F, T]
        clean_spec = torch.stft(
            clean,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.win_len,
            window=torch.hann_window(self.win_len).to(self.device),
            return_complex=True)  # complex[B, F, T]
        noisy_mag = torch.unsqueeze(torch.abs(noisy_spec), dim=1)
        clean_mag = torch.unsqueeze(torch.abs(clean_spec), dim=1)
        label_mask = noisy_mag / clean_mag
        enhanced_mask = torch.squeeze(self.model(noisy_mag))
        enhanced_spec = noisy_spec * enhanced_mask
        enhanced = torch.istft(
            enhanced_spec,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.win_len,
            window=torch.hann_window(self.win_len).to(self.device),
            center=True,
            return_complex=False)
        return clean, enhanced, label_mask, enhanced_mask

    def _TrainEpoch(self, epoch):
        loss_total = 0.0
        # for noisy_mag, noisy_mfcc, labels_list,  num_frames_list in self.train_dataloader:
        for noisy, clean, _ in self.train_dataloader:
            self.optimizer.zero_grad()
            noisy = torch.squeeze(noisy.to(self.device))  # [B, T]
            clean = torch.squeeze(clean.to(self.device))  # [B, T]
            noisy_spec = torch.stft(
                noisy,
                n_fft=self.nfft,
                hop_length=self.hop_length,
                win_length=self.win_len,
                window=torch.hann_window(self.win_len).to(self.device),
                return_complex=True)  # complex[B, F, T]
            clean_spec = torch.stft(
                clean,
                n_fft=self.nfft,
                hop_length=self.hop_length,
                win_length=self.win_len,
                window=torch.hann_window(self.win_len).to(self.device),
                return_complex=True)  # complex[B, F, T]
            noisy_mag = torch.unsqueeze(torch.abs(noisy_spec), dim=1)
            clean_mag = torch.unsqueeze(torch.abs(clean_spec), dim=1)
            label_mask = torch.squeeze(clean_mag/(noisy_mag+0.000001))
            enhanced_mask = torch.squeeze(self.model(noisy_mag))
            enhanced_spec = noisy_spec*enhanced_mask
            enhanced = torch.istft(
                enhanced_spec,
                n_fft=self.nfft,
                hop_length=self.hop_length,
                win_length=self.win_len,
                window=torch.hann_window(self.win_len).to(self.device),
                center=True,
                return_complex=False)
            # clean, enhanced, label_mask, enhanced_mask = self._forwad_process(noisy, clean)
            loss = self.loss_function(label_mask, enhanced_mask)
            if epoch > 100:
                loss = loss*100/epoch
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)
        print(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad() # no need to calculate backward gradient
    def _ValidationEpoch(self, epoch, save_ep):
        loss_total = 0.0
        score_total = 0
        # visualization_limit = self.validation_custom_config["visualization_limit"]
        # n_fft = self.validation_custom_config["n_fft"]
        # hop_length = self.validation_custom_config["hop_length"]
        # win_length = self.validation_custom_config["win_length"]
        # batch_size = self.validation_custom_config["batch_size"]
        # unfold_size = self.validation_custom_config["unfold_size"]
        for i, (noisy, clean, name) in tqdm(enumerate(self.validation_dataloader), desc="Inference"):
            noisy = torch.squeeze(noisy.to(self.device))  # [B, T]
            clean = torch.squeeze(clean.to(self.device))  # [B, T]
            noisy_spec = torch.stft(
                noisy,
                n_fft=self.nfft,
                hop_length=self.hop_length,
                win_length=self.win_len,
                window=torch.hann_window(self.win_len).to(self.device),
                return_complex=True)  # complex[B, F, T]
            clean_spec = torch.stft(
                clean,
                n_fft=self.nfft,
                hop_length=self.hop_length,
                win_length=self.win_len,
                window=torch.hann_window(self.win_len).to(self.device),
                return_complex=True)  # complex[B, F, T]
            noisy_mag = torch.unsqueeze(torch.abs(noisy_spec), dim=1)
            clean_mag = torch.unsqueeze(torch.abs(clean_spec), dim=1)
            label_mask = torch.squeeze(clean_mag/(noisy_mag+0.000001))
            enhanced_mask = torch.squeeze(self.model(noisy_mag))
            enhanced_spec = noisy_spec * enhanced_mask
            enhanced = torch.istft(
                enhanced_spec,
                n_fft=self.nfft,
                hop_length=self.hop_length,
                win_length=self.win_len,
                window=torch.hann_window(self.win_len).to(self.device),
                center=True,
                return_complex=False)
            loss_total += self.loss_function(label_mask, enhanced_mask).item()
            # loss_total += self.uloss(label_mask, enhanced_mask).item()
            # no need to backward process, so detach and transmit it to cpu, convert tensor to numpy.array()
            noisy = noisy.detach().cpu().numpy()
            clean = clean.detach().cpu().numpy()
            enhanced = enhanced.detach().cpu().numpy()
            # assert len(noisy) == len(clean) == len(enhanced)
            if epoch >= save_ep:
                # save enhanced wav
                for j in range(np.size(enhanced, 0)):
                    AudioWrite('/data4/xiaolonz/dns_database/vail/enhanced/enhanced_'+str(i*np.size(enhanced, 0)+j)+'.wav',
                               enhanced[j, :])
            if epoch >= save_ep:
                score_total += self.metrics_visualization(noisy, clean, enhanced, epoch)

        self.writer.add_scalar(f"Loss/Validation", loss_total / len(self.validation_dataloader), epoch)
        print(f"Loss/Validation", loss_total / len(self.validation_dataloader), epoch)
        return score_total/len(self.validation_dataloader)

