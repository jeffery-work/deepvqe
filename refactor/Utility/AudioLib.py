"""
# File       : AudioLib.py
# Time       : 2024/3/14 22:05
# Author     : fei jie
# version    : 1.0
# Description: 
"""
import os
import numpy as np
import soundfile as sf
import subprocess
import glob
import librosa
import random
import tempfile

from scipy import signal

EPS = np.finfo(float).eps
np.random.seed(0)


def IsClipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def Normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def NormalizeSegmentalRMS(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def AudioRead(path, norm=False, start=0, stop=None, target_level=-25):
    '''Function to read audio'''
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')
        return (None, None)

    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio ** 2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms + EPS)
            audio = audio * scalar
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0) / audio.shape[0]
        if norm:
            audio = Normalize(audio, target_level)

    return audio, sample_rate


def AudioWrite(destpath, audio, sample_rate=16000, norm=False, target_level=-25, \
               clipping_threshold=0.99, clip_test=False):
    '''Function to write audio'''

    if clip_test:
        if IsClipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError("Clipping detected in audiowrite()! " + \
                             destpath + " file not written to disk.")

    if norm:
        audio = Normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio / max_amp * (clipping_threshold - EPS)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return


def AddReverberation(clean_speech, rir):
    ''' Function to add reverb'''
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0: clean_speech.shape[0]]

    return reverb_speech


def AddClipping(audio, max_thresh_perc=0.8):
    '''Function to add clipping'''
    threshold = max(abs(audio)) * max_thresh_perc
    audio_clipped = np.clip(audio, -threshold, threshold)
    return audio_clipped


def SnrMixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various SNR levels'''
    cfg = params['cfg']
    # assert (len(clean) > len(noise))  # this case is not allowed

    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))

    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = Normalize(clean, target_level)
    rmsclean = (clean ** 2).mean() ** 0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = Normalize(noise, target_level)
    rmsnoise = (noise ** 2).mean() ** 0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    clipped_flag = IsClipped(noisyspeech)
    if clipped_flag:
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level, clipped_flag


def SegmentalSnrMixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    cfg = params['cfg']
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))
    clean = clean / (max(abs(clean)) + EPS)
    noise = noise / (max(abs(noise)) + EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = NormalizeSegmentalRMS(clean, rms=rmsclean, target_level=target_level)
    noise = NormalizeSegmentalRMS(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if IsClipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = (noise_win ** 2).mean() ** 0.5
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs) != 0:
        noise_rms = (noise_active_segs ** 2).mean() ** 0.5
    else:
        noise_rms = EPS

    if len(clean_active_segs) != 0:
        clean_rms = (clean_active_segs ** 2).mean() ** 0.5
    else:
        clean_rms = EPS

    return clean_rms, noise_rms


def ActivityDetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    '''Return the percentage of the time the audio signal is above an energy threshold'''

    audio = Normalize(audio, target_level)
    window_size = 50  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win ** 2) + EPS)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > energy_thresh:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def BatchResample(input_dir, target_sr=16000, ext='*.wav'):
    '''Resamples the audio files in input_dir to target_sr'''
    files = glob.glob(f"{input_dir}/" + ext)
    for pathname in files:
        print(pathname)
        try:
            audio, fs = AudioRead(pathname)
            audio_resampled = librosa.core.resample(audio, fs, target_sr)
            AudioWrite(pathname, audio_resampled, target_sr)
        except:
            continue


def AudioSegment(input_dir, dest_dir, segment_len=10, ext='*.wav'):
    '''Segments the audio clips in dir to segment_len in secs'''
    files = glob.glob(f"{input_dir}/" + ext)  # include all '*.wav' in current dir, not include subdirectory
    for i in range(len(files)):
        audio, fs = AudioRead(files[i])

        if len(audio) > (segment_len * fs) and len(audio) % (segment_len * fs) != 0:
            audio = np.append(audio, audio[0: segment_len * fs - (len(audio) % (segment_len * fs))])
        if len(audio) < (segment_len * fs):
            while len(audio) < (segment_len * fs):
                audio = np.append(audio, audio)  # repeat ???
            audio = audio[:segment_len * fs]

        num_segments = int(len(audio) / (segment_len * fs))
        audio_segments = np.split(audio, num_segments)

        basefilename = os.path.basename(files[i])  # no dir
        basename, ext = os.path.splitext(basefilename)

        for j in range(len(audio_segments)):
            newname = basename + '_' + str(j) + ext
            pathname = os.path.join(dest_dir, newname)  # include dir and name
            AudioWrite(pathname, audio_segments[j], fs)


def LimiteWavLength(wav, limite_len):
    wav_len = len(wav)
    if wav_len >= limite_len:
        wav = wav[:limite_len]
    else:
        wav = np.append(wav, np.zeros(limite_len - wav_len))
    return wav
