
import csv
import os
import random
from pathlib import Path
import json5
import numpy as np
from Utility.AudioLib import AudioRead, LimiteWavLength, SnrMixer, AudioWrite
from Utility.FileOperation import MakeEmptyDir

import torch

project_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def GetUniformRandom(min_value, max_value,
                  size):  # if gaussian 69.27%->(u-std, u+std), 95%->(u-2*std, u+2*std), 99%->(u-3*std, u+3*std)
    mean = (max_value + min_value) / 2
    range = abs(max_value - min_value)
    uniform_random = (np.random.rand(size) - 0.5) * range + mean  # rand():0~1
    return uniform_random

def GetFileList(dir,suffix): # get all files has suffix, include subdirectories
    file_list = []
    for root, dirs, files in os.walk(dir): # files is all the files in this root directory
        for file in files:
            if file.endswith(suffix):
                temp_file_dict = {'dir': [], 'name': []} # must create here, otherwise the previous members in file_list will be modified
                temp_file_dict['dir'] = root
                temp_file_dict['name'] = file
                file_list.append(temp_file_dict)
    return file_list


'''
def CalcCleanSpeechActiveRMS(clean, sample_rate=48000, noise_thr=1e-5):

    frame_size = int(sample_rate / 100)
    frame_num = int(len(clean)/fframe_numrame_size)
    assert frame_size* == len(clean)

    vad_flag = np.zeros((frame_num,1))
    vad_history = np.zeros((3,1))
    vad = np.zeros(len(clean),1)

    clean_frame = np.reshape(clean, [frame_num,frame_size])
    clean_frame_rms = np.mean(clean_frame**2, axis=1) # by col
    clean_active_rms = np.zeros(0,0)

    rms_his_buf = clean_frame_rms[: 15]
    noise_level = np.mean(rms_his_buf)
    noise_global_min = 0
    noise_local_min = 0
    track_period = 100


    for ii in range(15,frame_num):
        current_frame_rms = clean_frame_rms[ii]
        avg_his_rms = np.mean(rms_his_buf)
        if np.mod(ii, track_period) == 0:
            noise_global_min = np.min(noise_local_min, current_frame_rms)
            noise_local_min = current_frame_rms
        else:
            noise_global_min = np.min(noise_global_min, current_frame_rms)
            noise_local_min = np.min(noise_local_min,current_frame_rms)

        noise_level = np.max(noise_global_min,noise_thr)

        if current_frame_rms > 3*noise_level :
            vad_flag[ii] = 1
        else:
            vad_flag[ii] = 0

        # filter vad_flag
        vad_history_temp = vad_history
        vad_history[: 2] = vad_history_temp[1:]
        vad_history[-1] = vad_flag[ii]
        if np.sum(vad_history) != 3:
            if vad_flag[ii] !=0 :
                vad_flag[ii] = 0
        if np.sum(vad_history) != 0:
            if vad_flag[ii] == 0:
                vad_flag[ii] = 1

        if vad_flag[ii] != 0:
            vad[frame_size*ii : frame_size*(ii+1)] = 1
            clean_active_rms = np.append(clean_active_rms,clean_frame_rms[ii])

    if len(clean_active_rms) > 0:
        clean_rms = np.sqrt(np.mean(clean_active_rms))
    else:
        clean_rms = sys.float_info.epsilon

    # add vad time stamp of segments
    voice_active_timestamp = np.zeros((0,2))
    voice_state = 0
    voice_start = 0
    voice_end = 0
    for ii in range(1,frame_num):
        if not voice_state:
            if vad_flag[ii-1] == 0 and vad_flag[ii] == 1:
                voice_start = (ii-1)*frame_size/sample_rate
                voice_state = 1
        else:
            if vad_flag[ii-1] == 1 and vad_flag[ii] == 0:
                voice_end = (ii-1)*frame_size/sample_rate
                voice_active_timestamp = np.append(voice_active_timestamp,[voice_start,voice_end],axis=0)
                voice_state = 0
            elif ii+1 == frame_num:
                voice_end = ii * frame_size / sample_rate
                voice_active_timestamp = np.append(voice_active_timestamp, [voice_start, voice_end], axis=0)
                voice_state = 0

    return clean_rms, voice_active_timestamp
'''


def GenerateBatchNoisySignalFiles(params):
    noise_dir = params['noise_dir']
    clean_dir = params['clean_dir']
    train_dir = Path(params['train_dir']).expanduser().absolute()
    vail_dir = Path(params['vail_dir']).expanduser().absolute()
    train_clean_dir = train_dir / "clean"
    train_noisy_dir = train_dir / "noisy"
    train_metafile = train_dir / "train_metadata.csv"
    vail_clean_dir = vail_dir / "clean"
    vail_noisy_dir = vail_dir / "noisy"
    vail_metafile = vail_dir / "vail_metadata.csv"
    MakeEmptyDir(train_clean_dir)
    MakeEmptyDir(train_noisy_dir)
    MakeEmptyDir(vail_clean_dir)
    MakeEmptyDir(vail_noisy_dir)
    wav_len = params['mix_params']['wav_len']
    target_level = params['mix_params']['target_level']
    clipping_threshold = params['mix_params']['clipping_threshold']
    vail_rate = params['mix_params']['vail_rate']

    with open(train_metafile, 'w', newline='') as csvfile:
        field_name = ['index', 'noisy', 'clean', 'noise', 'clean_rms', 'clean_type', 'noise_rms',
                      'noise_type', 'noisy_rms', 'SNR', 'is_clip']
        writer = csv.DictWriter(csvfile, fieldnames=field_name)
        writer.writeheader()

    with open(vail_metafile, 'w', newline='') as csvfile:
        field_name = ['index', 'noisy', 'clean', 'noise', 'clean_rms', 'clean_type', 'noise_rms',
                      'noise_type', 'noisy_rms', 'SNR', 'is_clip']
        writer = csv.DictWriter(csvfile, fieldnames=field_name)
        writer.writeheader()

    noise_files = os.listdir(noise_dir)
    clean_files = os.listdir(clean_dir)
    random.shuffle(noise_files)
    noise_len = len(noise_files)
    clean_len = len(clean_files)
    if noise_len > clean_len:
        wlen = clean_len
    else:
        wlen = noise_len
    train_len = int(wlen*(1 - vail_rate))
    vail_len = wlen - train_len

    for i in range(train_len):
        current_noise_file = noise_files[i]
        current_clean_file = clean_files[i]
        GenerateNoisySignal(params, noise_dir, clean_dir, i, current_noise_file, current_clean_file, train_noisy_dir, train_clean_dir,
                      target_level, wav_len, clipping_threshold, train_metafile)

    for i in range(vail_len):
        current_noise_file = noise_files[i+train_len]
        current_clean_file = clean_files[i+train_len]
        GenerateNoisySignal(params, noise_dir, clean_dir, i, current_noise_file, current_clean_file, vail_noisy_dir, vail_clean_dir,
                      target_level, wav_len, clipping_threshold, vail_metafile)


def GenerateNoisySignal(params, noise_dir, clean_dir, file_index, noise_name, clean_name, noisy_out_dir, clean_out_dir, target_level,
                  wav_len, clipping_threshold, metafile):
    print(file_index, noise_name, clean_name)
    noise_file = os.path.join(noise_dir, noise_name)
    clean_file = os.path.join(clean_dir, clean_name)
    noisy_path = os.path.join(noisy_out_dir, 'noisy_' + str(file_index) + '.wav')
    new_clean_path = os.path.join(clean_out_dir, clean_name)
    clean, fs = AudioRead(clean_file, norm=True, target_level=target_level)
    noise, fs = AudioRead(noise_file, norm=True, target_level=target_level)
    clean = LimiteWavLength(clean, wav_len)  # limit to ten second
    noise = LimiteWavLength(noise, wav_len)
    cfg_param = params['mix_params']
    snr = np.random.randint(params['mix_params']['snr_lower'], params['mix_params']['snr_upper'])
    clean_new, noisenewlevel, noisy_speech, noisy_rms_level, is_clip = SnrMixer(cfg_param, clean, noise, snr,
                                                                                clipping_threshold=clipping_threshold)
    rmsclean_new = int(20 * np.log10((clean_new ** 2).mean() ** 0.5))
    rmsnoise_new = int(20 * np.log10((noisenewlevel ** 2).mean() ** 0.5))
    AudioWrite(noisy_path, noisy_speech)
    AudioWrite(new_clean_path, clean_new)
    speech_type = 'ReadBook'
    noise_type = 'Steady'
    with open(metafile, 'a', newline='') as csvfile:
        fieldnames = ['index', 'noisy', 'clean', 'noise', 'clean_rms', 'clean_type', 'noise_rms',
                      'noise_type', 'noisy_rms', 'SNR', 'is_clip']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'index': file_index, 'noisy': noisy_path, 'clean': new_clean_path, 'noise': noise_file,
                         'clean_rms': rmsclean_new, 'clean_type': speech_type,
                         'noise_rms': rmsnoise_new, 'noise_type': noise_type,
                         'noisy_rms': noisy_rms_level,
                         'SNR': snr, 'is_clip': is_clip})


if __name__ == '__main__':
    a = np.random.rand(3,4)
    b = a[:,:-1]
    snr = GetUniformRandom(5, 10, 8)
    wav = GetFileList("D:\AudioProcessing\SpeechAlgorithms-master\AudioFingerPrinting",".mp3")

    json_file = '../Config/unittest/mixsp_wme.json5'
    configuration = json5.load(open(json_file))
    GenerateBatchNoisySignalFiles(configuration)

    a = np.random.randint(1,10,size=(3,3))
    b = np.sum(a,axis=-2)
    print('PyCharm is running.')
