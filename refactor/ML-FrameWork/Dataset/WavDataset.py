import csv
import os
import pandas as pd

import json5
import torchaudio
from torch.utils.data import Dataset


class WavDataset(Dataset):
    def __init__(self, params):
        """
        dataset_metadata_file(*.csv):
            <noisy_path> <clean_path> ....
        e.g:
            noisy_1.wav clean_1.wav  ....
            noisy_2.wav clean_2.wav  ....
            ...
            noisy_n.wav clean_n.wav  ....
        """
        super(Dataset, self).__init__()
        # self.sr = params['sr']
        metafile = params['metafile']
        self.dataset_list = pd.read_csv(metafile)
        self.length = len(self.dataset_list)
        # self.length = 32

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path = self.dataset_list["noisy"][item]
        clean_path = self.dataset_list["clean"][item]
        noisy, sample_rate = torchaudio.load(noisy_path)
        clean, sample_rate = torchaudio.load(clean_path)
        assert sample_rate == 16000
        assert noisy.shape == clean.shape
        frame_num = (noisy.size(1) - 320) // 160 + 1
        return noisy, clean, frame_num


class TestDataset(Dataset):
    def __init__(self, params):
        super(Dataset, self).__init__()
        test_dir = params['test_dir']
        self.dataset_list = os.listdir(test_dir)
        self.length = len(self.dataset_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy, sr = torchaudio.load(self.dataset_list[item])
        assert sr == 16000
        n_frames = (noisy.size(1) - 320) // 160 + 1
        return noisy, n_frames


if __name__ == "__main__":
    jsonfile = '../config/unittest/WavDataset.json5'
    configuration = json5.load(open(jsonfile))
    dataset = WavDataset(configuration)
    noisy, clean, frame_num = dataset.__getitem__(0)
    print(frame_num)
    print('dones!')
