import os
import constants
import torchaudio
import random
import torch
from functions import transform

def load_audios(path):
    audios = []
    for filename in os.listdir(path):
        channels, rate = torchaudio.load(os.path.join(path, filename))
        resampler = torchaudio.transforms.Resample(rate, constants.RATE)
        sample = resampler(channels[0])
        audios.append(sample)
        
    return audios

def load_random_dataset(path, dataset_size=1000, min_audio_size = 400, max_audio_size = 50000):
    dataset = []
    audios = load_audios(path)

    for i in range(dataset_size):
        audio = random.choice(audios)
        sample_size = random.randint(min_audio_size, max_audio_size)
        start_position = random.randint(0, len(audio) - sample_size)
        res_sample = audio[start_position:start_position + sample_size]
        dataset.append(res_sample)

    return dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        neg_clean_dataset = load_audios(constants.NEG_CLEAN_ROOT)
        neg_noisy_dataset = load_audios(constants.NEG_NOISY_ROOT)
        pos_clean_dataset = load_audios(constants.POS_CLEAN_ROOT)
        pos_noisy_dataset = load_audios(constants.POS_NOISY_ROOT)
        neg_rand_dataset = load_random_dataset(constants.NEG_RAND_ROOT, dataset_size=constants.RANDOM_DATASET_SIZE,
                                                                        min_audio_size=constants.MIN_RANDOM_AUDIO_SIZE,
                                                                        max_audio_size=constants.MAX_RANDOM_AUDIO_SIZE)
        self.neg_clean_dataset = self.to_dataset(neg_clean_dataset, False)
        self.neg_noisy_dataset = self.to_dataset(neg_noisy_dataset, False)
        self.pos_clean_dataset = self.to_dataset(pos_noisy_dataset, True)
        self.pos_noisy_dataset = self.to_dataset(pos_noisy_dataset, True)
        self.neg_rand_dataset = self.to_dataset(neg_rand_dataset, False)

        self.dataset = self.neg_clean_dataset + self.neg_noisy_dataset + self.neg_rand_dataset + \
                       self.pos_clean_dataset + self.pos_noisy_dataset

        train_size = int(len(self.dataset) * constants.TRAIN_DATA_PERCENTAGE)
        test_size = len(self.dataset) - train_size
        self.train_dataloader, self.test_dataloader = torch.utils.data.random_split(self.dataset, [train_size, test_size])

    def to_dataset(self, dataset, is_keyword):
        dataset = [transform(data).t() for data in dataset]
        dataset = [(data, torch.zeros(len(data), dtype=torch.long), 1 if is_keyword else 0) for data in dataset]
        
        if is_keyword:
            for data in dataset:
                n = len(data[1])

                for i in range(n):
                    if i >= (n - constants.KEYWORD_MARKER_NUMBER) and i < (n - 0):
                        data[1][i] = 1

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]