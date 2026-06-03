import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from torch.utils.data import Dataset
import torch

class IndonesianMixDataset(Dataset):
    SPEED_FACTORS = [0.95, 0.975, 1.0, 1.0, 1.025, 1.05]

    def __init__(self, split='train', dataset_dir=None, num_speakers=2, augment=False, sample_rate=16000, target_duration=5.0):
        self.split = split
        self.dataset_dir = Path(dataset_dir)
        self.split_dir = self.dataset_dir / split
        self.num_speakers = num_speakers
        self.augment = augment
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_len = int(target_duration * sample_rate)
        self.mix_files = sorted(list((self.split_dir / 'mix').glob('*.wav')))
        print(f"[{split}] Loaded {len(self.mix_files)} mixtures ({num_speakers}-speaker){(' (augment ON)' if augment else '')}")

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix_file = self.mix_files[idx]
        file_id = mix_file.stem
        mix, sr = sf.read(mix_file)
        sources = [sf.read(self.split_dir / f's{i}' / f'{file_id}.wav')[0] for i in range(1, self.num_speakers + 1)]

        def normalize_length(x):
            if len(x) > self.target_len:
                return x[:self.target_len]
            if len(x) < self.target_len:
                return np.pad(x, (0, self.target_len - len(x)))
            return x
        mix = normalize_length(mix)
        sources = [normalize_length(s) for s in sources]
        if self.augment:
            factor = random.choice(self.SPEED_FACTORS)
            if factor != 1.0:
                new_sr = int(self.sample_rate * factor)
                mix = librosa.resample(y=mix, orig_sr=self.sample_rate, target_sr=new_sr, res_type='polyphase')
                sources = [librosa.resample(y=s, orig_sr=self.sample_rate, target_sr=new_sr, res_type='polyphase') for s in sources]
                mix = librosa.resample(y=mix, orig_sr=new_sr, target_sr=self.sample_rate, res_type='polyphase')
                sources = [librosa.resample(y=s, orig_sr=new_sr, target_sr=self.sample_rate, res_type='polyphase') for s in sources]
                mix = normalize_length(mix)
                sources = [normalize_length(s) for s in sources]
        sample = {'mix': torch.FloatTensor(mix), 'file_id': file_id}
        for i, src in enumerate(sources, 1):
            sample[f's{i}'] = torch.FloatTensor(src)
        return sample
