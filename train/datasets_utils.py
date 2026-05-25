import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from torch.utils.data import Dataset
import torch

def build_utterance_split(raw_dir, seed=42, train_ratio=0.8, dev_ratio=0.1):
    rng = random.Random(seed)
    speech_dir = Path(raw_dir) / 'Speech'
    speakers = {}
    for speaker_dir in sorted(speech_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        audio_files = sorted(speaker_dir.glob('*.wav'))
        if audio_files:
            speakers[speaker_dir.name] = audio_files
    train_utts, dev_utts, test_utts = ({}, {}, {})
    for speaker_id, files in speakers.items():
        shuffled = list(files)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        train_utts[speaker_id] = shuffled[:n_train]
        dev_utts[speaker_id] = shuffled[n_train:n_train + n_dev]
        test_utts[speaker_id] = shuffled[n_train + n_dev:]
    return (train_utts, dev_utts, test_utts)

class DynamicMixDataset(Dataset):
    SPEED_FACTORS = [0.95, 0.975, 1.0, 1.0, 1.025, 1.05]

    def __init__(self, utterances_by_speaker, num_speakers=2, target_duration=5.0, target_sr=16000, snr_range=(-5.0, 5.0), epoch_size=28800, gender_balance=True, augment=False):
        self.utterances = {k: list(v) for k, v in utterances_by_speaker.items() if len(v) > 0}
        self.num_speakers = num_speakers
        self.target_duration = target_duration
        self.target_sr = target_sr
        self.snr_range = snr_range
        self.epoch_size = epoch_size
        self.gender_balance = gender_balance
        self.augment = augment
        self.max_offset = int(1.0 * target_sr)
        self.target_len = int(target_duration * target_sr)
        self.speaker_list = list(self.utterances.keys())
        self.males = [s for s in self.speaker_list if s.startswith('m')]
        self.females = [s for s in self.speaker_list if s.startswith('f')]
        total_utts = sum((len(v) for v in self.utterances.values()))
        print(f"[train-dynamic] {len(self.speaker_list)} speakers, {total_utts} utterances, {num_speakers}-speaker, epoch_size={epoch_size}{(' (augment ON)' if augment else '')}")

    def __len__(self):
        return self.epoch_size

    def _load_audio(self, file_path):
        try:
            audio, sr = sf.read(file_path)
        except Exception:
            return None
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.target_sr, res_type='polyphase')
        audio, _ = librosa.effects.trim(y=audio, top_db=30)
        if len(audio) < self.target_sr:
            return None
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        if len(audio) > self.target_len:
            start = random.randint(0, len(audio) - self.target_len)
            audio = audio[start:start + self.target_len]
        elif len(audio) < self.target_len:
            audio = np.pad(audio, (0, self.target_len - len(audio)))
        return audio.astype(np.float32)

    def _pick_speakers(self):
        if self.gender_balance and self.males and self.females:
            if self.num_speakers == 2:
                if random.random() < 0.5:
                    return [random.choice(self.males), random.choice(self.females)]
            elif self.num_speakers == 3:
                r = random.random()
                if r < 0.5 and len(self.males) >= 2:
                    return random.sample(self.males, 2) + random.sample(self.females, 1)
                if r < 0.8 and len(self.females) >= 2:
                    return random.sample(self.males, 1) + random.sample(self.females, 2)
        return random.sample(self.speaker_list, self.num_speakers)

    def _mix(self, audios):
        offset_audios = [audios[0]]
        for i in range(1, len(audios)):
            offset = random.randint(0, self.max_offset)
            offset_audios.append(np.pad(audios[i], (offset, 0)))

        def fit(x):
            if len(x) > self.target_len:
                return x[:self.target_len]
            if len(x) < self.target_len:
                return np.pad(x, (0, self.target_len - len(x)))
            return x
        aligned = [fit(a) for a in offset_audios]
        p1 = float(np.mean(aligned[0] ** 2)) + 1e-10
        scaled = [aligned[0]]
        for i in range(1, len(aligned)):
            snr_db = random.uniform(*self.snr_range)
            pi = float(np.mean(aligned[i] ** 2)) + 1e-10
            scale = np.sqrt(p1 / (pi * 10 ** (snr_db / 10)))
            scaled.append(aligned[i] * scale)
        mixture = sum(scaled)
        max_val = np.max(np.abs(mixture))
        if max_val > 1.0:
            scale = 0.9 / max_val
            mixture = mixture * scale
            scaled = [s * scale for s in scaled]
        return tuple((s.astype(np.float32) for s in [mixture] + scaled))

    def __getitem__(self, idx):
        for _ in range(5):
            spks = self._pick_speakers()
            audios = [self._load_audio(random.choice(self.utterances[spk])) for spk in spks]
            if any((a is None for a in audios)):
                continue
            result = self._mix(audios)
            mix, *sources = result
            if self.augment:
                factor = random.choice(self.SPEED_FACTORS)
                if factor != 1.0:
                    new_sr = int(self.target_sr * factor)
                    mix = librosa.resample(y=mix, orig_sr=self.target_sr, target_sr=new_sr, res_type='polyphase')
                    mix = librosa.resample(y=mix, orig_sr=new_sr, target_sr=self.target_sr, res_type='polyphase')
                    sources = [librosa.resample(y=s, orig_sr=self.target_sr, target_sr=new_sr, res_type='polyphase') for s in sources]
                    sources = [librosa.resample(y=s, orig_sr=new_sr, target_sr=self.target_sr, res_type='polyphase') for s in sources]
                    mix = mix[:self.target_len] if len(mix) > self.target_len else np.pad(mix, (0, self.target_len - len(mix)))
                    sources = [s[:self.target_len] if len(s) > self.target_len else np.pad(s, (0, self.target_len - len(s))) for s in sources]
            sample = {'mix': torch.FloatTensor(mix), 'file_id': f'dyn_{idx}'}
            for i, src in enumerate(sources, 1):
                sample[f's{i}'] = torch.FloatTensor(src)
            return sample
        raise RuntimeError('DynamicMixDataset: failed to produce mixture after 5 retries')

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
