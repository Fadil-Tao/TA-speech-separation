# -*- coding: utf-8 -*-
"""Shared dataset utilities for speech-separation training."""

import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def build_utterance_split(
    raw_dir,
    seed=42,
    train_ratio=0.8,
    dev_ratio=0.1,
):
    """Rebuild the generator's per-speaker utterance split.

    Args:
        raw_dir: Path to the raw TTML-IDN dataset.
        seed: Random seed used by the dataset generator.
        train_ratio: Fraction of each speaker's utterances used for training.
        dev_ratio: Fraction of each speaker's utterances used for validation.

    Returns:
        Three dictionaries mapping speaker IDs to train, dev, and test files.
    """
    speech_dir = Path(raw_dir) / 'Speech'
    if not speech_dir.is_dir():
        raise FileNotFoundError(
            f'Raw speech directory does not exist: {speech_dir}'
        )

    rng = random.Random(seed)
    speakers = {}
    for speaker_dir in sorted(speech_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        audio_files = sorted(speaker_dir.glob('*.wav'))
        if audio_files:
            speakers[speaker_dir.name] = audio_files

    if not speakers:
        raise FileNotFoundError(
            f'No speaker WAV files found under: {speech_dir}'
        )

    train_utts = {}
    dev_utts = {}
    test_utts = {}
    for speaker_id, files in speakers.items():
        shuffled = list(files)
        rng.shuffle(shuffled)
        num_files = len(shuffled)
        num_train = int(num_files * train_ratio)
        num_dev = int(num_files * dev_ratio)
        train_utts[speaker_id] = shuffled[:num_train]
        dev_utts[speaker_id] = shuffled[
            num_train:num_train + num_dev
        ]
        test_utts[speaker_id] = shuffled[num_train + num_dev:]

    return train_utts, dev_utts, test_utts


class DynamicMixDataset(Dataset):
    """Generate fresh N-speaker mixtures from raw training utterances.

    Args:
        utterances_by_speaker: Mapping from speaker IDs to audio paths.
        num_speakers: Number of speakers in each mixture.
        target_duration: Output duration in seconds.
        target_sr: Output sample rate.
        snr_range: Minimum and maximum random SNR in dB.
        epoch_size: Number of mixtures generated per epoch.
        gender_balance: Prefer gender-diverse speaker combinations.
        augment: Apply random speed perturbation when true.
    """

    SPEED_FACTORS = [0.95, 0.975, 1.0, 1.0, 1.025, 1.05]

    def __init__(
        self,
        utterances_by_speaker,
        num_speakers=2,
        target_duration=5.0,
        target_sr=16000,
        snr_range=(-5.0, 5.0),
        epoch_size=28800,
        gender_balance=True,
        augment=False,
    ):
        self.utterances = {
            speaker_id: list(files)
            for speaker_id, files in utterances_by_speaker.items()
            if files
        }
        self.num_speakers = num_speakers
        self.target_duration = target_duration
        self.target_sr = target_sr
        self.snr_range = snr_range
        self.epoch_size = epoch_size
        self.gender_balance = gender_balance
        self.augment = augment
        self.max_offset = target_sr
        self.target_len = int(target_duration * target_sr)

        self.speaker_list = list(self.utterances)
        self.males = [
            speaker_id
            for speaker_id in self.speaker_list
            if speaker_id.startswith('m')
        ]
        self.females = [
            speaker_id
            for speaker_id in self.speaker_list
            if speaker_id.startswith('f')
        ]
        if len(self.speaker_list) < num_speakers:
            raise ValueError(
                f'Need at least {num_speakers} speakers, found '
                f'{len(self.speaker_list)}'
            )

        total_utts = sum(
            len(files) for files in self.utterances.values()
        )
        augment_label = ' (augment ON)' if augment else ''
        print(
            f'[train-dynamic] {len(self.speaker_list)} speakers, '
            f'{total_utts} utterances, {num_speakers}-speaker, '
            f'epoch_size={epoch_size}{augment_label}'
        )

    def __len__(self):
        return self.epoch_size

    def _fit_length(self, audio):
        if len(audio) > self.target_len:
            return audio[:self.target_len]
        if len(audio) < self.target_len:
            return np.pad(audio, (0, self.target_len - len(audio)))
        return audio

    def _load_audio(self, file_path):
        try:
            audio, sample_rate = sf.read(file_path)
        except Exception:
            return None

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != self.target_sr:
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=self.target_sr,
                res_type='polyphase',
            )
        audio, _ = librosa.effects.trim(y=audio, top_db=30)
        if len(audio) < self.target_sr:
            return None

        max_value = np.max(np.abs(audio))
        if max_value > 0:
            audio = audio / max_value * 0.9
        if len(audio) > self.target_len:
            start = random.randint(0, len(audio) - self.target_len)
            audio = audio[start:start + self.target_len]
        else:
            audio = self._fit_length(audio)
        return audio.astype(np.float32)

    def _pick_speakers(self):
        if self.gender_balance and self.males and self.females:
            if self.num_speakers == 2 and random.random() < 0.5:
                return [
                    random.choice(self.males),
                    random.choice(self.females),
                ]
            if self.num_speakers == 3:
                chance = random.random()
                if chance < 0.5 and len(self.males) >= 2:
                    return (
                        random.sample(self.males, 2)
                        + random.sample(self.females, 1)
                    )
                if chance < 0.8 and len(self.females) >= 2:
                    return (
                        random.sample(self.males, 1)
                        + random.sample(self.females, 2)
                    )
        return random.sample(self.speaker_list, self.num_speakers)

    def _mix(self, audios):
        offset_audios = [audios[0]]
        for audio in audios[1:]:
            offset = random.randint(0, self.max_offset)
            offset_audios.append(np.pad(audio, (offset, 0)))
        aligned = [self._fit_length(audio) for audio in offset_audios]

        reference_power = float(np.mean(aligned[0] ** 2)) + 1e-10
        scaled = [aligned[0]]
        for audio in aligned[1:]:
            snr_db = random.uniform(*self.snr_range)
            source_power = float(np.mean(audio ** 2)) + 1e-10
            scale = np.sqrt(
                reference_power
                / (source_power * 10 ** (snr_db / 10))
            )
            scaled.append(audio * scale)

        mixture = sum(scaled)
        max_value = np.max(np.abs(mixture))
        if max_value > 1.0:
            scale = 0.9 / max_value
            mixture *= scale
            scaled = [source * scale for source in scaled]

        return tuple(
            source.astype(np.float32)
            for source in [mixture] + scaled
        )

    def _speed_perturb(self, audio, factor):
        new_sample_rate = int(self.target_sr * factor)
        audio = librosa.resample(
            y=audio,
            orig_sr=self.target_sr,
            target_sr=new_sample_rate,
            res_type='polyphase',
        )
        audio = librosa.resample(
            y=audio,
            orig_sr=new_sample_rate,
            target_sr=self.target_sr,
            res_type='polyphase',
        )
        return self._fit_length(audio)

    def __getitem__(self, idx):
        for _ in range(5):
            speakers = self._pick_speakers()
            audios = [
                self._load_audio(
                    random.choice(self.utterances[speaker_id])
                )
                for speaker_id in speakers
            ]
            if any(audio is None for audio in audios):
                continue

            mixture, *sources = self._mix(audios)
            if self.augment:
                factor = random.choice(self.SPEED_FACTORS)
                if factor != 1.0:
                    mixture = self._speed_perturb(mixture, factor)
                    sources = [
                        self._speed_perturb(source, factor)
                        for source in sources
                    ]

            sample = {
                'mix': torch.FloatTensor(mixture),
                'file_id': f'dyn_{idx}',
            }
            for source_index, source in enumerate(sources, 1):
                sample[f's{source_index}'] = torch.FloatTensor(source)
            return sample

        raise RuntimeError(
            'DynamicMixDataset failed to produce a mixture after 5 retries'
        )


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
