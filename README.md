# TA-speech-separation

Indonesian speech separation with **SkiM** (Skipping Memory LSTM) and **SkiM-Attention** (SkiM with Multi-Head Self-Attention replacing the inter-segment memory). Built on ESPnet. Trained on synthetic mixtures from the TITML-IDN corpus.

Encoder/decoder: Conv1D, `kernel_size=16`, `stride=8` (1 ms window, 0.5 ms hop at 16 kHz).

## Layout

```
implementation/   model code (SkiM, SkiM-Attention, ESPnet separator wrappers)
train/            training scripts (2-speaker, 3-speaker, transfer)
eval/             test-set SI-SNR / SI-SNRi evaluation
dataset/          synthetic mixture generators
utils/            path resolution helpers
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchaudio espnet==202304 soundfile librosa tqdm matplotlib mir_eval pesq pystoi gdown
```

GPU required for training.

## Paths

All scripts resolve paths in this order: CLI flag > env var > project default.

```bash
export TSS_RAW_DIR=/path/to/TTML-IDN
export TSS_SYNTHETIC_DIR=/path/to/synthetic
export TSS_CHECKPOINT_DIR=/path/to/checkpoints
```

## Generate dataset

Source: TITML-IDN (20 speakers, 16 kHz). Output: 5-second mixtures, SNR -5 to +5 dB, 28800/3600/3600 train/dev/test.

```bash
python dataset/generator/titml_mix_generator_2spk.py
python dataset/generator/titml_mix_generator_3spk.py
```

## Train

```bash
python train/2speaker/skim/train_skim_2spk.py
python train/2speaker/skim-attention/train_skim_attention_2spk.py
python train/3speaker/skim/train_skim_3spk.py
python train/3speaker/skim-attention/train_skim_attention_3spk.py
```

Transfer learning (2-speaker checkpoint → 3-speaker):

```bash
python train/3speaker/skim/train_skim_3spk_transfer.py
python train/3speaker/skim-attention/train_skim_attention_3spk_transfer.py
```

Resume:

```bash
python train/2speaker/skim/train_skim_2spk.py --resume-from checkpoint_epoch_30.pth
```

Checkpoints, `training_curves.png`, `training_history.json`, and `config.json` are written to `$TSS_CHECKPOINT_DIR/{2,3}speaker/<model>/`.

## Evaluate

Computes SI-SNR and SI-SNRi on the full test set; saves per-file CSV and a JSON summary.

```bash
python eval/run_eval.py
python eval/run_eval.py --models 2speaker-skim 3speaker-skim-attention
```

## Loss and training

SI-SNR loss with Permutation Invariant Training (PIT). Adam + `ReduceLROnPlateau` (factor 0.5, patience 5). Gradient clip 5.0. Mixed-precision (AMP). Best checkpoint by validation loss; periodic checkpoints every 10 epochs.
