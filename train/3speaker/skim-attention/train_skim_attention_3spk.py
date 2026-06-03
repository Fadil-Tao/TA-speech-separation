import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'train'))
from datasets_utils import IndonesianMixDataset
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from implementation.conv_encoder_abs import ConvEncoderAbs
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from implementation.skim_attention.skim_attention_separator import SkiMAttentionSeparator
MODEL_CONFIG = {'encoder': {'channel': 256, 'kernel_size': 16, 'stride': 8}, 'decoder': {'channel': 256, 'kernel_size': 16, 'stride': 8}, 'separator': {'input_dim': 256, 'causal': False, 'num_spk': 3, 'predict_noise': False, 'nonlinear': 'relu', 'layer': 4, 'unit': 256, 'segment_size': 150, 'dropout': 0.2, 'mem_type': 'hc', 'seg_overlap': False, 'num_heads': 4}}
TRAIN_CONFIG = {'batch_size': 8, 'num_epochs': 100, 'learning_rate': 0.001, 'weight_decay': 1e-05, 'gradient_clip': 5.0, 'seed': 42}
DATASET_DIR = project_root / 'dataset' / 'synthetic' / 'TITML-3spk-v2'
CHECKPOINT_DIR = project_root / 'checkpoints' / '3speaker' / 'skim-attention'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def build_model(device):
    print('\n' + '=' * 60)
    print('Building SkiM Attention 3-Speaker Model')
    print('=' * 60)
    encoder = ConvEncoder(channel=MODEL_CONFIG['encoder']['channel'], kernel_size=MODEL_CONFIG['encoder']['kernel_size'], stride=MODEL_CONFIG['encoder']['stride'])
    print(f"✓ Encoder: Conv1D ({MODEL_CONFIG['encoder']['channel']} channels)")
    separator = SkiMAttentionSeparator(input_dim=MODEL_CONFIG['separator']['input_dim'], causal=MODEL_CONFIG['separator']['causal'], num_spk=MODEL_CONFIG['separator']['num_spk'], predict_noise=MODEL_CONFIG['separator']['predict_noise'], nonlinear=MODEL_CONFIG['separator']['nonlinear'], layer=MODEL_CONFIG['separator']['layer'], unit=MODEL_CONFIG['separator']['unit'], segment_size=MODEL_CONFIG['separator']['segment_size'], dropout=MODEL_CONFIG['separator']['dropout'], num_heads=MODEL_CONFIG['separator']['num_heads'], mem_type=MODEL_CONFIG['separator']['mem_type'], seg_overlap=MODEL_CONFIG['separator']['seg_overlap'])
    print(f"✓ Separator: SkiM Attention ({MODEL_CONFIG['separator']['layer']} layers, {MODEL_CONFIG['separator']['unit']} units, {MODEL_CONFIG['separator']['num_heads']} heads, {MODEL_CONFIG['separator']['num_spk']} speakers)")
    decoder = ConvDecoder(channel=MODEL_CONFIG['decoder']['channel'], kernel_size=MODEL_CONFIG['decoder']['kernel_size'], stride=MODEL_CONFIG['decoder']['stride'])
    print(f'✓ Decoder: ConvTranspose1D')
    criterion = SISNRLoss()
    pit_wrapper = PITSolver(criterion=criterion)
    print(f'✓ Loss: SI-SNR with PIT')
    model = ESPnetEnhancementModel(encoder=encoder, separator=separator, decoder=decoder, mask_module=None, loss_wrappers=[pit_wrapper], loss_type='si_snr')
    model = model.to(device)
    num_params = sum((p.numel() for p in model.parameters()))
    print(f'\nModel parameters: {num_params:,}')
    return model

def save_training_history(train_losses, val_losses, out_dir):
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

def load_training_history(out_dir):
    history_path = out_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path) as f:
            data = json.load(f)
        return (data.get('train_losses', []), data.get('val_losses', []))
    return ([], [])

def save_training_curves(train_losses, val_losses, out_dir):
    if not train_losses:
        return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(val_losses, label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative SI-SNR)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot([-l for l in train_losses], label='Train SI-SNR', marker='o', markersize=3)
    plt.plot([-l for l in val_losses], label='Val SI-SNR', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('SI-SNR (dB)')
    plt.title('Training and Validation SI-SNR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=150)
    plt.close()

def train_epoch(model, train_loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        mix = batch['mix'].to(device)
        s1 = batch['s1'].to(device)
        s2 = batch['s2'].to(device)
        s3 = batch['s3'].to(device)
        batch_size = mix.size(0)
        mix_lengths = torch.full((batch_size,), mix.size(1), dtype=torch.long, device=device)
        speech_ref1 = s1
        speech_ref2 = s2
        speech_ref3 = s3
        ref_lengths = mix_lengths.clone()
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            loss, stats, weight = model(speech_mix=mix, speech_mix_lengths=mix_lengths, speech_ref1=speech_ref1, speech_ref1_lengths=ref_lengths, speech_ref2=speech_ref2, speech_ref2_lengths=ref_lengths, speech_ref3=speech_ref3, speech_ref3_lengths=ref_lengths)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f'\n⚠️ Warning: NaN/Inf loss detected at batch {batch_idx}, skipping...')
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        si_snr_db = -loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'SI-SNR': f'{si_snr_db:.2f} dB'})
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, val_loader, device, epoch):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            mix = batch['mix'].to(device)
            s1 = batch['s1'].to(device)
            s2 = batch['s2'].to(device)
            s3 = batch['s3'].to(device)
            batch_size = mix.size(0)
            mix_lengths = torch.full((batch_size,), mix.size(1), dtype=torch.long, device=device)
            speech_ref1 = s1
            speech_ref2 = s2
            speech_ref3 = s3
            ref_lengths = mix_lengths.clone()
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                loss, stats, weight = model(speech_mix=mix, speech_mix_lengths=mix_lengths, speech_ref1=speech_ref1, speech_ref1_lengths=ref_lengths, speech_ref2=speech_ref2, speech_ref2_lengths=ref_lengths, speech_ref3=speech_ref3, speech_ref3_lengths=ref_lengths)
            total_loss += loss.item()
            val_si_snr_db = -loss.item()
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'val_SI-SNR': f'{val_si_snr_db:.2f} dB'})
    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    random.seed(TRAIN_CONFIG['seed'])
    np.random.seed(TRAIN_CONFIG['seed'])
    torch.manual_seed(TRAIN_CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('\nLoading datasets...')
    train_dataset = IndonesianMixDataset(split='train', dataset_dir=DATASET_DIR, num_speakers=3, augment=False, target_duration=5.0)
    dev_dataset = IndonesianMixDataset(split='dev', dataset_dir=DATASET_DIR, num_speakers=3, augment=False, target_duration=5.0)
    test_dataset = IndonesianMixDataset(split='test', dataset_dir=DATASET_DIR, num_speakers=3, augment=False, target_duration=5.0)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    print(f'✓ Train batches: {len(train_loader)}')
    print(f'✓ Dev batches: {len(dev_loader)}')
    print(f'✓ Test batches: {len(test_loader)}')
    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=TRAIN_CONFIG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-06, verbose=True)
    best_val_loss = float('inf')
    train_losses, val_losses = load_training_history(CHECKPOINT_DIR)
    print('\n' + '=' * 60)
    print('Starting Training')
    print('=' * 60)
    try:
        for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
            train_losses.append(train_loss)
            val_loss = validate(model, dev_loader, device, epoch)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val SI-SNR = {-val_loss:.2f} dB')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'train_loss': train_loss, 'val_loss': val_loss, 'best_val_loss': best_val_loss, 'config': MODEL_CONFIG, 'train_losses': train_losses, 'val_losses': val_losses}, CHECKPOINT_DIR / 'best_model.pth')
                print(f'  ✓ Best model saved (SI-SNR: {-val_loss:.2f} dB)')
            if epoch % 10 == 0:
                checkpoint_path = CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'train_loss': train_loss, 'val_loss': val_loss, 'train_losses': train_losses, 'val_losses': val_losses}, checkpoint_path)
                print(f'  ✓ Checkpoint saved: epoch_{epoch}.pth')
            save_training_curves(train_losses, val_losses, CHECKPOINT_DIR)
            save_training_history(train_losses, val_losses, CHECKPOINT_DIR)
    except KeyboardInterrupt:
        print('\n⚠️ Training interrupted by user')
        if train_losses:
            interrupted_path = CHECKPOINT_DIR / 'checkpoint_interrupted.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'train_loss': train_losses[-1], 'val_loss': val_losses[-1] if val_losses else float('inf'), 'best_val_loss': best_val_loss, 'train_losses': train_losses, 'val_losses': val_losses}, interrupted_path)
            print(f'  ✓ Interrupted checkpoint saved: checkpoint_interrupted.pth (epoch {epoch})')
    finally:
        if train_losses:
            print('\n' + '=' * 60)
            print(f'Best validation SI-SNR: {-best_val_loss:.2f} dB')
            save_training_curves(train_losses, val_losses, CHECKPOINT_DIR)
            print(f'✓ Training curves saved')
            with open(CHECKPOINT_DIR / 'config.json', 'w') as f:
                json.dump({'model_config': MODEL_CONFIG, 'train_config': TRAIN_CONFIG, 'best_val_loss': best_val_loss, 'best_si_snr': -best_val_loss}, f, indent=2)
            print(f'✓ Config saved')
if __name__ == '__main__':
    main()
