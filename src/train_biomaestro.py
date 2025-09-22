import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import pandas as pd
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# --- Configuration, Loss Functions, and Dataset are the same ---
# ... (all code from before the LitBioMaestro class) ...
# --- 1. Configuration ---
class BioMaestroConfig:
    TRAIN_METADATA_PATH = 'data/rfcx-species-audio-detection/train_tp_strong_labels.csv'
    TRAIN_AUDIO_PATH = 'data/rfcx-species-audio-detection/train/'
    SAMPLE_RATE = 16000
    DURATION = 8
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 160
    IMG_SIZE = (N_MELS, 800)
    MODEL_NAME = 'vit_base_patch16_224'
    PATCH_SIZE = 16
    NUM_CLASSES = 24
    BATCH_SIZE = 16
    LEARNING_RATE = 1.5e-4
    EPOCHS = 30
    NUM_WORKERS = 0

# --- 2. Loss Functions ---
class BetaBalancedBCELoss(nn.Module):
    def __init__(self, beta=0.99, gamma_ema=0.99):
        super().__init__()
        self.beta = beta
        self.gamma_ema = gamma_ema
        self.register_buffer('gamma', torch.tensor(0.5))
    def forward(self, input, target):
        gamma_current = target.mean()
        self.gamma = self.gamma_ema * self.gamma + (1 - self.gamma_ema) * gamma_current
        w_pos = (1 - self.beta) / (1 - self.gamma + 1e-6)
        w_neg = self.beta / (self.gamma + 1e-6)
        pos_weight = torch.full_like(input, w_neg)
        pos_weight[target > 0.5] = w_pos
        loss = F.binary_cross_entropy_with_logits(input, target, weight=pos_weight)
        return loss

def focal_asymmetric_loss(y_pred, y_true, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    p = torch.sigmoid(y_pred)
    pt = torch.where(y_true == 1, p, 1 - p)
    focal_weight = (1 - pt)**gamma
    asymmetric_weight = torch.where(y_true == 1, 1.0, 0.5)
    return (focal_weight * asymmetric_weight * bce).mean()

# --- 3. Dataset ---
class BirdClefDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(config.TRAIN_METADATA_PATH)
        self.recording_groups = self.df.groupby('recording_id')
        self.recordings = list(self.recording_groups.groups.keys())
        
        self.species_map = {species: i for i, species in enumerate(sorted(self.df['species_id'].unique()))}
        if config.NUM_CLASSES != len(self.species_map): self.config.NUM_CLASSES = len(self.species_map)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH, n_mels=config.N_MELS, power=2.0)
        self.db_transform = T.AmplitudeToDB()
        self.augmenter = Compose([AddGaussianNoise(p=0.5), TimeStretch(p=0.5), PitchShift(p=0.5)])

        # --- Load Noise Files ---
        self.noise_path = 'data/noise_profiles/'
        self.noise_files = []
        if os.path.exists(self.noise_path):
            self.noise_files = [os.path.join(self.noise_path, f) for f in os.listdir(self.noise_path) if f.endswith('.wav')]
        if self.noise_files:
            print(f"Found {len(self.noise_files)} noise profiles for synthetic negatives.")

    def __len__(self):
        # We increase the dataset size to account for the synthetic negatives
        return len(self.recordings) + (len(self.recordings) // 4)

    def __getitem__(self, idx):
        # --- SYNTHETIC NEGATIVE SAMPLING ---
        # 20% of the time (1 / (4+1)), we'll generate a pure noise sample
        if idx >= len(self.recordings) and self.noise_files:
            return self.get_noise_sample()

        rec_id = self.recordings[idx]
        file_path = os.path.join(self.config.TRAIN_AUDIO_PATH, f"{rec_id}.flac")
        
        try: 
            waveform, sr = sf.read(file_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
        except Exception: return self.__getitem__(np.random.randint(len(self)))

        # (The rest of the logic for regular samples is the same as before)
        target_samples = self.config.DURATION * sr; start_sample = 0
        if len(waveform) > target_samples: start_sample = np.random.randint(0, len(waveform) - target_samples)
        chunk = waveform[start_sample : start_sample + target_samples]
        if len(chunk) < target_samples: chunk = np.pad(chunk, (0, target_samples - len(chunk)))
        chunk = self.augmenter(samples=chunk, sample_rate=sr)
        log_mel_spec = self.preprocess_chunk(chunk)
        cls_target = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)
        det_target = torch.zeros(self.config.IMG_SIZE[1], dtype=torch.float32)
        events_in_file = self.recording_groups.get_group(rec_id)
        for _, event in events_in_file.iterrows():
            species_id = self.species_map.get(event['species_id']);
            if species_id is None: continue
            start_frame_abs, end_frame_abs = int(event['f_min']), int(event['f_max'])
            start_frame_chunk = int(start_sample / self.config.HOP_LENGTH)
            start_frame_relative, end_frame_relative = start_frame_abs - start_frame_chunk, end_frame_abs - start_frame_chunk
            if start_frame_relative < self.config.IMG_SIZE[1] and end_frame_relative > 0:
                cls_target[species_id] = 1.0
                det_start, det_end = max(0, start_frame_relative), min(self.config.IMG_SIZE[1], end_frame_relative)
                det_target[det_start:det_end] = 1.0
        return log_mel_spec, cls_target, det_target

    def get_noise_sample(self):
        """Creates a training sample from a pure noise file."""
        noise_filepath = np.random.choice(self.noise_files)
        waveform, sr = sf.read(noise_filepath)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        
        target_samples = self.config.DURATION * sr
        start_sample = 0
        if len(waveform) > target_samples:
            start_sample = np.random.randint(0, len(waveform) - target_samples)
        
        chunk = waveform[start_sample : start_sample + target_samples]
        if len(chunk) < target_samples:
             chunk = np.pad(chunk, (0, target_samples - len(chunk)))

        log_mel_spec = self.preprocess_chunk(chunk)
        
        # The labels for noise are all zeros
        cls_target = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)
        det_target = torch.zeros(self.config.IMG_SIZE[1], dtype=torch.float32)
        
        return log_mel_spec, cls_target, det_target

    def preprocess_chunk(self, waveform_chunk):
        waveform = torch.tensor(waveform_chunk, dtype=torch.float32); mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.db_transform(mel_spec + 1e-6)
        std = log_mel_spec.std()
        if std > 1e-6: log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / std
        else: log_mel_spec = torch.zeros_like(log_mel_spec)
        target_len = self.config.IMG_SIZE[1]; current_len = log_mel_spec.shape[1]
        if current_len > target_len: log_mel_spec = log_mel_spec[:, :target_len]
        else: log_mel_spec = torch.cat([log_mel_spec, torch.zeros(self.config.N_MELS, target_len - current_len)], dim=1)
        return log_mel_spec
    
# --- 4. Gradual Unfreezing Callback ---
class GradualUnfreezing(Callback):
    def __init__(self, unfreeze_epoch=10, num_layers_to_unfreeze=2):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.num_layers_to_unfreeze = num_layers_to_unfreeze

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            print(f"\nEpoch {self.unfreeze_epoch}: Unfreezing last {self.num_layers_to_unfreeze} encoder blocks.")
            # The encoder blocks are in a ModuleList
            num_blocks = len(pl_module.encoder.blocks)
            for i in range(num_blocks - self.num_layers_to_unfreeze, num_blocks):
                for param in pl_module.encoder.blocks[i].parameters():
                    param.requires_grad = True

# --- 5. BioMaestro Model (using your corrected forward pass) ---
class LitBioMaestro(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.encoder = timm.create_model(config.MODEL_NAME, pretrained=True, in_chans=3, img_size=config.IMG_SIZE)
        conv1_weight = self.encoder.patch_embed.proj.weight
        self.encoder.patch_embed.proj = nn.Conv2d(1, self.encoder.embed_dim, kernel_size=config.PATCH_SIZE, stride=config.PATCH_SIZE)
        self.encoder.patch_embed.proj.weight.data = conv1_weight.mean(dim=1, keepdim=True)
        print(">> Encoder loaded with ImageNet weights.")

        # --- FREEZE THE ENCODER INITIALLY ---
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(">> Encoder frozen initially.")

        encoder_dim = self.encoder.embed_dim
        self.detection_head = nn.Sequential(nn.Conv1d(encoder_dim, 128, 3, 1, padding=1), nn.ReLU(), nn.Conv1d(128, 1, 1))
        self.attention = nn.Sequential(nn.Linear(encoder_dim, 1), nn.Softmax(dim=1))
        self.classification_head = nn.Sequential(nn.Linear(encoder_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, config.NUM_CLASSES))
        self.detection_loss = BetaBalancedBCELoss()
        self.classification_loss = focal_asymmetric_loss

    def forward(self, x):
        B = x.shape[0]
        x = self.encoder.patch_embed(x.unsqueeze(1))
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.encoder.pos_embed
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        patch_tokens_out = x[:, 1:]
        
        det_input = patch_tokens_out.permute(0, 2, 1)
        det_logits = self.detection_head(det_input).squeeze(1)
        
        gate = (torch.sigmoid(det_logits) > 0.5).float().unsqueeze(-1)
        if gate.sum() > 0: gated_tokens = patch_tokens_out * gate
        else: gated_tokens = patch_tokens_out
        attn_weights = self.attention(gated_tokens)
        pooled_output = torch.sum(gated_tokens * attn_weights, dim=1)
        cls_logits = self.classification_head(pooled_output)
        return det_logits, cls_logits

    def training_step(self, batch, batch_idx):
        x, cls_target, det_target = batch
        det_logits, cls_logits = self(x)
        det_logits_resampled = F.interpolate(det_logits.unsqueeze(1), size=self.config.IMG_SIZE[1], mode='linear').squeeze(1)
        loss_det = self.detection_loss(det_logits_resampled, det_target)
        if cls_target.sum() > 0: loss_cls = self.classification_loss(cls_logits, cls_target)
        else: loss_cls = torch.tensor(0.0, device=self.device)
        total_loss = loss_det + loss_cls
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.LEARNING_RATE)
        # Warmup scheduler logic
        try: train_loader_len = len(self.trainer.train_dataloader)
        except (AttributeError, TypeError):
            dataset_size = len(BirdClefDataset(self.config))
            train_loader_len = dataset_size // self.config.BATCH_SIZE
            if dataset_size % self.config.BATCH_SIZE != 0: train_loader_len += 1
        num_training_steps = train_loader_len * self.config.EPOCHS; num_warmup_steps = train_loader_len
        def lr_lambda(current_step):
            if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# --- Main Execution Block ---
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    config = BioMaestroConfig()
    dataset = BirdClefDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
    model = LitBioMaestro(config)
    
    checkpoint_callback = ModelCheckpoint(dirpath='models/biomaestro_final_run', filename='biomaestro-final-{epoch:02d}', save_top_k=1, monitor='train_loss', mode='min', save_last=True)
    
    # Add our new callback to the trainer
    unfreeze_callback = GradualUnfreezing(unfreeze_epoch=10)
    
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS, accelerator='gpu', devices=1,
        callbacks=[checkpoint_callback, unfreeze_callback], # Pass both callbacks
        gradient_clip_val=1.0
    )
    
    print(">> Starting FINAL BioMaestro run with strong labels and gradual unfreezing...")
    trainer.fit(model, dataloader)
    print(">> BioMaestro fine-tuning finished!")