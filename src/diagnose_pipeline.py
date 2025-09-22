import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
import os

# Import our classes from the training script
from train_biomaestro import BioMaestroConfig, LitBioMaestro, BirdClefDataset

# --- DIAGNOSTIC CHECK 1: LABEL SANITY CHECK ---
# --- DIAGNOSTIC CHECK 1: LABEL SANITY CHECK (DEEP DEBUGGING) ---
def check_pseudo_labels():
    """
    Helps manually verify the quality of the generated pseudo strong labels by saving snippets to a file.
    """
    config = BioMaestroConfig()
    df = pd.read_csv(config.TRAIN_METADATA_PATH)
    
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    print("--- Check 1a & 1b: Manual Verification ---")
    
    samples_to_check = df.sample(50)
    correct_count = 0
    
    for i, (_, row) in enumerate(samples_to_check.iterrows()):
        rec_id = row['recording_id']
        file_path = os.path.join(config.TRAIN_AUDIO_PATH, f"{rec_id}.flac")
        
        try:
            waveform, sr = sf.read(file_path)
        except Exception:
            print(f"🚨 WARNING: Could not read audio file {file_path}. Skipping.")
            continue
            
        start_frame, end_frame = int(row['f_min']), int(row['f_max'])
        
        start_sample = start_frame * config.HOP_LENGTH
        end_sample = (end_frame + 1) * config.HOP_LENGTH
        
        # --- DEEP DEBUGGING PRINTS ---
        print("\n" + "="*40)
        print(f"Processing Sample {i+1}/50")
        print(f"Source File: {rec_id}.flac")
        print(f"Event Frames from CSV: start={start_frame}, end={end_frame}")
        print(f"Calculated Sample Slice: start={start_sample}, end={end_sample}")
        
        # Check if slice is valid before attempting it
        if start_sample >= end_sample:
            print("🚨 ERROR: start_sample is greater than or equal to end_sample. Skipping.")
            continue
        if end_sample > len(waveform):
            print(f"🚨 WARNING: Calculated end_sample ({end_sample}) is beyond waveform length ({len(waveform)}). Clamping.")
            end_sample = len(waveform)
        
        snippet = waveform[start_sample:end_sample]
        snippet_duration_ms = 1000 * len(snippet) / sr
        print(f"Final Snippet Length: {len(snippet)} samples ({snippet_duration_ms:.1f} ms)")
        print("="*40)
        
        if len(snippet) == 0:
            print("🚨 ERROR: Snippet is empty. Cannot save or play. Skipping to next sample.")
            continue

        # Save and get user feedback
        snippet_path = os.path.join(temp_dir, 'current_snippet.wav')
        sf.write(snippet_path, snippet, sr)
        
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        plt.figure(figsize=(10, 4))
        spec_db = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH), ref=np.max)
        librosa.display.specshow(spec_db, sr=sr, hop_length=config.HOP_LENGTH, x_axis='time', y_axis='mel')
        plt.axvspan(start_time, end_time, color='red', alpha=0.4, label='Pseudo-Label')
        plt.title(f"Sample {i+1}/50: {rec_id}\nSpecies: {row['species_id']}")
        plt.legend()
        plt.show(block=False)

        print(f"\nSaved snippet to '{snippet_path}'")
        response = input("Please listen to the file. Does it contain a clear bird call? (y/n): ").lower()
        if response == 'y':
            correct_count += 1
        plt.close()

    # --- (The rest of the function remains the same) ---
    accuracy = (correct_count / 50) * 100
    print(f"\n--- Verification Result ---")
    print(f"✅ Correctly labeled snippets: {correct_count}/50 ({accuracy:.1f}%)")
    if accuracy < 80: print("🚨 Warning: Pseudo-label accuracy is below the 80% threshold.")
    print("\n--- Check 1c: Label Distribution ---")
    total_frames, positive_frames = 0, 0
    for rec_id, group in df.groupby('recording_id'):
        file_path = os.path.join(config.TRAIN_AUDIO_PATH, f"{rec_id}.flac")
        try:
            info = sf.info(file_path)
            total_frames += info.frames // config.HOP_LENGTH
            for _, row in group.iterrows():
                positive_frames += int(row['f_max']) - int(row['f_min'])
        except Exception: continue
    positive_ratio = (positive_frames / total_frames) * 100
    print(f"Total frames in labeled dataset: {total_frames}")
    print(f"Positive (labeled '1') frames: {positive_frames}")
    print(f"Positive frame ratio: {positive_ratio:.2f}%")
    if positive_ratio < 0.5: print("🚨 Warning: Positive frame ratio is very low.")


# --- (The other diagnostic checks remain the same) ---
def check_encoder_collapse(checkpoint_path):
    print("\n--- Check 2: Encoder Collapse ---")
    # ... (code is unchanged) ...
    config = BioMaestroConfig()
    model = LitBioMaestro.load_from_checkpoint(checkpoint_path, config=config, strict=False).to('cuda').eval()
    dataset = BirdClefDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0)
    
    with torch.no_grad():
        x, _, _ = next(iter(dataloader))
        x = x.to('cuda')
        det_logits, _ = model(x)
        preds = torch.sigmoid(det_logits)
        
    print("Output probabilities from one batch:")
    print(f"min={preds.min():.4f}  max={preds.max():.4f}  mean={preds.mean():.4f}  std={preds.std():.4f}")
    if preds.mean().abs() < 1e-4 and preds.std() < 1e-4:
        print("🚨 Diagnosis: Encoder appears to be collapsing to zero.")
    else:
        print("✅ Diagnosis: Encoder is not collapsing.")

def check_gradient_norm(checkpoint_path):
    print("\n--- Check 3: Gradient Norm Inspection ---")
    # ... (code is unchanged) ...
    config = BioMaestroConfig()
    model = LitBioMaestro.load_from_checkpoint(checkpoint_path, config=config, strict=False).to('cuda')
    dataset = BirdClefDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0)
    
    x, cls_target, det_target = next(iter(dataloader))
    x, cls_target, det_target = x.to('cuda'), cls_target.to('cuda'), det_target.to('cuda')
    
    optimizer = model.configure_optimizers()['optimizer']
    optimizer.zero_grad()
    
    det_logits, cls_logits = model(x)
    det_logits_resampled = F.interpolate(det_logits.unsqueeze(1), size=config.IMG_SIZE[1], mode='linear').squeeze(1)
    loss_det = model.detection_loss(det_logits_resampled, det_target)
    loss_cls = model.classification_loss(cls_logits, cls_target) if cls_target.sum() > 0 else 0
    total_loss = loss_det + loss_cls
    
    total_loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    
    print(f"Total gradient norm for one batch: {total_norm.item():.6f}")
    if total_norm.item() < 1e-6:
        print("🚨 Diagnosis: Gradient is vanishingly small. The learning signal is likely being killed.")
    else:
        print("✅ Diagnosis: Gradient norm is healthy.")

# --- MAIN ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run diagnostic checks on the BioMaestro pipeline.")
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint file for checks 2 & 3.')
    parser.add_argument('--check', type=int, required=True, choices=[1, 2, 3], help='Which check to run: 1 (Labels), 2 (Collapse), 3 (Gradient).')
    
    args = parser.parse_args()
    
    if args.check == 1:
        check_pseudo_labels()
    elif args.check == 2:
        if not args.checkpoint: raise ValueError("--checkpoint is required for check 2")
        check_encoder_collapse(args.checkpoint)
    elif args.check == 3:
        if not args.checkpoint: raise ValueError("--checkpoint is required for check 3")
        check_gradient_norm(args.checkpoint)