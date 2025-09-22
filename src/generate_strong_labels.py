import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from scipy.signal import medfilt
from tqdm import tqdm

# --- Configuration ---
class LabelGenConfig:
    TRAIN_AUDIO_PATH = 'data/rfcx-species-audio-detection/train/'
    INPUT_METADATA_PATH = 'data/rfcx-species-audio-detection/train_tp.csv'
    OUTPUT_METADATA_PATH = 'data/rfcx-species-audio-detection/train_tp_strong_labels.csv'
    
    SAMPLE_RATE = 16000
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 160
    
    ENERGY_THRESHOLD_STD_MULTIPLIER = 2.0
    MEDIAN_FILTER_SIZE = 5

def generate_labels():
    """
    Generates a new CSV with pseudo strong labels (start and end frames)
    for sound events based on mel-spectrogram energy.
    """
    config = LabelGenConfig()
    df = pd.read_csv(config.INPUT_METADATA_PATH)
    
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_mels=config.N_MELS, power=2.0
    )

    all_strong_labels = []
    unique_recordings = df['recording_id'].unique()
    
    print(f"Generating pseudo strong labels for {len(unique_recordings)} recordings...")
    
    skipped_count = 0
    # Use a variable for the progress bar to update its description
    progress_bar = tqdm(unique_recordings, desc="Processing files", unit="file")
    
    for rec_id in progress_bar:
        # Update the postfix to show the current file and skipped count
        progress_bar.set_postfix(file=rec_id, skipped=skipped_count)
        
        file_path = os.path.join(config.TRAIN_AUDIO_PATH, f"{rec_id}.flac")
        
        try:
            waveform, sr = sf.read(file_path)
            waveform = torch.tensor(waveform, dtype=torch.float32)

            mel_spec = mel_spectrogram(waveform)
            frame_energy = mel_spec.sum(dim=0).numpy()

            filtered_energy = medfilt(frame_energy, kernel_size=config.MEDIAN_FILTER_SIZE)
            
            median_energy = np.median(filtered_energy)
            std_energy = np.std(filtered_energy)
            threshold = median_energy + config.ENERGY_THRESHOLD_STD_MULTIPLIER * std_energy
            
            active_frames = np.where(filtered_energy > threshold)[0]
            if not len(active_frames):
                continue

            events = []
            start_frame = active_frames[0]
            for i in range(1, len(active_frames)):
                if active_frames[i] != active_frames[i-1] + 1:
                    events.append((start_frame, active_frames[i-1]))
                    start_frame = active_frames[i]
            events.append((start_frame, active_frames[-1]))
            
            original_rows = df[df['recording_id'] == rec_id]
            for _, original_row in original_rows.iterrows():
                for start_f, end_f in events:
                    new_row = original_row.copy()
                    new_row['f_min'] = start_f
                    new_row['f_max'] = end_f
                    all_strong_labels.append(new_row)
        
        except Exception:
            # Instead of printing, we just increment the counter and continue
            skipped_count += 1
            continue

    strong_labels_df = pd.DataFrame(all_strong_labels)
    strong_labels_df.to_csv(config.OUTPUT_METADATA_PATH, index=False)
    
    print(f"\n\n🎉 Pseudo strong label generation complete!")
    print(f"Skipped {skipped_count} corrupted files.")
    print(f"Saved {len(strong_labels_df)} frame-level labels to:")
    print(config.OUTPUT_METADATA_PATH)

if __name__ == '__main__':
    generate_labels()