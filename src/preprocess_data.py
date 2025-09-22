import os
import torch
import soundfile as sf
import torchaudio.transforms as T
from tqdm import tqdm

# --- Configuration (must match train_mae.py) ---
class PreprocessConfig:
    # Paths
    INPUT_PATH = 'data/unlabeled_corpus'
    OUTPUT_PATH = 'data/preprocessed_spectrograms'
    
    # Audio
    SAMPLE_RATE = 16000
    
    # Spectrogram
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 160
    IMG_WIDTH = 800 # This is the fixed time dimension from MAEConfig.IMG_SIZE

def preprocess_data():
    """
    Iterates through the audio corpus, converts each file to a standardized
    log-mel spectrogram tensor, and saves it to disk.
    """
    config = PreprocessConfig()
    
    # Create the output directory if it doesn't exist
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    print(f"✅ Output directory created at: {config.OUTPUT_PATH}")
    
    # Get a list of all audio files to process
    file_list = [f for f in os.listdir(config.INPUT_PATH) if f.endswith('.wav')]
    
    # Initialize the same TorchAudio transforms we use in training
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        power=2.0
    ).to('cuda') # Run transforms on GPU for speed
    
    db_transform = T.AmplitudeToDB().to('cuda')
    
    print(f"Starting pre-processing for {len(file_list)} files...")
    
    for filename in tqdm(file_list, desc="Pre-processing audio", unit="file"):
        try:
            input_path = os.path.join(config.INPUT_PATH, filename)
            output_filename = os.path.splitext(filename)[0] + '.pt'
            output_path = os.path.join(config.OUTPUT_PATH, output_filename)

            # --- This logic is a direct copy from our robust Dataset ---
            waveform, sr = sf.read(input_path)
            waveform = torch.tensor(waveform, dtype=torch.float32).to('cuda')

            mel_spec = mel_spectrogram(waveform)
            log_mel_spec = db_transform(mel_spec + 1e-6)

            std = log_mel_spec.std()
            if std > 1e-6:
                log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / std
            else:
                log_mel_spec = torch.zeros_like(log_mel_spec)

            target_len = config.IMG_WIDTH
            current_len = log_mel_spec.shape[1]
            if current_len > target_len:
                log_mel_spec = log_mel_spec[:, :target_len]
            else:
                padding = torch.zeros(config.N_MELS, target_len - current_len, device='cuda')
                log_mel_spec = torch.cat([log_mel_spec, padding], dim=1)
            
            # --- Save the final tensor to disk (move back to CPU for saving) ---
            torch.save(log_mel_spec.cpu(), output_path)

        except Exception as e:
            print(f"\nCould not process {filename}. Error: {e}. Skipping.")
            continue
            
    print(f"\n🎉 Pre-processing complete!")
    print(f"Saved {len(os.listdir(config.OUTPUT_PATH))} spectrogram tensors to {config.OUTPUT_PATH}")

if __name__ == '__main__':
    preprocess_data()