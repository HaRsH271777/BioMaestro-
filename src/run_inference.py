import torch
import torch.nn.functional as F
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import torchaudio.transforms as T

# Import our model and config classes
from train_biomaestro import LitBioMaestro, BioMaestroConfig
from train_mae import MAEConfig

def get_species_map():
    df = pd.read_csv(BioMaestroConfig.TRAIN_METADATA_PATH)
    return {i: species for i, species in enumerate(df['species_id'].unique())}

def preprocess_chunk(waveform_chunk, config):
    """Preprocesses a single audio chunk."""
    waveform_tensor = torch.tensor(waveform_chunk, dtype=torch.float32)

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_mels=config.N_MELS, power=2.0
    )(waveform_tensor)
    
    log_mel_spec = T.AmplitudeToDB()(mel_spectrogram + 1e-6)
    
    std = log_mel_spec.std()
    if std > 1e-6:
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / std
    else:
        log_mel_spec = torch.zeros_like(log_mel_spec)

    # Resize to match model input
    target_len = config.IMG_SIZE[1]
    current_len = log_mel_spec.shape[1]
    if current_len > target_len:
        log_mel_spec = log_mel_spec[:, :target_len]
    else:
        log_mel_spec = torch.cat([log_mel_spec, torch.zeros(config.N_MELS, target_len - current_len)], dim=1)

    return log_mel_spec.unsqueeze(0)

def visualize_predictions(full_waveform, detection_probs, top_species, top_probs, config, title, output_path=None):
    """Generates and saves/shows a plot of the model's predictions for the full audio."""
    fig, ax = plt.subplots(figsize=(20, 6))
    
    spec_db = librosa.amplitude_to_db(librosa.feature.melspectrogram(
        y=full_waveform, sr=config.SAMPLE_RATE, n_fft=config.N_FFT, 
        hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
    ), ref=np.max)
    librosa.display.specshow(spec_db, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
    
    time_axis = np.linspace(0, len(full_waveform) / config.SAMPLE_RATE, len(detection_probs))
    ax.plot(time_axis, detection_probs * config.N_MELS, color='cyan', label='Detection Probability')
    
    detection_threshold = 0.5
    ax.fill_between(time_axis, 0, config.N_MELS, where=detection_probs > detection_threshold, color='cyan', alpha=0.3)

    result_text = "Top 5 Predictions (Overall):\n"
    for species, prob in zip(top_species, top_probs):
        result_text += f"{species}: {prob:.2f}\n"
    ax.text(0.01, 0.95, result_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.tight_layout()

    if output_path:
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

def main(args):
    config = BioMaestroConfig()
    species_map = get_species_map()

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = LitBioMaestro.load_from_checkpoint(args.checkpoint, config=config, strict=False).to('cuda').eval()

    if args.export_onnx:
        # ... (export logic remains the same) ...
        return

    if not args.input:
        print("Error: Please provide an input audio file with --input.")
        return

    print(f"Processing audio file with sliding window: {args.input}")
    waveform, sr = sf.read(args.input)

    # --- Sliding Window Logic ---
    chunk_samples = config.DURATION * config.SAMPLE_RATE
    step_samples = chunk_samples // 2 # 50% overlap
    all_det_probs = []
    all_cls_probs = []

    with torch.no_grad():
        for i in range(0, len(waveform) - chunk_samples + 1, step_samples):
            chunk = waveform[i : i + chunk_samples]
            spectrogram_tensor = preprocess_chunk(chunk, config).to('cuda')
            
            det_logits, cls_logits = model(spectrogram_tensor)
            
            all_det_probs.append(torch.sigmoid(det_logits.squeeze()))
            all_cls_probs.append(torch.sigmoid(cls_logits.squeeze()))

    # --- Stitch Predictions Together ---
    # For classification, we take the max probability across all chunks
    overall_cls_probs = torch.stack(all_cls_probs).max(dim=0)[0].cpu().numpy()
    
    # For detection, we need to carefully stitch the overlapping parts
    num_frames_per_chunk = all_det_probs[0].shape[0]
    num_frames_step = num_frames_per_chunk // 2
    total_frames = num_frames_per_chunk + (len(all_det_probs) - 1) * num_frames_step
    stitched_det_probs = torch.zeros(total_frames)
    
    for i, probs in enumerate(all_det_probs):
        start_idx = i * num_frames_step
        # Average the predictions in the overlapping regions
        stitched_det_probs[start_idx : start_idx + num_frames_per_chunk] += probs.cpu()
        if i > 0:
            stitched_det_probs[start_idx : start_idx + num_frames_step] /= 2.0
            
    stitched_det_probs = stitched_det_probs.numpy()

    top5_indices = np.argsort(overall_cls_probs)[-5:][::-1]
    top5_species = [species_map.get(i, "Unknown") for i in top5_indices]
    top5_probs = overall_cls_probs[top5_indices]
    
    title = f"BioMaestro Inference on {os.path.basename(args.input)}"
    visualize_predictions(waveform, stitched_det_probs, top5_species, top5_probs, config, title, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BioMaestro model inference with a sliding window.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--input', type=str, help='Path to a single audio file to process.')
    parser.add_argument('--output', type=str, help='Path to save the output plot image. If not provided, the plot is shown on screen.')
    parser.add_argument('--export-onnx', type=str, help='Path to save the exported model in ONNX format.')
    
    args = parser.parse_args()
    main(args)