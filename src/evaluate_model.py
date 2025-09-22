import torch
import numpy as np
import pandas as pd
import soundfile as sf
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import medfilt # New import for the median filter

# Import necessary components
from train_biomaestro import LitBioMaestro, BioMaestroConfig
from run_inference import preprocess_chunk

# --- (Helper functions like predictions_to_events, calculate_iou, etc. remain the same) ---
def predictions_to_events(probs, threshold, sr, hop_length):
    frames = np.where(probs > threshold)[0]
    if not len(frames): return []
    events, start_frame = [], frames[0]
    for i in range(1, len(frames)):
        if frames[i] != frames[i-1] + 1:
            events.append((start_frame, frames[i-1]))
            start_frame = frames[i]
    events.append((start_frame, frames[-1]))
    return [(s * hop_length / sr, e * hop_length / sr) for s, e in events]

def calculate_iou(event1, event2):
    start1, end1 = event1
    start2, end2 = event2
    intersection_start, intersection_end = max(start1, start2), min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    if intersection == 0: return 0.0
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union

def calculate_tp_fp_fn(gt_events, pred_events, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    if not pred_events: return 0, 0, len(gt_events)
    if not gt_events: return 0, len(pred_events), 0
    matches = []
    for gt_event in gt_events:
        best_iou, best_pred_idx = 0, -1
        for i, pred_event in enumerate(pred_events):
            iou = calculate_iou(gt_event, pred_event)
            if iou > best_iou:
                best_iou, best_pred_idx = iou, i
        if best_iou > iou_threshold and best_pred_idx not in matches:
            tp += 1
            matches.append(best_pred_idx)
        else:
            fn += 1
    fp = len(pred_events) - len(matches)
    return tp, fp, fn

def main(args):
    config = BioMaestroConfig()
    print(f"Loading checkpoint: {args.checkpoint}")
    model = LitBioMaestro.load_from_checkpoint(args.checkpoint, config=config, strict=False).to('cuda').eval()

    df = pd.read_csv(config.TRAIN_METADATA_PATH)
    all_recordings = df['recording_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_recordings)
    split_idx = int(len(all_recordings) * 0.8)
    val_recordings = all_recordings[split_idx:]
    if args.limit: val_recordings = val_recordings[:args.limit]
    print(f"Evaluating on {len(val_recordings)} validation recordings...")

    all_predictions = {}
    for rec_id in tqdm(val_recordings, desc="Step 1: Running Inference", unit="file"):
        # ... (Inference logic is the same as before) ...
        file_path = f"{config.TRAIN_AUDIO_PATH}/{rec_id}.flac"
        try:
            waveform, sr = sf.read(file_path)
        except sf.LibsndfileError as e:
            print(f"\nWarning: Skipping corrupted file: {file_path}")
            continue
        chunk_samples, step_samples = config.DURATION * config.SAMPLE_RATE, (config.DURATION * config.SAMPLE_RATE) // 2
        all_det_probs = []
        with torch.no_grad():
            for i in range(0, max(1, len(waveform) - chunk_samples + 1), step_samples):
                chunk = waveform[i : i + chunk_samples]
                if len(chunk) < chunk_samples: # Pad the last chunk if it's too short
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                spectrogram_tensor = preprocess_chunk(chunk, config).to('cuda')
                det_logits, _ = model(spectrogram_tensor)
                all_det_probs.append(torch.sigmoid(det_logits.squeeze()))
        
        if not all_det_probs: continue

        num_frames_per_chunk, num_frames_step = all_det_probs[0].shape[0], all_det_probs[0].shape[0] // 2
        total_frames = num_frames_per_chunk + (len(all_det_probs) - 1) * num_frames_step
        stitched_det_probs = torch.zeros(total_frames)
        stitch_counts = torch.zeros(total_frames)
        for i, probs in enumerate(all_det_probs):
            start_idx = i * num_frames_step
            stitched_det_probs[start_idx : start_idx + num_frames_per_chunk] += probs.cpu()
            stitch_counts[start_idx : start_idx + num_frames_per_chunk] += 1
        
        stitched_det_probs /= stitch_counts
        all_predictions[rec_id] = stitched_det_probs.numpy()

    thresholds = np.linspace(0.05, 0.95, 20)
    precision_scores, recall_scores, f1_scores = [], [], []
    
    for threshold in tqdm(thresholds, desc="Step 2: Finding Best Threshold", unit="thresh"):
        total_tp, total_fp, total_fn = 0, 0, 0
        for rec_id, pred_probs in all_predictions.items():
            
            # --- KEY CHANGE: APPLY POST-PROCESSING ---
            # Kernel size of 5 means it looks at 5 frames at a time.
            filtered_probs = medfilt(pred_probs, kernel_size=5)

            pred_events = predictions_to_events(filtered_probs, threshold, config.SAMPLE_RATE, config.HOP_LENGTH)
            gt_calls = df[df['recording_id'] == rec_id]
            gt_events = list(zip(gt_calls['t_min'], gt_calls['t_max']))
            
            tp, fp, fn = calculate_tp_fp_fn(gt_events, pred_events)
            total_tp += tp; total_fp += fp; total_fn += fn
            
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precision_scores.append(precision); recall_scores.append(recall); f1_scores.append(f1)

    best_f1_idx = np.argmax(f1_scores)
    best_threshold, best_f1 = thresholds[best_f1_idx], f1_scores[best_f1_idx]
    best_precision, best_recall = precision_scores[best_f1_idx], recall_scores[best_f1_idx]

    print("\n--- Evaluation Complete (with Post-Processing) ---")
    print(f"✅ Best F1-Score: {best_f1:.4f}")
    print(f"📈 Found at threshold: {best_threshold:.2f}")
    print(f"Precision at best F1: {best_precision:.4f}")
    print(f"Recall at best F1:    {best_recall:.4f}")
    print("--------------------------------------------------\n")

    plt.figure(figsize=(8, 6))
    plt.plot(recall_scores, precision_scores, marker='o', linestyle='--')
    plt.scatter(best_recall, best_precision, color='red', zorder=5, s=100, label=f'Best F1-Score ({best_f1:.2f})')
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.xlim([0, 1]); plt.ylim([0, 1]); plt.grid(True); plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the BioMaestro model with post-processing.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--limit', type=int, default=None, help='Limit evaluation to N files.')
    args = parser.parse_args()
    main(args)