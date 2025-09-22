import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings

# --- Configuration ---
# Define paths based on the project structure
BASE_DATA_PATH = 'data/rfcx-species-audio-detection'
TRAIN_AUDIO_PATH = os.path.join(BASE_DATA_PATH, 'train')
TEST_AUDIO_PATH = os.path.join(BASE_DATA_PATH, 'test')
METADATA_PATH = os.path.join(BASE_DATA_PATH, 'train_tp.csv')
OUTPUT_CORPUS_PATH = 'data/unlabeled_corpus'

# Define audio processing parameters
TARGET_SR = 16000  # 16 kHz as per the plan
CHUNK_DURATION = 8  # 8 seconds
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION)

# Suppress librosa's audioread warning if it pops up
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# --- Main Script ---
def create_unlabeled_corpus():
    """
    Scans the RFCx dataset, identifies audio without labels,
    and processes them into a standardized, chunked corpus for SSL pre-training.
    """
    # 1. Create the output directory
    print(f"✅ Creating output directory at: {OUTPUT_CORPUS_PATH}")
    os.makedirs(OUTPUT_CORPUS_PATH, exist_ok=True)

    # 2. Identify all recordings that have ground-truth labels
    print(f"Reading metadata from: {METADATA_PATH}")
    df_tp = pd.read_csv(METADATA_PATH)
    labeled_ids = set(df_tp['recording_id'].unique())
    print(f"Found {len(labeled_ids)} unique recordings with labels.")

    # 3. Identify unlabeled recordings by comparing against all train files
    all_train_files = {os.path.splitext(f)[0] for f in os.listdir(TRAIN_AUDIO_PATH) if f.endswith('.flac')}
    unlabeled_train_ids = all_train_files - labeled_ids
    print(f"Found {len(all_train_files)} total train recordings, so {len(unlabeled_train_ids)} are unlabeled.")

    # 4. Get all test recordings (which are all unlabeled)
    test_ids = {os.path.splitext(f)[0] for f in os.listdir(TEST_AUDIO_PATH) if f.endswith('.flac')}
    print(f"Found {len(test_ids)} test recordings.")

    # 5. Create a final list of full paths for all files to be processed
    unlabeled_files_to_process = []
    for rec_id in unlabeled_train_ids:
        unlabeled_files_to_process.append(os.path.join(TRAIN_AUDIO_PATH, f"{rec_id}.flac"))
    for rec_id in test_ids:
        unlabeled_files_to_process.append(os.path.join(TEST_AUDIO_PATH, f"{rec_id}.flac"))

    print(f"Total files to process for the unlabeled corpus: {len(unlabeled_files_to_process)}")
    print("-" * 50)

    # 6. Process each file: load, resample, slice, and save
    total_chunks_saved = 0
    for audio_path in tqdm(unlabeled_files_to_process, desc="🎛️ Processing files", unit="file"):
        try:
            # Load audio and resample to the target sample rate
            waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

            # Calculate the number of full, non-overlapping chunks we can extract
            num_chunks = len(waveform) // CHUNK_SAMPLES

            if num_chunks == 0:
                continue

            # Slice the waveform and save each chunk
            for i in range(num_chunks):
                start_sample = i * CHUNK_SAMPLES
                end_sample = start_sample + CHUNK_SAMPLES
                chunk = waveform[start_sample:end_sample]

                # Define a clear output filename
                original_filename = os.path.basename(audio_path)
                output_filename = f"{os.path.splitext(original_filename)[0]}_chunk{i}.wav"
                output_path = os.path.join(OUTPUT_CORPUS_PATH, output_filename)

                # Save the chunk as a .wav file
                sf.write(output_path, chunk, TARGET_SR)
                total_chunks_saved += 1

        except Exception as e:
            print(f"Could not process file {audio_path}. Error: {e}")
            continue

    print(f"\n🎉 --- Corpus Creation Complete! --- 🎉")
    print(f"Successfully created and saved {total_chunks_saved} audio chunks.")
    print(f"Your unlabeled corpus is ready at: {OUTPUT_CORPUS_PATH}")

if __name__ == '__main__':
    create_unlabeled_corpus()