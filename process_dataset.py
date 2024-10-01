from datasets import load_dataset
import librosa
import numpy as np
import soundfile as sf
import argparse
import os
import torch

def process_audio(example, segment_size):
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    segment_size_samples = int(segment_size * sr)
    if len(audio) > segment_size_samples:
        max_start_index = len(audio) - segment_size_samples
        start_index = np.random.randint(0, max_start_index + 1)
        end_index = start_index + segment_size_samples
        audio = audio[start_index:end_index]
    else:
        padding_length = segment_size_samples - len(audio)
        audio = np.pad(audio, (0, padding_length), 'constant')
        
    output_path = os.path.join(output_dir, f"processed_{example['audio']['path'].split('/')[-1]}")
    sf.write(output_path, audio, sr)

    return {"path": output_path, "sampling_rate": sr}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and resample audio files from a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to load.")
    parser.add_argument("--split", type=str, required=True, help="Subset of the dataset to use.")
    parser.add_argument("--ratio", type=float, required=True, help="Ratio of the dataset to use.")
    parser.add_argument("--segment_length", type=float, required=True, help="Length of each audio segment in seconds.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.select(range(int(len(dataset) * args.ratio)))

    output_dir = "processed_dataset"
    os.makedirs(output_dir, exist_ok=True)

    processed_dataset = dataset.map(lambda example: process_audio(example, args.segment_length), remove_columns=dataset.column_names)
    processed_dataset.save_to_disk(output_dir)
