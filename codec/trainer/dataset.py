from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np

def collate_fn(data):
    # return pad_sequence(data, batch_first=True)
    # return pad_sequence(*data)
    is_one_data = not isinstance(data[0], tuple)
    outputs = []
    if is_one_data:
        for datum in data:
            if isinstance(datum, torch.Tensor):
                output = datum.unsqueeze(0)
            else:
                output = torch.tensor([datum])
            outputs.append(output)
        return tuple(outputs)        
    for datum in zip(*data):
        if isinstance(datum[0], torch.Tensor):
            output = pad_sequence(datum, batch_first=True)
        else:
            output = torch.tensor(list(datum))
        outputs.append(output)

    return tuple(outputs)

def get_dataloader(ds, **kwargs):
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)

class audioDataset(Dataset):
    
    def __init__(self,
                 file_list,
                 segment_size,
                 sample_rate,
                 teacher,
                 downsample_rate = 320,
                 valid=False):
        super().__init__()
        self.file_list = file_list
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.valid = valid
        self.downsample_rate = downsample_rate
        self.teacher = teacher
        
    def __len__(self):
        return len(self.file_list)
    
    
    def __getitem__(self, index):
        file = self.file_list[index].strip()
        
        if self.teacher.startswith('combined'):
            audio_file, hubert_feature_file, llm_feature_file = file.split('\t')
            audio, sr = torchaudio.load(audio_file)
            hubert_feature = torch.from_numpy(np.load(hubert_feature_file))
            llm_feature = torch.from_numpy(np.load(llm_feature_file))
            audio = audio.mean(axis=0)
            return audio, hubert_feature, llm_feature
            
        audio_file, feature_file = file.split('\t')
        audio, sr = torchaudio.load(audio_file)
        feature = torch.from_numpy(np.load(feature_file))
        audio = audio.mean(axis=0)
        # print("debug before:-------->", audio.shape, feature.shape)     
        
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        
        if audio.size(-1) > self.segment_size:
            print("############## audio.size(-1) > self.segment_size!! Check if dataset was processed correctly.")
            if self.valid:
                return audio[:self.segment_size], feature[:self.segment_size // self.downsample_rate]
            # Randomly select a starting point for the audio segment
            max_start_index = audio.size(-1) - self.segment_size
            start_index = torch.randint(0, max_start_index + 1, (1,)).item()
            end_index = start_index + self.segment_size
            # Slice the audio and feature to the segment size
            audio = audio[start_index:end_index]
            feature = feature[:, start_index // self.downsample_rate : end_index // self.downsample_rate, :]
            
        elif audio.size(-1) < self.segment_size:
            if not self.valid:
                print("############## audio.size(-1) < self.segment_size!!! Check if dataset was processed correctly.")
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')
                # Calculate the number of frames in the padded segment
                num_frames = self.segment_size // self.downsample_rate
                # If feature has fewer frames, pad it
                if feature.size(1) < num_frames:
                    feature_padding_length = num_frames - feature.size(1)
                    feature = torch.nn.functional.pad(feature, (0, 0, 0, feature_padding_length), 'constant')
                feature = feature[:, :num_frames, :]

        # print("debug after:-------->", audio.shape, feature.shape)     
        return audio, feature