from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, AutoModel, AutoTokenizer, AutoFeatureExtractor
from pathlib import Path
import torchaudio
import torch
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Config file path')
    parser.add_argument('--audio_dir', type=str, required=True, help='Audio folder path')
    parser.add_argument('--rep_typ', type=str, required=True, help='Representation type: "hubert", "llm", or "combined"')
    parser.add_argument('--exts', type=str, required=True, help="Audio file extensions, splitting with ','")
    parser.add_argument('--split_seed', type=int, required=True, help="Random seed")
    parser.add_argument('--valid_set_size', type=float, required=True, help="Size of the validation set")
    parser.add_argument('--rep_dir', type=str, required=True, help="Path to representation folder")
    parser.add_argument('--semantic_model_path', type=str, required=True, help="Path to the Hubert model")
    parser.add_argument('--semantic_model_layer', type=str, required=True, help="Target layer for Hubert model")
    parser.add_argument('--stt_model_path', type=str, required=True, help="Path to the STT model")
    parser.add_argument('--llm_model_path', type=str, required=True, help="Path to the LLM model")
    parser.add_argument('--llm_model_layer', type=str, required=True, help="Target layer for LLM model")
    
    args = parser.parse_args()
    exts = args.exts.split(',')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.config) as f:
        cfg = json.load(f)
    sample_rate = cfg.get('sample_rate')

    if "last" in args.rep_typ:
        args.semantic_model_layer = 12
        args.llm_model_layer = 12
        args.rep_dir = "last"
    
    if "nine" in args.rep_typ:
        args.semantic_model_layer = 10
        args.llm_model_layer = 10
        args.rep_dir = "nine"

    if "electra" in args.rep_typ:
        args.llm_model_path = "google/electra-base-discriminator"
    
    if "wav2vec" in args.rep_typ:
        args.semantic_model_path = "facebook/wav2vec2-base-960h"
        
    if args.rep_typ.startswith(('hubert', 'combined')):
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.semantic_model_path)
        hubert_model = AutoModel.from_pretrained(args.semantic_model_path).eval().to(device)
        hubert_target_layer = args.semantic_model_layer

    if args.rep_typ.startswith(('llm', 'combined')):
        stt_model_path = args.stt_model_path
        stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_path).eval().to(device)
        stt_tokenizer = Wav2Vec2Tokenizer.from_pretrained(stt_model_path)
        llm_model_path = args.llm_model_path
        llm_model = AutoModel.from_pretrained(llm_model_path).eval().to(device)
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        llm_target_layer = args.llm_model_layer

    path = Path(args.audio_dir)
    file_list = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]

    if args.valid_set_size != 0 and args.valid_set_size < 1:
        valid_set_size = int(len(file_list) * args.valid_set_size)
    else:
        valid_set_size = int(args.valid_set_size)

    train_file_list = f"{args.rep_typ}_train_file_list.txt"
    valid_file_list = f"{args.rep_typ}_dev_file_list.txt"
    segment_size = cfg.get('segment_size')
    random.seed(args.split_seed)
    random.shuffle(file_list)
    print(f'A total of {len(file_list)} samples will be processed, and {valid_set_size} of them will be included in the validation set.')

    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):
            wav, sr = torchaudio.load(audio_file)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.size(-1) < segment_size:
                print("@@@@@@@@ wav.size(-1) < segment_size!! Check if dataset was processed correctly.")
                wav = torch.nn.functional.pad(wav, (0, segment_size - wav.size(-1)), 'constant')

            if args.rep_typ.startswith(('hubert', 'combined')):
                input_values = feature_extractor(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
                hubert_output = hubert_model(input_values.to(hubert_model.device), output_hidden_states=True)
                if hubert_target_layer == 'avg':
                    hubert_rep = torch.mean(torch.stack(hubert_output.hidden_states), axis=0)
                else:
                    hubert_rep = hubert_output.hidden_states[hubert_target_layer]
                hubert_rep_file = audio_file.replace(args.audio_dir, f'hubert_{args.rep_dir}').split('.')[0] + '.hubert.npy'
                hubert_rep_sub_dir = '/'.join(hubert_rep_file.split('/')[:-1])
                if not os.path.exists(hubert_rep_sub_dir):
                    os.makedirs(hubert_rep_sub_dir)
                np.save(hubert_rep_file, hubert_rep.detach().cpu().numpy())

            if args.rep_typ.startswith(('llm', 'combined')):
                input_values = stt_tokenizer(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
                logits = stt_model(input_values.to(stt_model.device)).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = stt_tokenizer.batch_decode(predicted_ids)[0]
                llm_inputs = llm_tokenizer(transcription, return_tensors="pt", truncation=True,
                                           padding="max_length", max_length=logits.shape[1]).to(device)
                llm_outputs = llm_model(**llm_inputs, output_hidden_states=True)
                if llm_target_layer == 'avg':
                    llm_rep = torch.mean(torch.stack(llm_outputs.hidden_states), axis=0)
                else:
                    llm_rep = llm_outputs.hidden_states[llm_target_layer]
                llm_rep_file = audio_file.replace(args.audio_dir, f'llm_{args.rep_dir}').split('.')[0] + '.llm.npy'
                llm_rep_sub_dir = '/'.join(llm_rep_file.split('/')[:-1])
                if not os.path.exists(llm_rep_sub_dir):
                    os.makedirs(llm_rep_sub_dir)
                np.save(llm_rep_file, llm_rep.detach().cpu().numpy())

            if args.rep_typ.startswith('hubert'):
                rep_line = audio_file + "\t" + hubert_rep_file + "\n"
            elif args.rep_typ.startswith('llm'):
                rep_line = audio_file + "\t" + llm_rep_file + "\n"
            elif args.rep_typ.startswith('combined'):
                rep_line = audio_file + "\t" + hubert_rep_file + "\t" + llm_rep_file + "\n"

            if i == 0 or i == valid_set_size:
                with open(valid_file_list if i < valid_set_size else train_file_list, 'w') as f:
                    f.write(rep_line)
            else:
                with open(valid_file_list if i < valid_set_size else train_file_list, 'a+') as f:
                    f.write(rep_line)
