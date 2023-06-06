import whisperx
import argparse
import pandas as pd
import os
from tqdm import tqdm
import torch
import pickle
import random
from args import MODEL_DIR

parser = argparse.ArgumentParser(description='Easy ASR extractor')
parser.add_argument('--csv', type=str, required=True,
                    help='input csv with video input path')
parser.add_argument('--asr', type=str, required=True,
                    help='path to extracted ASR w/ Whisper')
parser.add_argument('--output_path', type=str, required=True,
                    help='path where to save results')
parser.add_argument('--model_path', type=str, default=MODEL_DIR,
                    help='path to model weights')
parser.add_argument('--device', type=str, default='cuda',
                    help='device')
args = parser.parse_args()

df = pd.read_csv(args.csv)
df = df.sample(frac=1)

mapping = {x.split('/')[-1][:11]: x for x in df['video_path']}

asr = pickle.load(open(args.asr, 'rb'))
languages = set(x['language'] for x in asr.values())

print("starting extraction")
with torch.no_grad():
    for language in languages:
        print(language)
        try:
            model_a, metadata = whisperx.load_align_model(language_code=language, device=args.device,
                                                          model_dir=args.model_path)
        except ValueError:
            continue
        videos = [(x, mapping[x]) for x in asr if asr[x].get('language', '') == language]
        random.shuffle(videos)
        print(len(videos))

        for x in tqdm(videos):
            target_path = os.path.join(args.output_path, x[1].split('/')[-1] + '.pkl')
            if os.path.exists(target_path):
                continue
            audio = whisperx.load_audio(x[1])
            try:
                aligned_asr = whisperx.align(asr[x[0]]["segments"], model_a, metadata, audio, args.device, return_char_alignments=False)
            except:
                print(x)
                continue
            pickle.dump(aligned_asr, open(target_path, 'wb'))
