import whisper
import whisperx
import argparse
import pandas as pd
import os
from tqdm import tqdm
import torch
import pickle
from args import MODEL_DIR

parser = argparse.ArgumentParser(description='Easy ASR extractor')
parser.add_argument('--csv', type=str,
                    help='input csv with video input path')
parser.add_argument('--type', type=str, default='large-v2', choices=['large-v2'],
                    help='model type')
parser.add_argument('--device', type=str, default='cuda',
                    help='device')
parser.add_argument('--output_path', type=str, required=True,
                    help='path where to save results')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--faster', action='store_true')
args = parser.parse_args()

df = pd.read_csv(args.csv)
df = df.sample(frac=1)

if args.faster:
    model = whisperx.load_model(args.type, model=args.model_path, device=args.device)
else:
    model = whisper.load_model(args.type, args.device, download_root=os.path.join(MODEL_DIR, 'models--guillaumekln--faster-whisper-large-v2/snapshots/fecb99cc227a240ccd295d99b6c9026e7a179508'))

print("starting extraction")
with torch.no_grad():
    for index, row in tqdm(df.iterrows()):
        video_path = row["video_path"]
        target_path = os.path.join(args.output_path, video_path.split('/')[-1] + '.pkl')
        if os.path.exists(target_path):
            continue
        if args.faster:
            audio = whisperx.load_audio(video_path)
            try:
                result = model.transcribe(audio, batch_size=args.batch_size)
            except RuntimeError:
                continue
        else:
            try:
                result = model.transcribe(video_path)
            except RuntimeError:
                continue
        pickle.dump(result, open(target_path, 'wb'))