import argparse
import os
import numpy as np
import random
import torch
import subprocess
import re
import pickle
import clip
import ffmpeg
from model import build_vid2seq_model, _get_tokenizer
from args import get_args_parser, MODEL_DIR


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor

class Preprocessing(object):
    def __init__(self):
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def __call__(self, tensor):
        tensor = tensor / 255.0
        tensor = self.norm(tensor)
        return tensor

def time_tokenize(x, duration, num_bins, num_text_tokens):
    time_token = int(float((num_bins - 1) * x) / float(duration))
    assert time_token <= num_bins
    return time_token + num_text_tokens

# Args
parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args()
args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
device = torch.device(args.device)

# Fix seeds
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Build Vid2Seq model
print("load Vid2Seq model")
tokenizer = _get_tokenizer(args.model_name, args.num_bins)
model = build_vid2seq_model(args, tokenizer)
model.eval()
model.to(device)
assert args.load
checkpoint = torch.load(args.load, map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)

# Extract video frames from video
print("loading visual backbone")
preprocess = Preprocessing()
backbone, _ = clip.load("ViT-L/14", download_root=MODEL_DIR, device=device)
backbone.eval()
backbone.to(device)
print("extracting visual features")
probe = ffmpeg.probe(args.video_example)
video_stream = next(
    (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
)
width = int(video_stream["width"])
height = int(video_stream["height"])
num, denum = video_stream["avg_frame_rate"].split("/")
frame_rate = int(num) / int(denum)
if height >= width:
    h, w = int(height * 224 / width), 224
else:
    h, w = 224, int(width * 224 / height)
assert frame_rate >= 1

cmd = ffmpeg.input(args.video_example).filter("fps", fps=1).filter("scale", w, h)
x = int((w - 224) / 2.0)
y = int((h - 224) / 2.0)
cmd = cmd.crop(x, y, 224, 224)
out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
    capture_stdout=True, quiet=True
)

h, w = 224, 224
video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
video = torch.from_numpy(video.astype("float32"))
video = video.permute(0, 3, 1, 2)
video = video.squeeze()
video = preprocess(video)
with torch.no_grad():
    video = backbone.encode_image(video.to(device))

# Subsample or pad visual features
if len(video) >= args.max_feats:
    sampled = []
    for j in range(args.max_feats):
        sampled.append(video[(j * len(video)) // args.max_feats])
    video = torch.stack(sampled)
    video_len = args.max_feats
else:
    video_len = len(video)
    video = torch.cat(
        [video, torch.zeros(args.max_feats - video_len, 768).to(device)], 0
    )
video = video.unsqueeze(0).to(device)
print("visual features extracted")

# Extract ASR from video
assert args.asr_example
print("load ASR")
segments = pickle.load(open(args.asr_example, 'rb'))["segments"]
texts, starts, ends = [], [], []
for i in range(len(segments)):
    text = segments[i]['text']
    if text.strip():
        texts.append(text)
        starts.append(segments[i]['start'])
        ends.append(segments[i]['end'])
sub = {'text': texts,
       'start': starts,
       'end': ends}

# ASR to tokens
print("ASR to tokens")
probe = subprocess.run(
    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
     args.video_example], stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT)
duration = float(probe.stdout)
if not sub['text']:
    input_tokens = (torch.ones(1) * tokenizer.eos_token_id).long()
else:
    time_input_tokens = [torch.LongTensor([time_tokenize(st, duration, args.num_bins, len(tokenizer) - args.num_bins),
                                           time_tokenize(ed, duration, args.num_bins, len(tokenizer) - args.num_bins)])
                         for st, ed in zip(sub['start'], sub['end'])]
    text_input_tokens = [tokenizer(x, add_special_tokens=False, max_length=args.max_input_tokens,
                                   padding="do_not_pad", truncation=True, return_tensors="pt",)['input_ids'][0]
                         for x in sub['text']]
    input_tokens = [torch.cat([ti, te], 0) for ti, te in zip(time_input_tokens, text_input_tokens)]
    input_tokens = torch.cat(input_tokens, 0)
    input_tokens = input_tokens[:args.max_input_tokens - 1]
    input_tokens = torch.cat([input_tokens, torch.LongTensor([tokenizer.eos_token_id])], 0)
input_tokens = input_tokens.unsqueeze(0).to(device)
input_tokenized = {'input_ids': input_tokens,
                   'attention_mask': input_tokens != 0}

# Forward to the Vid2Seq model
print("forward to Vid2Seq")
with torch.no_grad():
    output = model.generate(video=video,
                            input_tokenized=input_tokenized,
                            use_nucleus_sampling=args.num_beams == 0,
                            num_beams=args.num_beams,
                            max_length=args.max_output_tokens,
                            min_length=1,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                            length_penalty=args.length_penalty,
                            num_captions=1,
                            temperature=1)

# Decode result
print("decode results")
sequences = re.split(r'(?<!<)\s+(?!>)', output[0]) # "<time=5> <time=7> Blablabla <time=7> <time=9> Blobloblo <time=2>" -> ['<time=5>', '<time=7>', 'Blablabla', '<time=7>', '<time=9>', 'Blobloblo', '<time=2>']
indexes = [j for j in range(len(sequences) - 1) if sequences[j][:6] == '<time=' and sequences[j + 1][:6] == '<time=']
last_processed = -2
res = []
for j, idx in enumerate(indexes):  # iterate on predicted events
    if idx == last_processed + 1:  # avoid processing 3 time tokens in a row as 2 separate events
        continue
    seq = [sequences[k] for k in range(idx + 2, indexes[j + 1] if j < len(indexes) - 1 else len(sequences)) if sequences[k] != '<time=']
    if seq:
        text = ' '.join(seq)
    else:  # no text
        continue
    start_re = re.search(r'\<time\=(\d+)\>', sequences[idx])
    assert start_re, sequences[idx]
    start_token = int(start_re.group(1))
    start = float(start_token) * float(duration) / float(args.num_bins - 1)
    end_re = re.search(r'\<time\=(\d+)\>', sequences[idx + 1])
    assert end_re, sequences[idx + 1]
    end_token = int(end_re.group(1))
    end = float(end_token) * float(duration) / float(args.num_bins - 1)
    if end <= start:  # invalid time
        continue
    res.append({'sentence': text,
                'timestamp': [start, end]})
    last_processed = idx
print(res)
