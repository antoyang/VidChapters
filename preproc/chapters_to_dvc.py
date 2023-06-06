from tqdm import tqdm
import pickle
import json
import os
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from args import DATA_DIR, name2folder

tokenizer = PTBTokenizer()

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def filter(sentence):
    mark = [',', ':', '!', '_', ';', '-', '.', '?', '/', '"', '\\n', '\\']
    for m in mark:
        if m in sentence:
            sentence = sentence.replace(m, " ")
        sentence = sentence.replace("  ", " ")
        sentence = sentence.replace("  ", " ")
        sentence = sentence.replace("  ", " ")

    sentence = sentence.lstrip()
    sentence = sentence.rstrip()
    sentence = sentence.lower()
    return sentence

data = pickle.load(open(os.path.join(DATA_DIR, name2folder["chapters"], 'chapters.pkl'), 'rb'))

vid2tok = {}
for vid, anns in tqdm(data.items()):
    if not "duration" in anns:
        continue
    timestamps = [[float(x['time']),
                   float(anns['chapters'][i + 1]['time']) if i < len(anns['chapters']) - 1 else float(anns['duration'])]
                  for i, x in enumerate(anns['chapters'])]
    timestamps = [[x[0], x[1]] for x in timestamps if x[0] <= x[1] <= anns['duration']]
    if not timestamps:
        continue
    assert all(x[0] <= x[1] <= anns['duration'] for x in timestamps), (vid, anns)

    dico = {i: [{'caption': remove_nonascii(x['label']).strip()}] for i, x in enumerate(anns['chapters'])}
    vid2tok.update({vid + '_' + str(k) : v for k, v in dico.items()})
vid2tok = tokenizer.tokenize(vid2tok)
vid2tok = {k: v[0].strip() for k, v in vid2tok.items()}


qid = 0

out = {}
bug = 0
for vid, anns in tqdm(data.items()):
    if not "duration" in anns:
        continue
    timestamps = [[float(x['time']), float(anns['chapters'][i+1]['time']) if i < len(anns['chapters']) - 1 else float(anns['duration'])] for i, x in enumerate(anns['chapters'])]
    timestamps = [[x[0], x[1]] for x in timestamps if x[0] <= x[1] <= anns['duration']]
    if not timestamps:
        continue
    assert all(x[0] <= x[1] <= anns['duration'] for x in timestamps), (vid, anns)

    if (not all([filter(x['label']) for x in anns['chapters']])):  # weird chapters without text content
        continue

    dico = [vid2tok[vid + '_' + str(k)] for k in range(len(anns['chapters']))]
    if not all(dico):
        bug += 1
        continue

    out[vid] = {'duration': float(anns['duration']),
                'timestamps': timestamps,
                'sentences': [x['label'] for x in anns['chapters']],
                'path': vid + '.mp4.npy'}

print(bug)
print(len(out))
print(list(out.values())[-1])

train_videos = set(json.load(open(os.path.join(DATA_DIR, name2folder["chapters"], 'train.json'), 'r')))
val_videos = set(json.load(open(os.path.join(DATA_DIR, name2folder["chapters"], 'val.json'), 'r')))
test_videos = set(json.load(open(os.path.join(DATA_DIR, name2folder["chapters"], 'test.json'), 'r')))

with open(os.path.join(DATA_DIR, name2folder["chapters"], 'chapters_dvc_train.json'), 'w') as outfile:
    json.dump({x: out[x] for x in out if x in train_videos}, outfile)

with open(os.path.join(DATA_DIR, name2folder["chapters"], 'chapters_dvc_val.json'), 'w') as outfile:
    json.dump({x: out[x] for x in out if x in val_videos}, outfile)

with open(os.path.join(DATA_DIR, name2folder["chapters"], 'chapters_dvc_test.json'), 'w') as outfile:
    json.dump({x: out[x] for x in out if x in test_videos}, outfile)