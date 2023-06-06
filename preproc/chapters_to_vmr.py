from tqdm import tqdm
import pickle
import json
import os
from args import DATA_DIR, name2folder

data = pickle.load(open(os.path.join(DATA_DIR, name2folder["chapters"], 'chapters.pkl'), 'rb'))
qid = 0

train_videos = set(json.load(open(os.path.join(DATA_DIR, name2folder["chapters"], 'train.json'), 'r')))
val_videos = set(json.load(open(os.path.join(DATA_DIR, name2folder["chapters"], 'val.json'), 'r')))
test_videos = set(json.load(open(os.path.join(DATA_DIR, name2folder["chapters"], 'test.json'), 'r')))

out = []
cnt = {}
for vid, anns in tqdm(data.items()):
    if vid not in val_videos and vid not in train_videos and vid not in test_videos:
        continue
    cur = []
    for i, x in enumerate(anns['chapters']):
        window = [float(x['time']), float(anns['chapters'][i+1]['time']) if i < len(anns['chapters']) - 1 else float(anns['duration'])]
        if window[1] - window[0] >= 0.9 * anns['duration']:
            continue
        if window[0] <= window[1] <= anns['duration']:
            cur.append({'label': x['label'], 'window': window})
    if cur:
        if vid in train_videos:
            split = 'train'
            out.append({'qid': qid,
                        'query': [x['label'] for x in cur],
                        'vid': vid,
                        'duration': int(anns['duration']),
                        'split': split,
                        'relevant_windows': [[x['window']] for x in cur]})
            qid += 1
        elif vid in val_videos:
            split = 'val'
            for x in cur:
                out.append({'qid': qid,
                            'query': x['label'],
                            'vid': str(cnt.get(vid, 0)) + vid,
                            'duration': int(anns['duration']),
                            'split': split,
                            'relevant_windows': [x['window']]})
                cnt[vid] = cnt.get(vid, 0) + 1
                qid += 1
        elif vid in test_videos:
            split = 'test'
            for x in cur:
                out.append({'qid': qid,
                            'query': x['label'],
                            'vid': str(cnt.get(vid, 0)) + vid,
                            'duration': int(anns['duration']),
                            'split': split,
                            'relevant_windows': [x['window']]})
                cnt[vid] = cnt.get(vid, 0) + 1
                qid += 1

print(len(out))
print(out[-1])

with open(os.path.join(DATA_DIR, name2folder["chapters"], 'chapters_vmr_train.json'), 'w') as outfile:
    for item in out:
        if item['split'] == 'train':
            json.dump(item, outfile)
            outfile.write('\n')

with open(os.path.join(DATA_DIR, name2folder["chapters"], 'chapters_vmr_val.json'), 'w') as outfile:
    for item in out:
        if item['split'] == 'val':
            json.dump(item, outfile)
            outfile.write('\n')

with open(os.path.join(DATA_DIR, name2folder["chapters"], 'chapters_vmr_test.json'), 'w') as outfile:
    for item in out:
        if item['split'] == 'test':
            json.dump(item, outfile)
            outfile.write('\n')