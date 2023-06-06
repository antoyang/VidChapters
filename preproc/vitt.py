from args import DATA_DIR, name2folder  # to launch in $WORK/vid2seq
import json
import os
import torch
import subprocess
from tqdm import tqdm
from args import DATA_DIR


with open(os.path.join(DATA_DIR, name2folder['vitt'], 'ViTT-annotations.json')) as f:
    annotations = [json.loads(x) for x in f]
id_mapping = json.load(open(os.path.join(DATA_DIR, name2folder['vitt'], 'id_mapping.json'), 'r'))
train_ids = set(open(os.path.join(DATA_DIR, name2folder['vitt'], 'train_id.txt')).read().split('\n')[:-1])
val_ids = set(open(os.path.join(DATA_DIR, name2folder['vitt'], 'dev_id.txt')).read().split('\n')[:-1])
test_ids = set(open(os.path.join(DATA_DIR, name2folder['vitt'], 'test_id.txt')).read().split('\n')[:-1])
features = torch.load(os.path.join(DATA_DIR, name2folder['vitt'], 'clipvitl14.pth'))
train = {}
val = {}
test = {}
counter = {}

for x in tqdm(annotations):
    vid = x['id']
    if vid not in id_mapping:
        continue
    video_id = id_mapping[vid]
    if video_id not in features:
        continue
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
         os.path.join(DATA_DIR, name2folder["vitt"], "videos", + video_id + '.mp4')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = float(result.stdout)
    start = [float(x['timestamp']) for x in x['annotations']]
    end = start[1:] + [duration * 1000]
    out = {'duration': duration,
           'timestamps': [[st / 1000, ed / 1000] for st, ed in zip(start, end)],
           'sentences': [x['tag'] for x in x['annotations']],
           'path': video_id + '.mp4.npy'}
    if vid in train_ids:
        if any([y[1] > out['duration'] for y in out['timestamps']]) or any([x.strip() == "" for x in out['sentences']]):
            continue
        if not all([ed > st for st, ed in zip(start, end)]):
            continue
        train[video_id] = out
    elif vid in val_ids:
        if vid not in counter:
            counter[vid] = 0
        val[str(counter[vid]) + '_' + video_id] = out
        counter[vid] += 1
    elif vid in test_ids:
        if vid not in counter:
            counter[vid] = 0
        test[str(counter[vid]) + '_' + video_id] = out
        counter[vid] += 1
    else:
        raise NotImplementedError

for vid in counter:
    if counter[vid] > 3:
        if '0_' + vid in val:
            for j in range(counter[vid]):
                del val[str(j) + '_' + vid]
        else:
            for j in range(counter[vid]):
                del test[str(j) + '_' + vid]

rem = []
for x in val:
    out = val[x]
    if any([y[1] > out['duration'] for y in out['timestamps']]) or any([x.strip() == "" for x in out['sentences']]):
        rem.append(x)
    elif not all([x[1] > x[0] for x in out['timestamps']]):
        rem.append(x)
val = {x: val[x] for x in val if x not in rem}
print(len(rem))

rem = []
for x in test:
    out = test[x]
    if any([y[1] > out['duration'] for y in out['timestamps']]) or any([x.strip() == "" for x in out['sentences']]):
        rem.append(x)
    elif not all([x[1] > x[0] for x in out['timestamps']]):
        rem.append(x)
test = {x: test[x] for x in test if x not in rem}
print(len(rem))

print(len(train), len(val), len(test))
json.dump(train, open(os.path.join(DATA_DIR, name2folder['vitt'], 'train.json'), 'w'))
json.dump(val, open(os.path.join(DATA_DIR, name2folder['vitt'], 'dev.json'), 'w'))
json.dump(test, open(os.path.join(DATA_DIR, name2folder['vitt'], 'test.json'), 'w'))