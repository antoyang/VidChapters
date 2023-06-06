from args import DATA_DIR, name2folder
import json
import os
import torch

features = torch.load(os.path.join(DATA_DIR, name2folder['youcook'], 'clipvitl14.pth'))
annotations = json.load(open(os.path.join(DATA_DIR, name2folder['youcook'], 'youcookii_annotations_trainval.json'), 'r'))
train = {}
val = {}
for video, anns in annotations['database'].items():
    if video not in features:
        continue
    if anns['subset'] == 'training':
        train[video] = {'duration': anns['duration'],
                        'timestamps': [x['segment'] for x in anns['annotations']],
                        'sentences': [x['sentence'] for x in anns['annotations']]}
    elif anns['subset'] == 'validation':
        val[video] = {'duration': anns['duration'],
                      'timestamps': [x['segment'] for x in anns['annotations']],
                      'sentences': [x['sentence'] for x in anns['annotations']]}
    else:
        raise NotImplementedError
json.dump(train, open(os.path.join(DATA_DIR, name2folder['youcook'], 'train.json'), 'w'))
json.dump(val, open(os.path.join(DATA_DIR, name2folder['youcook'], 'val.json'), 'w'))