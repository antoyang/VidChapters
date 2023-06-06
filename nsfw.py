import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from detoxify import Detoxify
import math
import collections
from args import DATA_DIR, SSD_DIR, name2folder, MODEL_DIR


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            pkl_path,
            features_path,
    ):
        """
        Args:
        """
        self.data = pickle.load(open(pkl_path, 'rb'))
        self.videos = list(self.data)
        self.features_path = features_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid = self.videos[idx]
        item = self.data[vid]
        chapters = item["chapters"]
        text = [x["label"] for x in chapters]
        features_path = os.path.join(self.features_path, vid + '.npy')
        if not os.path.exists(features_path):
            features_path = os.path.join(self.features_path, vid + '.mp4.npy')
        if not os.path.exists(features_path):
            raise NotImplementedError
        features = torch.from_numpy(np.load(features_path))

        return {'vid': vid,
                'features': features,
                'text': text}

def custom_collate_fn(batch):
    return {
        x: batch[0][x] for x in batch[0]
    }

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def load_safety_model():
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel

    model_dir = os.path.join(MODEL_DIR, "clip_autokeras_binary_nsfw")

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10**3, 768).astype("float32"), batch_size=10**3)

    return loaded_model

model = load_safety_model()
detox = Detoxify('multilingual', device='cuda')

dataset = VideoLoader(
    os.path.join(DATA_DIR, name2folder["chapters"], 'chapters.pkl'),
    os.path.join(SSD_DIR, "chapters_clipvitl14_features")
	)
n_dataset = len(dataset)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=24,
    collate_fn=custom_collate_fn
)

output_path = os.path.join(SSD_DIR, "nsfw")
asr_path = os.path.join(SSD_DIR, "align")
asr_files = set(os.listdir(asr_path))
with torch.no_grad():
    for k, data in tqdm(enumerate(loader)):
        vid = data["vid"]
        if os.path.exists(os.path.join(output_path, vid+'.pkl')):
            continue
        features = np.asarray( normalized(data["features"].numpy()))
        predictions = model.predict_on_batch(features)
        frames_nsfw = np.hstack(predictions)
        text = data['text']
        chapters_tox = detox.predict(text)
        if vid + '.pkl' in asr_files:
            asr = pickle.load(open(os.path.join(asr_path, vid + '.pkl'), 'rb'))
            asr_text = [asr['segments'][i]['text'] for i in range(len(asr['segments'])) if asr['segments'][i]['text'].strip()]
        else:
            asr_text = None
        if asr_text is not None:
            bs = 32
            tox_asr = collections.defaultdict(list)
            n_batches = math.ceil(len(asr_text) / bs)
            for i in range(n_batches):
                tmp = detox.predict(asr_text[i * bs: (i + 1) * bs])
                for x in tmp:
                    tox_asr[x].extend(tmp[x])
        else:
            tox_asr = None

        pickle.dump({'nsfw_frames': frames_nsfw,
                     'chapters_toxicity': chapters_tox,
                     'asr_toxicity': tox_asr} if tox_asr else {'nsfw_frames': frames_nsfw,
                     'chapters_toxicity': chapters_tox},
                    open(os.path.join(output_path, vid+'.pkl'), 'wb'))
