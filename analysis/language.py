import torch
from torch.utils.data import Dataset
import os
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from args import DATA_DIR, name2folder, SSD_DIR
from langdetect import detect


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            pkl_path,
    ):
        """
        Args:
        """
        self.data = pickle.load(open(pkl_path, 'rb'))
        self.videos = list(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid = self.videos[idx]
        item = self.data[vid]
        chapters = item["chapters"]
        text = [x["label"] for x in chapters]

        return {'vid': vid,
                'text': text}

def custom_collate_fn(batch):
    return {
        x: batch[0][x] for x in batch[0]
    }

dataset = VideoLoader(
    os.path.join(DATA_DIR, name2folder["chapters"], "chapters.pkl")
)
n_dataset = len(dataset)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=24,
    collate_fn=custom_collate_fn
)

output_path = os.path.join(SSD_DIR, "asr_lang")
# Put path to chapters ot detect the language of chapters
asr_path = os.path.join(SSD_DIR, "allchapters_asr")
asr_files = set(os.listdir(asr_path))
with torch.no_grad():
    for k, data in tqdm(enumerate(loader)):
        vid = data["vid"]
        if os.path.exists(os.path.join(output_path, vid+'.pkl')):
            continue
        if vid + '.pkl' in asr_files:
            asr = pickle.load(open(os.path.join(asr_path, vid + '.pkl'), 'rb'))
            asr_text = [asr['text'][i] for i in range(len(asr['text'])) if
                        asr['text'][i].strip()]
            try:
                lang = detect(' '.join(asr_text))
            except:
                lang = "error"
        else:
            lang = None

        pickle.dump({'asr_lang': lang},
                    open(os.path.join(output_path, vid+'.pkl'), 'wb'))
