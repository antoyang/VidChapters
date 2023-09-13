import os
import torch as th
from torch.utils.data import Dataset
import json
import pickle
import random
import numpy as np
import ffmpeg
from args import DATA_DIR, name2folder


def _get_output_dim(h, w, resolution):
    if h >= w:
        return int(h * resolution / w), resolution
    else:
        return resolution, int(w * resolution / h)


def get_raw_video(video_path, resolution):
    try:
        # get metadata
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        h = int(video_stream['height'])
        w = int(video_stream['width'])
        height, width = _get_output_dim(h, w, resolution)

        # ffmpeg decoding
        cmd = (
            ffmpeg
                .input(video_path)
                .filter('fps', fps=1)
                .filter("scale", width, height)  # resize
        )
        # center crop
        x = int((width - resolution) / 2.0)
        y = int((height - resolution) / 2.0)
        cmd = cmd.crop(x, y, resolution, resolution)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )
        frames = np.frombuffer(out, np.uint8).reshape([-1, resolution, resolution, 3])
        frames = th.from_numpy(np.copy(frames))
        # T H W C -> T C H W.
        video = frames.permute(0, 3, 1, 2)
    except:
        video = th.zeros(1, 3, 224, 224)

    return video

class VideoCaptioning_Dataset(Dataset):
    def __init__(
        self,
        json_path,
        features_path,
        videos_path,
        training=True,
        max_feats=100,
        features_dim=768,
        subtitles_path=None,
        random=False
    ):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())
        self.features = None
        self.features_path = None
        if os.path.isdir(features_path):
            self.features_path = features_path
        else:
            self.features = th.load(features_path)
        if videos_path is not None:
            self.vid2path = json.load(open(videos_path, 'r'))
        else:
            self.vid2path = None
        self.training = training
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.subs = None
        self.subs_path = None
        if subtitles_path and os.path.exists(subtitles_path) and os.path.isdir(subtitles_path):
            self.subs_path = subtitles_path
        elif subtitles_path and os.path.exists(subtitles_path):
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            print("No subtitles given or found.")
        self.random = random

    def __len__(self):
        return len(self.data)

    def _get_text(self, text):
        text = text.strip()
        text = text.capitalize()
        if text[-1] != '.':
            text = text + '.'
        return text

    def _get_raw(self, path):
        return get_raw_video(path, 224)

    def _get_video(self, video_id):
        if self.features is not None:
            assert video_id in self.features, video_id
            video = self.features[video_id].float()
        else:
            features_path = os.path.join(self.features_path, video_id + '.mp4.npy')
            if not os.path.exists(features_path):
                features_path = os.path.join(self.features_path, video_id + '.npy')
            assert os.path.exists(features_path), features_path
            video = th.from_numpy(np.load(features_path)).float()

        return video

    def pad_video(self, video):
        if self.max_feats == 1:
            tmp = video[len(video) // 2: len(video) // 2 + 1]
            if len(tmp):
                return video[len(video) // 2: len(video) // 2 + 1]  # middle frame
            else:
                return th.zeros(self.max_feats, self.features_dim)
        if len(video) >= self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        return video

    def pad_raw_video(self, video):
        if self.max_feats == 1:
            tmp = video[len(video) // 2: len(video) // 2 + 1]
            if len(tmp):
                return tmp  # middle frame
            else:
                return th.zeros(self.max_feats, 3, 224, 224)
        if len(video) >= self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, 3, 224, 224)], 0
            )
        return video

    def __getitem__(self, idx):
        video_id = self.vids[idx]
        annotations = self.data[video_id]
        video = self._get_video(video_id[-11:])
        if self.training:
            idx = random.randint(0, len(annotations["sentences"]) - 1)
            sentence = annotations["sentences"][idx]
            timestamp = annotations["timestamps"][idx]
            start, end = timestamp
            video = self.pad_video(video[int(start): int(end) + 1])
        else:
            video = th.stack([self.pad_video(video[int(x[0]): int(x[1]) + 1]) for x in annotations["timestamps"]])

        # get subtitles
        if (self.subs is not None and video_id[-11:] in self.subs) or (self.subs_path is not None and os.path.exists(os.path.join(self.subs_path, video_id + '.pkl'))):
            if (self.subs is not None and video_id[-11:] in self.subs):
                sub = self.subs[video_id[-11:]]
            else:
                sub = pickle.load(open(os.path.join(self.subs_path, video_id[-11:] + '.pkl'), 'rb'))

            # keep only those in timestamps
            if self.training:
                to_keep = [(x >= start and y <= end) for x, y in zip(sub["start"], sub["end"])]
                text = ' '.join([self._get_text(x) for i, x in enumerate(sub['text']) if to_keep[i]])
            else:
                text = []
                for i in range(len(annotations["sentences"])):
                    to_keep = [(x >= annotations["timestamps"][i][0] and y <= annotations["timestamps"][i][1]) for x, y in zip(sub["start"], sub["end"])]
                    if self.random:
                        tmp = [self._get_text(x) for i, x in enumerate(sub['text']) if to_keep[i]]
                        txt = '' if not tmp else random.choice(tmp)
                    else:
                        txt = ' '.join([self._get_text(x) for i, x in enumerate(sub['text']) if to_keep[i]])
                    text.append(txt)
        else:
            if self.training:
                text = ''
            else:
                text = ['' for _ in annotations["sentences"]]

        # get annotations
        if self.training:
            caption = self._get_text(sentence)
        else:
            caption = [self._get_text(x) for x in annotations['sentences']]

        out = {
            "video_id": video_id,
            "video": video,
            "input_text": text,
            "output_text": caption,
        }
        if self.vid2path is not None:
            raw_video = self._get_raw(self.vid2path.get(video_id[-11:], None))
            if self.training:
                raw_video = self.pad_raw_video(raw_video[int(start): int(end) + 1])
            else:
                raw_video = th.stack(
                    [self.pad_raw_video(raw_video[int(x[0]): int(x[1]) + 1]) for x in annotations["timestamps"]])
            out['raw_video'] = raw_video
        return out


def videocaptioning_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    video = th.stack([batch[i]["video"] for i in range(bs)])
    input_text = [batch[i]["input_text"] for i in range(bs)]
    output_text = [batch[i]["output_text"] for i in range(bs)]
    out = {
        "video_id": video_id,
        "video": video,
        "input_text": input_text,
        "output_text": output_text,
    }
    if 'raw_video' in batch[0]:
        out["raw_video"] = th.stack([batch[i]["raw_video"] for i in range(bs)])
    return out


def build_videocaptioning_dataset(dataset_name, split, args):
    if dataset_name == "activitynet":
        if split == "train":
            json_path = args.activitynet_train_json_path
        elif split == "val":
            json_path = args.activitynet_val2_json_path  # val 2 videos contain val 1 videos
        else:
            raise NotImplementedError
        features_path = args.activitynet_features_path
        subtitles_path = args.activitynet_subtitles_path
    elif dataset_name == "youcook":
        if split == "train":
            json_path = args.youcook_train_json_path
        elif split == "val":
            json_path = args.youcook_val_json_path
        else:
            raise NotImplementedError
        features_path = args.youcook_features_path
        subtitles_path = args.youcook_subtitles_path
    elif dataset_name == "vitt":
        if split == "train":
            json_path = args.vitt_train_json_path
        elif split == "val":
            json_path = args.vitt_val_json_path
        elif split == "test":
            json_path = args.vitt_test_json_path
        else:
            raise NotImplementedError
        features_path = args.vitt_features_path
        subtitles_path = args.vitt_subtitles_path
    elif dataset_name == "chapters":
        if split == "train":
            json_path = args.chapters_train_json_path
        elif split == "val":
            json_path = args.chapters_val_json_path
        elif split == "test":
            json_path = args.chapters_test_json_path
        else:
            raise NotImplementedError
        features_path = args.chapters_features_path
        subtitles_path = args.chapters_subtitles_path
    else:
        raise NotImplementedError
    return VideoCaptioning_Dataset(json_path=json_path,
                                   features_path=features_path,
                                   videos_path=os.path.join(DATA_DIR, name2folder[dataset_name], "video_paths.json") if "blip" in args.model_name else None,
                                   training=split=="train",
                                   max_feats=args.max_feats,
                                   features_dim=args.features_dim,
                                   subtitles_path=subtitles_path,
                                   random=args.random)
