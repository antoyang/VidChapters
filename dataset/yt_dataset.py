import os
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
from util.t5 import create_sentinel_ids, filter_input_ids, random_spans_noise_mask


class YT_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        features_path,
        subtitles_path,
        max_feats=100,
        features_dim=768,
        tokenizer=None,
        num_bins=100,
        max_input_tokens=1000,
        max_output_tokens=1000,
        noise_density=0.25,
        mean_noise_span_length=5,
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.subtitles_path = subtitles_path
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.tokenizer = tokenizer
        self.num_bins = num_bins
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.num_text_tokens = len(tokenizer) - num_bins
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

    def __len__(self):
        return len(self.data)

    def _get_text(self, text):
        text = text.strip()
        text = text.capitalize()
        if text[-1] != '.':
            text = text + '.'
        return text

    def _get_video(self, video_path, sub):
        features_path = os.path.join(self.features_path, video_path)
        assert os.path.exists(features_path), features_path
        video = th.from_numpy(np.load(features_path)).float()

        if "duration" not in sub:
            sub["duration"] = len(video) + 1
        duration = sub["duration"]
        to_keep = [x >= 0 and y <= duration for x, y in zip(sub["start"], sub["end"])]
        sub["text"] = [x for i, x in enumerate(sub["text"]) if to_keep[i]]
        sub["start"] = [max(x, 0) for i, x in enumerate(sub["start"]) if to_keep[i]]
        sub["end"] = [min(x, duration) for i, x in enumerate(sub["end"]) if to_keep[i]]

        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        else:
            video_len = self.max_feats

        return video, sub

    def time_tokenize(self, x, duration, num_bins):
        time_token = int(float((num_bins - 1) * x) / float(duration))
        assert time_token <= self.num_bins
        return self.num_text_tokens + time_token

    def __getitem__(self, idx):
        video_id = self.data["video_id"][idx]
        subtitles_path = os.path.join(self.subtitles_path, video_id + '.pkl')
        assert os.path.exists(subtitles_path), subtitles_path
        sub = pickle.load(open(subtitles_path, 'rb'))

        # get video
        video, sub = self._get_video(self.data["video_path"][idx], sub)
        duration = sub['duration']

        # get subtitles
        sub['text'] = [self._get_text(x) for x in sub['text']]
        time_input_tokens = [th.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                            self.time_tokenize(ed, duration, self.num_bins)])
                             for st, ed in zip(sub['start'], sub['end'])]
        text_input_tokens = [self.tokenizer(x, add_special_tokens=False, max_length=self.max_input_tokens,
                                            padding="do_not_pad", truncation=True, return_tensors="pt",)['input_ids'][0]
                             for x in sub['text']]
        input_tokens = [th.cat([ti, te], 0) for ti, te in zip(time_input_tokens, text_input_tokens)]

        # denoising sequence
        if input_tokens:
            input_tokens = th.cat(input_tokens, 0)
            input_tokens = input_tokens[:self.max_input_tokens - 1]
            input_tokens = th.cat([input_tokens, th.LongTensor([self.tokenizer.eos_token_id])], 0)
            mask_indices = np.asarray(
                [random_spans_noise_mask(len(input_tokens), self.noise_density, self.mean_noise_span_length)])
            labels_mask = ~mask_indices

            input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), self.tokenizer, self.num_bins)
            labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), self.tokenizer, self.num_bins)

            denoising_output_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), labels_sentinel, self.tokenizer)).squeeze(0)
            denoising_input_tokens = th.from_numpy(
                filter_input_ids(input_tokens.unsqueeze(0).numpy(), input_ids_sentinel, self.tokenizer)).squeeze(0)
        else:
            print(video_id)
            input_tokens = th.LongTensor([self.tokenizer.eos_token_id])
            denoising_input_tokens = th.LongTensor([0])
            denoising_output_tokens = input_tokens

        return {
            "video_id": video_id,
            "duration": duration,
            "video": video,
            "output_tokens": input_tokens,
            "denoising_input_tokens": denoising_input_tokens,
            "denoising_output_tokens": denoising_output_tokens,
        }


def yt_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    duration = [batch[i]["duration"] for i in range(bs)]
    video = th.stack([batch[i]["video"] for i in range(bs)])
    denoising_input_tokens = [batch[i]["denoising_input_tokens"] for i in range(bs)]
    max_input_len = max(len(x) for x in denoising_input_tokens)
    for i in range(bs):
        if len(denoising_input_tokens[i]) < max_input_len:
            denoising_input_tokens[i] = th.cat([denoising_input_tokens[i], th.zeros(max_input_len - len(denoising_input_tokens[i])).long()], 0)
    denoising_input_tokens = th.stack(denoising_input_tokens)
    denoising_output_tokens = [batch[i]["denoising_output_tokens"] for i in range(bs)]
    max_denoising_output_len = max(len(x) for x in denoising_output_tokens)
    for i in range(bs):
        if len(denoising_output_tokens[i]) < max_denoising_output_len:
            denoising_output_tokens[i] = th.cat([denoising_output_tokens[i], th.zeros(max_denoising_output_len - len(denoising_output_tokens[i])).long()], 0)
    denoising_output_tokens = th.stack(denoising_output_tokens)
    output_tokens = [batch[i]["output_tokens"] for i in range(bs)]
    max_output_len = max(len(x) for x in output_tokens)
    for i in range(bs):
        if len(output_tokens[i]) < max_output_len:
            output_tokens[i] = th.cat([output_tokens[i], th.zeros(max_output_len - len(output_tokens[i])).long()], 0)
    output_tokens = th.stack(output_tokens)
    out = {
        "video_id": video_id,
        "duration": duration,
        "video": video,
        "denoising_input_tokens": denoising_input_tokens,
        "denoising_output_tokens": denoising_output_tokens,
        "output_tokens": output_tokens,
    }
    return out


def build_yt_dataset(dataset_name, split, args, tokenizer):
    if dataset_name == "htm":
        if split == "train":
            csv_path = args.howto100m_train_csv_path
        else:
            raise NotImplementedError
        features_path = args.howto100m_features_path
        subtitles_path = args.howto100m_subtitles_path
    else:
        raise NotImplementedError
    return YT_Dataset(csv_path=csv_path,
                      features_path=features_path,
                      max_feats=args.max_feats,
                      features_dim=args.features_dim,
                      tokenizer=tokenizer,
                      subtitles_path=subtitles_path,
                      num_bins=args.num_bins,
                      max_input_tokens=args.max_input_tokens,
                      max_output_tokens=args.max_output_tokens,
                      noise_density=args.mask_prob,
                      mean_noise_span_length=args.mask_len)
