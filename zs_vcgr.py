import os
import torch as th
from torch.utils.data import Dataset
import json
import pickle
import numpy as np
import argparse
import random
from args import get_args_parser
from torch.utils.data import DataLoader, DistributedSampler
from util.metrics import MetricLogger
from util import dist
from functools import reduce
from transformers import BertModel, BertTokenizer
import clip
import torch.nn.functional as F
from args import MODEL_DIR


def iou(interval_1, interval_2):
    start_i, end_i = interval_1[0], interval_1[1]
    start, end = interval_2[0], interval_2[1]
    intersection = max(0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
    iou = float(intersection) / (union + 1e-8)
    return iou

def evaluate_detection(results, tiou):
    recall = []
    for vid_id, cur in results.items():
        for pred, gt in zip(cur['pred'], cur['gt']):
            recall.append(int(iou(pred, gt) > tiou))
    return sum(recall) / len(recall)

def evaluate_navigation(results, tiou):
    recall = []
    for vid_id, cur in results.items():
        for pred, gt in zip(cur['pred'], cur['gt']):
            recall.append(abs(pred[0] - gt[0]) < tiou)
    return sum(recall) / len(recall)

def evaluate_predictions(results, tious=[0.3, 0.5, 0.7, 0.9], distances=[1, 3, 5, 10]):
    scores = {}
    for tiou in tious:
        scores["Recall@" + str(tiou)] = evaluate_detection(results, tiou)
    for dist in distances:
        scores["Recall@" + str(dist) + "s"] = evaluate_navigation(results, dist)
    return scores

class DenseVideoCaptioning_Dataset(Dataset):
    def __init__(
            self,
            json_path,
            features_path,
            max_feats=100,
            features_dim=768,
            subtitles_path=None,
    ):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())
        self.features = None
        self.features_path = None
        if os.path.isdir(features_path):
            self.features_path = features_path
        else:
            self.features = th.load(features_path)
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

    def __len__(self):
        return len(self.data)

    def _get_text(self, text):
        text = text.strip()
        text = text.capitalize()
        if text[-1] != '.':
            text = text + '.'
        return text

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

        vis_timestamps = []
        if len(video) >= self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
                vis_timestamps.append((j * len(video)) // self.max_feats)
            video = th.stack(sampled)
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
            vis_timestamps = [min(j, video_len) for j in range(self.max_feats)]

        return video, vis_timestamps

    def __getitem__(self, idx):
        video_id = self.vids[idx]
        video, vis_timestamps = self._get_video(video_id[-11:])

        # get subtitles
        if (self.subs is not None and video_id[-11:] in self.subs) or (
                self.subs_path is not None and os.path.exists(os.path.join(self.subs_path, video_id[-11:] + '.pkl'))):
            if (self.subs is not None and video_id[-11:] in self.subs):
                sub = self.subs[video_id[-11:]]
            else:
                sub = pickle.load(open(os.path.join(self.subs_path, video_id[-11:] + '.pkl'), 'rb'))
        else:
            sub = {"start": [], "end": [], "text": []}

        annotations = self.data[video_id]
        captions = [self._get_text(x) for x in annotations['sentences']]

        return {
            "video_id": video_id,
            "video": video,
            "sub": sub,
            "query": captions,
            "timestamps": annotations['timestamps'],
            "duration": annotations["duration"],
            "vis_timestamps": vis_timestamps
        }

def custom_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    video = th.stack([batch[i]["video"] for i in range(bs)])
    sub = [batch[i]["sub"] for i in range(bs)]
    query = [batch[i]["query"] for i in range(bs)]
    timestamps = [batch[i]["timestamps"] for i in range(bs)]
    vis_timestamps = [batch[i]["vis_timestamps"] for i in range(bs)]
    duration = [batch[i]["duration"] for i in range(bs)]
    return {
        "video_id": video_id,
        "video": video,
        "sub": sub,
        "query": query,
        "timestamps": timestamps,
        "vis_timestamps": vis_timestamps,
        "duration": duration
    }

def build_densevideocaptioning_dataset(dataset_name, split, args):
    if dataset_name == "youcook":
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
    return DenseVideoCaptioning_Dataset(json_path=json_path,
                                        features_path=features_path,
                                        max_feats=args.max_feats,
                                        features_dim=args.features_dim,
                                        subtitles_path=subtitles_path)

parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args()
if args.save_dir:
    args.save_dir = os.path.join(args.presave_dir, args.save_dir)
if dist.is_main_process():
    if args.save_dir and not (os.path.isdir(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir), exist_ok=True)
dist.init_distributed_mode(args)
dataset = build_densevideocaptioning_dataset(args.combine_datasets_val[0], "test" if args.combine_datasets_val[0] in ["vitt", "chapters"] else "val", args)
sampler = DistributedSampler(dataset, shuffle=False)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size_val,
    sampler=sampler,
    collate_fn=custom_collate_fn,
    num_workers=args.num_workers,
)

device = th.device(args.device)
if not args.use_video:
    model = BertModel.from_pretrained('bert-base-uncased').to(device).half()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
else:
    model, _ = clip.load("ViT-L/14", download_root=MODEL_DIR, device=device)
    tokenizer = None
threshold = 0.05

@th.no_grad()
def evaluate(
    model,
    tokenizer,
    data_loader,
    device: th.device,
    args,
    split="test",
    dataset_name="chapters"
):
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        vids = batch_dict["video_id"]
        videos = batch_dict["video"]
        subs = batch_dict["sub"]
        queries = batch_dict["query"]
        gts = batch_dict["timestamps"]
        durations = batch_dict["duration"]

        vis_timestamps = batch_dict["vis_timestamps"]
        preds = []
        for vid, video, sub, query, vis_se, gt, dur in zip(vids, videos, subs, queries, vis_timestamps, gts, durations):
            # segment
            if args.random:
                if len(sub['start']):
                    idx = random.randint(0, len(sub['start']) - 1)
                    res[vid] = {'pred': [[sub['start'][idx], sub['end'][idx]] * len(gt)],
                                'gt': gt}
                else:
                    s = float(random.randint(0, int(dur)))
                    e = float(random.randint(s, int(dur)))
                    res[vid] = {'pred': [[s, e] * len(gt)],
                                'gt': gt}
                continue
            if not args.use_video:
                if not sub["text"]:
                    s = float(random.randint(0, int(dur)))
                    e = float(random.randint(s, int(dur)))
                    res[vid] = {'pred': [[s, e] * len(gt)],
                                'gt': gt}
                else:
                    tokens = tokenizer(query, max_length=256, truncation=True, padding="longest", add_special_tokens=True, return_tensors="pt").to(device)
                    text = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])['last_hidden_state'][:, 0]
                    asrt = tokenizer(sub["text"], max_length=256, truncation=True, padding="longest", add_special_tokens=True, return_tensors="pt").to(device)
                    asr = model(asrt['input_ids'], attention_mask=asrt['attention_mask'])['last_hidden_state'][:, 0]
                    sim = F.normalize(text @ (asr.t()))
                    max_sim = th.max(sim, 1)
                    res[vid] = {'pred': [[sub["start"][idx.item()], sub["end"][idx.item()]] for idx in max_sim.indices],
                                'gt': gt}
                continue
            tokens = clip.tokenize(query, truncate=True).to(device)
            text = model.encode_text(tokens).float()  # N D and vid is L D
            sim = F.normalize(text @ (video.to(device).t()))  # N L
            max_sim = th.max(sim, 1)
            max_values, start_indexes = max_sim.values, max_sim.indices
            for i, (max_val, start_idx) in enumerate(zip(max_values, start_indexes)):
                idx = start_idx + 1
                while idx < args.max_feats and sim[i][idx] >= sim[i][start_idx] - threshold:
                    idx += 1
                preds.append([float(vis_se[start_idx]), float(vis_se[idx]) if idx < len(vis_se) else dur])
            res[vid] = {'pred': preds,
                        'gt': gt}

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    metrics = {}
    if dist.is_main_process():
        if args.save_dir:
            pred_path = os.path.join(args.save_dir, dataset_name + f"_{split}_preds.json",)
            json.dump({'results': results}, open(pred_path, "w",))
        metrics.update(evaluate_predictions(results))
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    metrics = dist.all_gather(metrics)
    metrics = reduce(lambda a, b: a.update(b) or a, metrics, {})

    return metrics

with th.no_grad():
    evaluate(model=model,
             tokenizer=tokenizer,
             data_loader=dataloader,
             device=device,
             dataset_name=args.combine_datasets_val[0],
             args=args,
             split="test",
             )