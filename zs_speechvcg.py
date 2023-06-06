from model import TextTilingTokenizer
import os
import torch as th
from torch.utils.data import Dataset
import json
import pickle
import argparse
import random
from args import get_args_parser
from torch.utils.data import DataLoader, DistributedSampler
from util.metrics import MetricLogger
from util import dist
from functools import reduce
import gc
import math
from dvc_eval import eval_dvc, eval_soda
from transformers import LlamaForCausalLM, LlamaTokenizer
from args import NLTK_FOLDER


class DenseVideoCaptioning_Dataset(Dataset):
    def __init__(
            self,
            json_path,
            subtitles_path=None,
    ):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())
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

    def __getitem__(self, idx):
        video_id = self.vids[idx]

        # get subtitles
        if (self.subs is not None and video_id[-11:] in self.subs) or (
                self.subs_path is not None and os.path.exists(os.path.join(self.subs_path, video_id[-11:] + '.pkl'))):
            if (self.subs is not None and video_id[-11:] in self.subs):
                sub = self.subs[video_id[-11:]]
            else:
                sub = pickle.load(open(os.path.join(self.subs_path, video_id[-11:] + '.pkl'), 'rb'))
        else:
            sub = {"start": [], "end": [], "text": []}

        return {
            "video_id": video_id,
            "sub": sub,
        }

def custom_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    sub = [batch[i]["sub"] for i in range(bs)]
    return {
        "video_id": video_id,
        "sub": sub,
    }

def build_densevideocaptioning_dataset(dataset_name, split, args):
    if dataset_name == "youcook":
        if split == "train":
            json_path = args.youcook_train_json_path
        elif split == "val":
            json_path = args.youcook_val_json_path
        else:
            raise NotImplementedError
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
        subtitles_path = args.chapters_subtitles_path
    else:
        raise NotImplementedError
    return DenseVideoCaptioning_Dataset(json_path=json_path,
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

# nltk.download('stopwords', download_dir=NLTK_FOLDER)
os.environ["NLTK_DATA"] = NLTK_FOLDER
tokenizer = TextTilingTokenizer(w=50)
device = th.device(args.device)
model_tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
model_tokenizer.pad_token = "<s>"
model = LlamaForCausalLM.from_pretrained(args.model_name).half()
model.to(device)
model.eval()

@th.no_grad()
def evaluate(
    model,
    model_tokenizer,
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
        subs = batch_dict["sub"]
        for vid, sub in zip(vids, subs):
            # segment
            sentences = [x.capitalize() + "." for x in sub['text']]
            paragraphs = ['\n'.join(sentences[i:i + 2]) for i in range(0, len(sentences), 2)]
            try:
                sections = tokenizer.tokenize('\n\n'.join(paragraphs))
            except:
                res[vid] = []
                continue

            segments = []
            for section in sections:
                start, end = float('inf'), 0
                for st, ed, txt in zip(sub["start"], sub["end"], sub["text"]):
                    if txt.strip() in section:
                        start = min(start, st)
                        end = max(end, ed)
                segments.append({"text": section, "start": start, "end": end})

            # generate title
            if args.random:
                sentences = []
                for segment in segments:
                    texts = segment['text'].split('\n')
                    sentence = random.choice(texts)
                    sentences.append(sentence)
                res[vid] = [{"sentence": sentence, "timestamp": [st, ed]} for sentence, st, ed in zip(sentences, [x["start"] for x in segments], [x["end"] for x in segments])]
            else:
                prompts = []
                for segment in segments:
                    text = segment['text'].replace('\n', '').strip()
                    if text and text[-1] != ".":
                        text + text + '.'
                    prompts.append(f"Summarize the following speech transcript in a chapter title. Transcript:{text} Chapter title:")
                bs = 8
                n_batches = math.ceil(len(prompts) / bs)
                chapters = []
                starts = [x["start"] for x in segments]
                ends = [x["end"] for x in segments]
                for i in range(n_batches):
                    prompts_tokenized = model_tokenizer(prompts[i * bs: (i + 1) * bs], padding="longest", truncation=True, max_length=512, return_tensors="pt").to(device)
                    output = model.generate(prompts_tokenized.input_ids, max_new_tokens=20)
                    output_text = model_tokenizer.batch_decode(output.detach().cpu(), skip_special_tokens=True)
                    chapters.extend([{"sentence": title[len(prompt):], "timestamp": [st, ed]} for title, prompt, st, ed in zip(output_text, prompts[i * bs: (i + 1) * bs], starts[i * bs: (i + 1) * bs], ends[i * bs: (i + 1) * bs])])
                    del output_text
                    del output
                    del prompts_tokenized
                    gc.collect()
                    th.cuda.empty_cache()
                res[vid] = chapters

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    metrics = {}
    if dist.is_main_process():
        if args.save_dir:
            pred_path = os.path.join(args.save_dir, dataset_name + f"_{split}_preds.json",)
            json.dump({'results': results}, open(pred_path, "w",))
        else:
            pred_path = {'results': results}
        if dataset_name == "youcook":
            references = [args.youcook_val_json_path]
        elif dataset_name == "vitt":
            references = [args.vitt_val_json_path if split == "val" else args.vitt_test_json_path]
        elif dataset_name == "chapters":
            references = [args.chapters_val_json_path if split == "val" else args.chapters_test_json_path]
        else:
            raise NotImplementedError
        metrics.update(eval_dvc(pred_path, references, tious=[0.3, 0.5, 0.7, 0.9], max_proposals_per_video=1000, verbose=False, no_lang_eval=False))
        metrics.update(eval_soda(pred_path, references, verbose=False))
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    metrics = dist.all_gather(metrics)
    metrics = reduce(lambda a, b: a.update(b) or a, metrics, {})

    return metrics

with th.no_grad():
    evaluate(model=model,
             model_tokenizer=model_tokenizer,
             data_loader=dataloader,
             device=device,
             dataset_name=args.combine_datasets_val[0],
             args=args,
             split="test",
             )