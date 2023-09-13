import subprocess
import os
import torch as th
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
from args import get_args_parser
from torch.utils.data import DataLoader, DistributedSampler
from util.metrics import MetricLogger
from util import dist
from functools import reduce
import gc
import math
from dvc_eval import eval_dvc, eval_soda
import ffmpeg
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from args import DATA_DIR, name2folder

def extract_shots_with_ffprobe(src_video, threshold=0.3):
    """
    uses ffprobe to produce a list of shot
    boundaries (in seconds)

    Args:
        src_video (string): the path to the source
            video
        threshold (float): the minimum value used
            by ffprobe to classify a shot boundary

    Returns:
        List[(float, float)]: a list of tuples of floats
        representing predicted shot boundaries (in seconds) and
        their associated scores
    """
    scene_ps = subprocess.Popen(("ffprobe",
                                 "-show_frames",
                                 "-of",
                                 "compact=p=0",
                                 "-f",
                                 "lavfi",
                                 "movie=" + src_video + ",select=gt(scene\," + str(threshold) + ")"),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    output = scene_ps.stdout.read()
    boundaries = extract_boundaries_from_ffprobe_output(output)
    return boundaries

def extract_boundaries_from_ffprobe_output(output):
    """
    extracts the shot boundaries from the string output
    producted by ffprobe

    Args:
        output (string): the full output of the ffprobe
            shot detector as a single string

    Returns:
        List[(float, float)]: a list of tuples of floats
        representing predicted shot boundaries (in seconds) and
        their associated scores
    """
    boundaries = []
    for line in output.decode().split('\n')[15:-1]:
        try:
            boundary = float(line.split('|')[4].split('=')[-1])
            score = float(line.split('|')[-1].split('=')[-1])
            boundaries.append((boundary, score))
        except:
            continue
    return boundaries

def _get_output_dim(h, w, resolution):
        if h >= w:
            return int(h * resolution / w), resolution
        else:
            return resolution, int(w * resolution / h)

def _get_video(video_path, resolution):
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

class DenseVideoCaptioning_Dataset(Dataset):
    def __init__(
            self,
            json_path,
            vids_path,
            resolution=224,
    ):
        self.data = json.load(open(json_path, 'r'))
        self.vids = list(self.data.keys())
        self.vids_path = json.load(open(vids_path, 'r'))
        self.resolution = resolution

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
        video_path = self.vids_path.get(video_id, None)
        duration = self.data[video_id]['duration']

        return {
            "video_id": video_id,
            "video_path": video_path,
            "duration": duration
        }

def custom_collate_fn(batch):
    bs = len(batch)
    video_id = [batch[i]["video_id"] for i in range(bs)]
    video_path = [batch[i]["video_path"] for i in range(bs)]
    duration = [batch[i]["duration"] for i in range(bs)]
    return {
        "video_id": video_id,
        "video_path": video_path,
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
        vids_path = os.path.join(DATA_DIR, name2folder["youcook"], "vids_path.json")
    elif dataset_name == "vitt":
        if split == "train":
            json_path = args.vitt_train_json_path
        elif split == "val":
            json_path = args.vitt_val_json_path
        elif split == "test":
            json_path = args.vitt_test_json_path
        else:
            raise NotImplementedError
        vids_path = os.path.join(DATA_DIR, name2folder["vitt"], "vids_path.json")
    elif dataset_name == "chapters":
        if split == "train":
            json_path = args.chapters_train_json_path
        elif split == "val":
            json_path = args.chapters_val_json_path
        elif split == "test":
            json_path = args.chapters_test_json_path
        else:
            raise NotImplementedError
        vids_path = os.path.join(DATA_DIR, name2folder["chapters"], "video_paths.json")
    else:
        raise NotImplementedError
    return DenseVideoCaptioning_Dataset(json_path=json_path,
                                        vids_path=vids_path,
                                        resolution=224)

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

processor = Blip2Processor.from_pretrained(args.model_name)
model = Blip2ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=th.float16)
device = th.device(args.device)
model.to(device)
model.eval()

@th.no_grad()
def evaluate(
    model,
    processor,
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
        vidspath = batch_dict["video_path"]
        durations = batch_dict["duration"]
        for vid, path, duration in zip(vids, vidspath, durations):
            prev_boundary = 0
            if path is not None:
                boundaries = extract_shots_with_ffprobe(path, threshold=0.7) + [(duration, 1.)]
            else:
                boundaries = [(duration, 1.)]
            video = _get_video(path, 224)
            chapters = []
            images = []
            prompts = []
            starts = [0] + [x[0] for x in boundaries[:-1]]
            ends = [x[0] for x in boundaries]
            for x in boundaries:
                boundary = x[0]
                try:
                    image = video[round((prev_boundary + boundary) / 2)]
                except:
                    image = th.zeros(3, 224, 224)
                prev_boundary = boundary
                prompt = f"Summarize the image in a chapter title. Chapter title:"
                images.append(image)
                prompts.append(prompt)
            images = th.stack(images)
            bs = 32
            n_batches = math.ceil(len(images) / bs)
            for i in range(n_batches):
                inputs = processor(images=images[i * bs: (i + 1) * bs], text=prompts[i * bs: (i + 1) * bs], return_tensors="pt", padding=True, truncation=True).to(device, th.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)
                chapters.extend([{"sentence": gen_txt, "timestamp": [st, ed]} for gen_txt, st, ed in zip(generated_text, starts[i * bs: (i + 1) * bs], ends[i * bs: (i + 1) * bs])])
                del inputs
                del generated_ids
                del generated_text
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
             processor=processor,
             data_loader=dataloader,
             device=device,
             dataset_name=args.combine_datasets_val[0],
             args=args,
             split="test",
             )