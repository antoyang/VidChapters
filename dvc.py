import os
import torch
import numpy as np
import random
import json
import math
import sys
from typing import Iterable
import argparse
import time
import datetime
import re
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce

from dataset import densevideocaptioning_collate_fn, build_densevideocaptioning_dataset, build_yt_dataset, yt_collate_fn
from model import build_vid2seq_model, _get_tokenizer
from args import get_args_parser
from util.misc import adjust_learning_rate
from util.metrics import MetricLogger
from dvc_eval import eval_dvc, eval_soda


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        output_tokens = batch_dict["output_tokens"].to(device)
        output_tokenized = {'input_ids': output_tokens,
                            'attention_mask': output_tokens != 0}
        if "input_tokens" not in batch_dict and args.use_speech:
            input_tokens = torch.ones((output_tokens.shape[0], 1)).long().to(device)
            input_tokenized = {'input_ids': input_tokens,
                               'attention_mask': input_tokens != 0}
        elif "input_tokens" in batch_dict:
            input_tokens = batch_dict["input_tokens"].to(device)
            input_tokenized = {'input_ids': input_tokens,
                               'attention_mask': input_tokens != 0}
        else:
            input_tokenized = {'input_ids': None,
                               'attention_mask': None}

        # forward
        if args.genasr and args.generative:  # vid2seq style generative loss on speech sequence
            geninput_tokens = torch.ones((output_tokens.shape[0], 1)).long().to(device)
            geninput_tokenized = {'input_ids': geninput_tokens,
                             'attention_mask': geninput_tokens != 0}
            loss_dict, video_dict = model(
                video=video,
                input_tokenized=geninput_tokenized,
                output_tokenized=input_tokenized,
            )
            loss = args.generative * loss_dict["loss"]

        elif args.generative:
            loss_dict, video_dict = model(
                video=video,
                input_tokenized=input_tokenized,
                output_tokenized=output_tokenized,
            )
            loss = args.generative * loss_dict["loss"]

        if args.denoising:
            denoising_output_tokens = batch_dict["denoising_output_tokens"].to(device)
            denoising_output_tokenized = {'input_ids': denoising_output_tokens,
                                          'attention_mask': denoising_output_tokens != 0}
            denoising_input_tokens = batch_dict["denoising_input_tokens"].to(device)
            denoising_input_tokenized = {'input_ids': denoising_input_tokens,
                                         'attention_mask': denoising_input_tokens != 0}
            if args.generative:
                denoising_loss_dict, _ = model(
                    video=video_dict,
                    input_tokenized=denoising_input_tokenized,
                    output_tokenized=denoising_output_tokenized,
                )
                loss_dict.update({"denoising_loss": denoising_loss_dict["loss"]})
                loss += args.denoising * denoising_loss_dict["loss"]
            else:
                denoising_loss_dict, _ = model(
                    video=video,
                    input_tokenized=denoising_input_tokenized,
                    output_tokenized=denoising_output_tokenized,
                )
                loss_dict = {"denoising_loss": denoising_loss_dict["loss"]}
                loss = args.denoising * denoising_loss_dict["loss"]

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_reduced = sum(loss_dict_reduced.values())
        loss_value = loss_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        with torch.no_grad():
            # Normalize time embeddings.
            frozen_norm = torch.norm(model.t5_model.shared.weight[:-args.num_bins, :], dim=1).mean(0)
            trainable_weight = model.t5_model.shared.weight[-args.num_bins:, :]
            model.t5_model.shared.weight[-args.num_bins:, :].div_(torch.norm(trainable_weight, dim=1).mean(0) / frozen_norm)

            frozen_norm = torch.norm(model.t5_model.lm_head.weight[:-args.num_bins, :], dim=1).mean(0)
            trainable_weight = model.t5_model.lm_head.weight[-args.num_bins:, :]
            model.t5_model.lm_head.weight[-args.num_bins:, :].div_(torch.norm(trainable_weight, dim=1).mean(0) / frozen_norm)

        adjust_learning_rate(
            optimizer,
            curr_step=epoch * len(data_loader) + i_batch,
            num_training_steps=num_training_steps,
            args=args,
        )

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    args,
    split="test",
    dataset_name="chapters"
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        duration = batch_dict["duration"]
        video = batch_dict["video"].to(device)
        if "input_tokens" not in batch_dict and args.use_speech:
            input_tokens = torch.ones((video.shape[0], 1)).long().to(device)
            input_tokenized = {'input_ids': input_tokens,
                               'attention_mask': input_tokens != 0}
        elif "input_tokens" in batch_dict:
            input_tokens = batch_dict["input_tokens"].to(device)
            input_tokenized = {'input_ids': input_tokens,
                               'attention_mask': input_tokens != 0}
        else:
            input_tokenized = {'input_ids': None,
                               'attention_mask': None}

        output = model.generate(video=video,
                                input_tokenized=input_tokenized,
                                use_nucleus_sampling=args.num_beams == 0,
                                num_beams=args.num_beams,
                                max_length=args.max_output_tokens,
                                min_length=1,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty,
                                length_penalty=args.length_penalty,
                                num_captions=1,
                                temperature=1)
        for i, vid in enumerate(batch_dict["video_id"]):
            sequences = re.split(r'(?<!<)\s+(?!>)', output[i]) # "<time=5> <time=7> Blablabla <time=7> <time=9> Blobloblo <time=2>" -> ['<time=5>', '<time=7>', 'Blablabla', '<time=7>', '<time=9>', 'Blobloblo', '<time=2>']
            indexes = [j for j in range(len(sequences) - 1) if sequences[j][:6] == '<time=' and sequences[j + 1][:6] == '<time=']
            last_processed = -2
            res[vid] = []
            for j, idx in enumerate(indexes):  # iterate on predicted events
                if idx == last_processed + 1:  # avoid processing 3 time tokens in a row as 2 separate events
                    continue
                seq = [sequences[k] for k in range(idx + 2, indexes[j + 1] if j < len(indexes) - 1 else len(sequences)) if sequences[k] != '<time=']
                if seq:
                    text = ' '.join(seq)
                else:  # no text
                    continue
                start_re = re.search(r'\<time\=(\d+)\>', sequences[idx])
                assert start_re, sequences[idx]
                start_token = int(start_re.group(1))
                start = float(start_token) * float(duration[i]) / float(args.num_bins - 1)
                end_re = re.search(r'\<time\=(\d+)\>', sequences[idx + 1])
                assert end_re, sequences[idx + 1]
                end_token = int(end_re.group(1))
                end = float(end_token) * float(duration[i]) / float(args.num_bins - 1)
                if end <= start:  # invalid time
                    continue
                res[vid].append({'sentence': text,
                                 'timestamp': [start,
                                               end]})
                last_processed = idx

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


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        print(args)

    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    tokenizer = _get_tokenizer(args.model_name, args.num_bins)

    nt = namedtuple(
        typename="data",
        field_names=[
            "dataset_name",
            "dataloader_val",
            "dataloader_train",
            "dataloader_test",
        ],
    )

    tuples = []
    for dset_name in args.combine_datasets:
        dataloader_val = None
        dataloader_test = None
        if dset_name in args.combine_datasets_val:
            dataset_val = build_densevideocaptioning_dataset(dset_name, "val", args, tokenizer)
            sampler_val = (
                DistributedSampler(dataset_val, shuffle=False)
                if args.distributed
                else torch.utils.data.SequentialSampler(dataset_val)
            )
            dataloader_val = DataLoader(
                dataset_val,
                batch_size=args.batch_size_val,
                sampler=sampler_val,
                collate_fn=densevideocaptioning_collate_fn,
                num_workers=args.num_workers,
            )
            if dset_name in ["vitt", "chapters"]:
                dataset_test = build_densevideocaptioning_dataset(dset_name, "test", args, tokenizer)
                sampler_test = (
                    DistributedSampler(dataset_test, shuffle=False)
                    if args.distributed
                    else torch.utils.data.SequentialSampler(dataset_test)
                )
                dataloader_test = DataLoader(
                    dataset_test,
                    batch_size=args.batch_size_val,
                    sampler=sampler_test,
                    collate_fn=densevideocaptioning_collate_fn,
                    num_workers=args.num_workers,
                )
            else:
                dataloader_test = dataloader_val

        if not args.eval:
            if dset_name in ["htm"]:
                dataset_train = build_yt_dataset(dset_name, "train", args, tokenizer)
            else:
                dataset_train = build_densevideocaptioning_dataset(dset_name, "train", args, tokenizer)
            sampler_train = (
                DistributedSampler(dataset_train)
                if args.distributed
                else torch.utils.data.RandomSampler(dataset_train)
            )
            dataloader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                sampler=sampler_train,
                collate_fn=yt_collate_fn if dset_name in ["htm"] else densevideocaptioning_collate_fn,
                num_workers=args.num_workers,
            )
        else:
            dataloader_train = None

        tuples.append(
            nt(
                dataset_name=dset_name,
                dataloader_test=dataloader_test,
                dataloader_val=dataloader_val,
                dataloader_train=dataloader_train,
            )
        )

    model = build_vid2seq_model(args, tokenizer)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.is_main_process():
        print("number of params:", n_parameters)
    # print(model)

    # Set up optimizer
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(
        params_for_optimization,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    for i, item in enumerate(tuples):
        if not args.eval:
            if dist.is_main_process():
                print("Start training")
            start_time = time.time()
            best_epoch = args.start_epoch
            best_acc = 0
            for epoch in range(args.start_epoch, args.epochs):
                if dist.is_main_process():
                    print(f"Starting epoch {epoch}")
                if args.distributed:
                    sampler_train.set_epoch(epoch)
                train_stats = train_one_epoch(
                    model=model,
                    data_loader=item.dataloader_train,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    args=args,
                )

                if (epoch + 1) % args.eval_skip == 0:
                    val_stats = {}
                    for i, item in enumerate(tuples):
                        if item.dataloader_val is None:
                            continue
                        print(f"Validating {item.dataset_name}")

                        out = evaluate(
                            model=model,
                            data_loader=item.dataloader_val,
                            device=device,
                            dataset_name=item.dataset_name,
                            args=args,
                            split="val",
                        )
                        val_stats.update(
                            {item.dataset_name + "_" + k: v for k, v in out.items()}
                        )
                        if out["CIDEr"] > best_acc:
                            best_epoch = epoch
                            best_acc = out["CIDEr"]

                            if dist.is_main_process() and args.save_dir:
                                checkpoint_path = os.path.join(
                                    args.save_dir, f"best_model.pth"
                                )
                                dist.save_on_master(
                                    {
                                        "model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "epoch": epoch,
                                        "args": args,
                                    },
                                    checkpoint_path,
                                )
                else:
                    val_stats = {}

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                if args.save_dir and dist.is_main_process():
                    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    checkpoint_path = os.path.join(args.save_dir, f"ckpt.pth")
                    dist.save_on_master(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            # load best ckpt
            if dist.is_main_process() and args.save_dir:
                print(f"loading best checkpoint from epoch {best_epoch}")
            if args.save_dir:
                torch.distributed.barrier()  # wait all processes
                checkpoint = torch.load(
                    os.path.join(args.save_dir, f"best_model.pth"),
                    map_location="cpu",
                )
                model.load_state_dict(checkpoint["model"], strict=False)

        out = evaluate(
            model=model,
            data_loader=item.dataloader_test,
            device=device,
            dataset_name=item.dataset_name,
            args=args,
            split="test",
        )

        if args.save_dir and dist.is_main_process():
            json.dump(
                out,
                open(
                    os.path.join(args.save_dir, item.dataset_name + "summary.json"), "w"
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
