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
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce

from dataset import videocaptioning_collate_fn, build_videocaptioning_dataset
from model import build_vid2seq_model, _get_tokenizer
from args import get_args_parser
from util.misc import adjust_learning_rate
from util.metrics import MetricLogger
from dvc_eval import COCOEvalCap
from transformers import LlamaForCausalLM, LlamaTokenizer, Blip2Processor, Blip2ForConditionalGeneration


def train_one_epoch(
    model: torch.nn.Module,
    tokenizer,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
):
    if isinstance(model, Blip2ForConditionalGeneration) or isinstance(model, LlamaForCausalLM):
        raise NotImplementedError
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        input_text = batch_dict["input_text"]
        output_text = batch_dict["output_text"]
        video = batch_dict["video"].to(device)
        input_tokenized = tokenizer(input_text, padding="longest", truncation=True, max_length=args.max_input_tokens, return_tensors="pt").to(device)
        output_tokenized = tokenizer(output_text, padding="longest", truncation=True, max_length=args.max_output_tokens, return_tensors="pt").to(device)
        loss_dict, _ = model(
            video=video,
            input_tokenized=input_tokenized,
            output_tokenized=output_tokenized,
        )

        loss = loss_dict["loss"]

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
    tokenizer,
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
        input_text = batch_dict["input_text"][0]
        if args.random:
            output = input_text
        elif isinstance(model, Blip2ForConditionalGeneration):
            video = batch_dict["raw_video"][0, :, 0]
            text = ["Summarize the image in a chapter title. Chapter title:"] * len(video)
            inputs = tokenizer(images=video, text=text, return_tensors="pt", padding=True, truncation=True).to(device,
                                                                                                    torch.float16)
            output_tokens = model.generate(**inputs,
                                           num_beams=args.num_beams,
                                           max_new_tokens=args.max_output_tokens,
                                           min_length=1,
                                           top_p=args.top_p,
                                           repetition_penalty=args.repetition_penalty,
                                           length_penalty=args.length_penalty,
                                           temperature=1)
            output = tokenizer.batch_decode(output_tokens.detach().cpu(), skip_special_tokens=True)
        elif isinstance(model, LlamaForCausalLM):
            text = [
                f"Summarize the following speech transcript in a chapter title. Transcript:{inx} Chapter title:"
                for inx in input_text]
            tokenized = tokenizer(text, padding="longest", truncation=True, max_length=args.max_input_tokens,
                                  return_tensors="pt").to(device)
            output_tokens = model.generate(**tokenized,
                                           num_beams=args.num_beams,
                                           max_new_tokens=args.max_output_tokens,
                                           min_length=1,
                                           top_p=args.top_p,
                                           repetition_penalty=args.repetition_penalty,
                                           length_penalty=args.length_penalty,
                                           temperature=1)
            output = tokenizer.batch_decode(output_tokens.detach().cpu(), skip_special_tokens=True)
        else:
            video = batch_dict["video"][0].to(device)
            input_tokenized = tokenizer(input_text, padding="longest", truncation=True,
                                        max_length=args.max_input_tokens, return_tensors="pt").to(device)
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

        gts = batch_dict["output_text"][0]
        video_id = batch_dict["video_id"][0]
        clip_ids = [video_id + str(i) for i in range(len(gts))]
        for clip_id, pred, gt in zip(clip_ids, output, gts):
            res[clip_id] = {'sentence': pred, 'gt': gt}

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    metrics = {}
    if dist.is_main_process():
        if args.save_dir:
            pred_path = os.path.join(args.save_dir, dataset_name + "_val_preds.json",)
            json.dump({'results': results}, open(pred_path, "w",))
        cocoeval = COCOEvalCap(results)
        metrics.update(cocoeval.evaluate())

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
            dataset_val = build_videocaptioning_dataset(dset_name, "val", args)
            sampler_val = (
                DistributedSampler(dataset_val, shuffle=False)
                if args.distributed
                else torch.utils.data.SequentialSampler(dataset_val)
            )
            dataloader_val = DataLoader(
                dataset_val,
                batch_size=args.batch_size_val,
                sampler=sampler_val,
                collate_fn=videocaptioning_collate_fn,
                num_workers=args.num_workers,
            )
            if dset_name in ["vitt", "chapters"]:
                dataset_test = build_videocaptioning_dataset(dset_name, "test", args)
                sampler_test = (
                    DistributedSampler(dataset_test, shuffle=False)
                    if args.distributed
                    else torch.utils.data.SequentialSampler(dataset_test)
                )
                dataloader_test = DataLoader(
                    dataset_test,
                    batch_size=args.batch_size_val,
                    sampler=sampler_test,
                    collate_fn=videocaptioning_collate_fn,
                    num_workers=args.num_workers,
                )
            else:
                dataloader_test = dataloader_val

        if not args.eval:
            dataset_train = build_videocaptioning_dataset(dset_name, "train", args)
            sampler_train = (
                DistributedSampler(dataset_train)
                if args.distributed
                else torch.utils.data.RandomSampler(dataset_train)
            )
            dataloader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                sampler=sampler_train,
                collate_fn=videocaptioning_collate_fn,
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

    if args.model_name == "Salesforce/blip2-flan-t5-xl":
        tokenizer = Blip2Processor.from_pretrained(args.model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float16)
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = False
    elif "7BHF" in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = "<s>"
        model = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        args.num_bins = 0
        tokenizer = _get_tokenizer(args.model_name, args.num_bins)
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
        # remove time tokens
        if 't5_model.shared.weight' in checkpoint["model"]:
            checkpoint["model"]['t5_model.shared.weight'] = checkpoint["model"]['t5_model.shared.weight'][:32100]
            checkpoint["model"]['t5_model.encoder.embed_tokens.weight'] = checkpoint["model"]['t5_model.encoder.embed_tokens.weight'][:32100]
            checkpoint["model"]['t5_model.decoder.embed_tokens.weight'] = checkpoint["model"]['t5_model.decoder.embed_tokens.weight'][:32100]
            checkpoint["model"]['t5_model.lm_head.weight'] = checkpoint["model"]['t5_model.lm_head.weight'][:32100]
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
                    tokenizer=tokenizer,
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
                            tokenizer=tokenizer,
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
            tokenizer=tokenizer,
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
    if "7BHF" not in args.model_name and "Salesforce" not in args.model_name:
        args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
