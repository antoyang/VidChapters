import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(lengths, max_length):
    """Computes a batch of padding masks given batched lengths"""
    mask = 1 * (
        torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths
    ).transpose(0, 1)
    return mask


def adjust_learning_rate(
    optimizer,
    curr_step: int,
    num_training_steps: int,
    args,
):
    num_warmup_steps: int = round(args.fraction_warmup_steps * num_training_steps)
    if args.schedule == "linear_with_warmup":
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
        optimizer.param_groups[0]["lr"] = args.lr * gamma
    elif args.schedule == "":  # constant LR
        gamma = 1
        optimizer.param_groups[0]["lr"] = args.lr * gamma
    elif args.schedule == 'cosine_with_warmup':
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
            optimizer.param_groups[0]["lr"] = args.lr * gamma
        else:
            optimizer.param_groups[0]["lr"] = args.lr * (1 + math.cos(math.pi * float(curr_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)))) / 2
    else:
        raise NotImplementedError


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction: str ='mean', ignore_index: int = -100):
        super().__init__()
        self.epsilon = epsilon
        assert reduction == 'mean'
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean', ignore_index=self.ignore_index)
        return self.epsilon * (loss / n) + nll * (1 - self.epsilon)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: (#items, #classes)
    target: int,
    """
    maxk = max(topk)
    num_items = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / num_items))
    return res