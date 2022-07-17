import argparse
import time
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np

from models import *
from utils import *

from generate_data import generate_data


def get_args_parser():
    parser = argparse.ArgumentParser(description="PSAQ-ViT", add_help=False)
    parser.add_argument("--model", default="deit_tiny",
                        choices=['deit_tiny', 'deit_small', 'deit_base', 'swin_tiny', 'swin_small'],
                        help="model")
    parser.add_argument('--dataset', default="/Path/to/Dataset/",
                        help='path to dataset')
    parser.add_argument("--calib-batchsize", default=32,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batchsize", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=16, type=int,
                        help="number of data loading workers (default: 16)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--print-freq", default=100,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    parser.add_argument("--mode", default=0,
                        type=int, help="mode of calibration data, 0: PSAQ-ViT, 1: Gaussian noise, 2: Real data")
    parser.add_argument('--w_bit', default=8,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bit', default=8,
                        type=int, help='bit-precision of activation')

    return parser


class Config:
    def __init__(self, w_bit, a_bit):
        self.weight_bit = w_bit
        self.activation_bit = a_bit


def str2model(name):
    model_zoo = {'deit_tiny': deit_tiny_patch16_224,
                 'deit_small': deit_small_patch16_224,
                 'deit_base': deit_base_patch16_224,
                 'swin_tiny': swin_tiny_patch4_window7_224,
                 'swin_small': swin_small_patch4_window7_224
                 }
    print('Model: %s' % model_zoo[name].__name__)
    return model_zoo[name]


def seed(seed=0):
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    print(args)
    seed(args.seed)

    device = torch.device(args.device)
    # Load bit-config
    cfg = Config(args.w_bit, args.a_bit)

    # Build model
    model = str2model(args.model)(pretrained=True, cfg=cfg)
    model = model.to(device)
    model.eval()

    # Build dataloader
    train_loader, val_loader = build_dataset(args)

    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # Get calibration set
    # Case 0: PASQ-ViT
    if args.mode == 0:
        print("Generating data...")
        calibrate_data = generate_data(args)
        print("Calibrating with generated data...")
        with torch.no_grad():
            output = model(calibrate_data)
    # Case 1: Gaussian noise
    elif args.mode == 1:
        calibrate_data = torch.randn((args.calib_batchsize, 3, 224, 224)).to(device)
        print("Calibrating with Gaussian noise...")
        with torch.no_grad():
            output = model(calibrate_data)
    # Case 2: Real data (Standard)
    elif args.mode == 2:
        for data, target in train_loader:
            calibrate_data = data.to(device)
            break
        print("Calibrating with real data...")
        with torch.no_grad():
            output = model(calibrate_data)
    # Not implemented
    else:
        raise NotImplementedError

    # Freeze model
    model.model_quant()
    model.model_freeze()

    # Validate the quantized model
    print("Validating...")
    val_loss, val_prec1, val_prec5 = validate(
        args, val_loader, model, criterion, device
    )


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser('PSAQ', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
