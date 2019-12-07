# TODO: Remove irrelevant code for colorization.
# TODO: Break into multiple files (CLI and trainer).
import argparse
import os
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import color

from dataset import ColorDataset
from models import BilateralColorNet, ColorModel, Decoder, ModelBuilder
from utils import AverageMeter, Logger

parser = argparse.ArgumentParser(description='PyTorch Colorization')

# Path related arguments
parser.add_argument('--list_train',
                    default='train_base_videofolder.txt')
parser.add_argument('--list_val',
                    default='val_base_videofolder.txt')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture')

parser.add_argument('--root', '-r', default='/shared/timbrooks/datasets/mirflickr', type=str,
                    help='data root directory')
parser.add_argument('--img_size', '-size', default=[224,224], type=int,
                    help='resize image to this size')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[10, 20, 25], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-up', '--use_palette', default=False, type=bool,
                    help='condition based on gt target palette')

parser.add_argument('--use_bilinear', default=False, action='store_true',
                    help='use a bilinear upsample rather than a bilateral network')
parser.add_argument('--learn_guide', default=False, action='store_true',
                    help='whether to learn bilateral guidance map or use luma')
parser.add_argument('--scale', default=4, type=int,
                    help='scale by which to downsample')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', '-l', default=10, type=int,
                    metavar='N', help='frequency to write in tensorboard (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--logdir', default='./logs',
                    help='folder to output tensorboard logs')
parser.add_argument('--logname', default='exp',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')

best_loss = 1000000


def main():
    global args, best_loss
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))

    builder = ModelBuilder()
    base_enc_model = builder.build_network(arch=args.arch)
    base_dec_model = Decoder(fc_dim=base_enc_model.fc_dim, fpn_dim=256,
                             use_palette=args.use_palette)

    model = ColorModel(
        base_enc_model, base_dec_model, use_bilinear=args.use_bilinear,
        learn_guide=args.learn_guide)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    if args.start_epoch is None:
        args.start_epoch = 0

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # create training and validation dataset
    transform = transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(args.img_size),
                    ])

    dataset_train = ColorDataset(args.root, split="train",
                    transform=transform, use_palette=args.use_palette)

    dataset_val = ColorDataset(args.root, split="val",
                    transform=transform, use_palette=args.use_palette)

    # create training and validation loader
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, drop_last=True,
        pin_memory=True
    )

    #start a logger
    tb_logdir = os.path.join(args.logdir, args.arch.lower() + '_{}'.format(args.logname))
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    if args.evaluate:
        validate(val_loader, model, tb_logger=tb_logger)
        return

    #training
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, tb_logger)

        if (epoch + 1) % 5 == 0:
            # evaluate on validation set
            loss = validate(val_loader, model, epoch, tb_logger)

            # remember best loss and save checkpoint
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_loss': best_loss,
                },
                is_best,
                os.path.join(args.ckpt, args.arch.lower() + '_{}'.format(args.logname)))


def train(train_loader, model, optimizer, epoch, tb_logger=None):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, contents in enumerate(train_loader):
        #reading palette if use_palette is true
        if args.use_palette:
            if args.shift_palette:
                luma, chroma, palette1, palette2 = contents
            else:
                luma, chroma, palette1 = contents
                palette2 = palette1
        else:
            luma, chroma = contents
            palette1, palette2 = None

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        loss = model(luma, chroma, scale=args.scale, palette=palette2)
        loss = loss.mean()

        # measure accuracy and record loss
        losses.update(loss.item(), luma.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

        # log training data into tensorboard
        if tb_logger is not None and i % args.log_freq == 0:
            logs = OrderedDict()
            logs['Train_IterLoss'] = losses.val
            # how many iterations we have trained
            iter_count = epoch * len(train_loader) + i
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, iter_count)

            tb_logger.flush()


def compute_psnr(im_0: torch.Tensor, im_1: torch.tensor) -> torch.Tensor:
    mse = (im_0 - im_1) ** 2.0
    mse = torch.mean(mse, dim=(1, 2, 3))
    psnr = 10 * (1.0 / mse).log10()
    return psnr.mean()


def lab_to_rgb(luma: torch.Tensor, chroma: torch.Tensor) -> torch.Tensor:
    assert luma.shape[1] == 1
    assert chroma.shape[1] == 2

    luma = luma.data.cpu().numpy()
    chroma = chroma.data.cpu().numpy()

    luma = np.clip(luma * 100.0, 0.0, 100.0)
    chroma = np.clip(chroma * 110.0, -110.0, 110.0)

    image = np.concatenate((luma, chroma), axis=1)
    N, C, H, W = image.shape

    image = np.transpose(image, (0, 2, 3, 1))
    image = np.reshape(image, (N * H, W, C))
    image = color.lab2rgb(image)
    image = np.reshape(image, (N, H, W, C))
    image = np.transpose(image, (0, 3, 1, 2))
    return torch.from_numpy(image)


def visualize_palette(palette: torch.Tensor) -> torch.Tensor:
    """Visualizes a color palette of AB channels in LAB color space.

    Args:
        palette (torch.Tensor): Palette AB channels of shape [N, 6, 2, 1, 1].

    Returns:
        torch.Tensor: Palette RGB image of shape [N, 3, H, W].
    """
    N, P, C, H, W = palette.size()
    assert P == 6 and C == 2 and H ==1 and W == 1

    chroma = palette.permute(0, 2, 1, 3, 4).view(N, 2, 2, 3)
    luma = 0.5 * torch.ones(N, 1, 2, 3)
    rgb = lab_to_rgb(luma, chroma)
    rgb = F.interpolate(rgb, size=(100, 150))
    return rgb


def validate(val_loader, model, epoch=None, tb_logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    psnr = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, contents in enumerate(val_loader):
        if args.use_palette:
            if args.shift_palette:
                luma, chroma, palette1, palette2 = contents
            else:
                luma, chroma, palette1 = contents
                palette2 = palette1
        else:
            luma, chroma = contents
            palette1, palette2 = None

        # compute output
        with torch.no_grad():
            loss, output = model(luma, chroma, is_inference=True,
                                 scale=args.scale, palette=palette2)
            loss = loss.mean()

        # measure accuracy and record loss
        losses.update(loss.item(), luma.size(0))

        gt_rgb = lab_to_rgb(luma, chroma)
        out_rgb = lab_to_rgb(luma, output)

        _psnr = compute_psnr(gt_rgb, out_rgb)
        psnr.update(_psnr.item(), luma.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PSNR {psnr.val:.2f} ({psnr.avg:.2f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, psnr=psnr))

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs['Val_EpochLoss'] = losses.avg
        logs['PSNR'] = psnr.avg
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

        tb_logger.log_image(gt_rgb, 'Label', epoch + 1)
        tb_logger.log_image(out_rgb, 'Prediction', epoch + 1)
        tb_logger.log_image(luma, 'Input', epoch + 1)
        tb_logger.log_image(visualize_palette(palette1), 'Original Palette', epoch + 1)
        tb_logger.log_image(visualize_palette(palette2), 'Recolorized Palette', epoch + 1)

        tb_logger.flush()

    return losses.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
