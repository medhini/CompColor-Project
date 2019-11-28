import argparse
import os
import shutil
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from models import ModelBuilder, ColorModel, BilateralColorNet
from dataset import ColorDataset
from utils import AverageMeter, Logger
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch Colorization')

# Path related arguments
parser.add_argument('--list_train',
                    default='train_base_videofolder.txt')
parser.add_argument('--list_val',
                    default='val_base_videofolder.txt')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture')

parser.add_argument('--root', '-r', default='/tmp_data1/flickr30k-images', type=str, 
                    help='data root directory')
parser.add_argument('--img_size', '-size', default=[224,224], type=int,
                    help='resize image to this size')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[35, 45], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

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
    base_model = builder.build_network(arch=args.arch)

    model = ColorModel(base_model, args)

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
                    transforms.RandomCrop(args.img_size),
                    ])

    dataset_train = ColorDataset(args.root, split="train",
                    transform=transform)

    dataset_val = ColorDataset(args.root, split="val",
                    transform=transform)

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
        num_workers=8, pin_memory=True
    )

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model)
        return

    # training, start a logger
    tb_logdir = os.path.join(args.logdir, args.arch.lower() + '_{}'.format(args.logname))
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, tb_logger)

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
    for i, (img_input, img_gt) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        loss = model(img_input, img_gt)
        loss = loss.mean()

        # measure accuracy and record loss
        losses.update(loss.item(), img_input.size(0))

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


def validate(val_loader, model, epoch=None, tb_logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (img_input, img_gt) in enumerate(val_loader):
        # compute output
        with torch.no_grad():
            loss, output = model(img_input, img_gt, is_inference=True)
            loss = loss.mean()

        # measure accuracy and record loss
        losses.update(loss.item(), img_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs['Val_EpochLoss'] = losses.avg
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

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
