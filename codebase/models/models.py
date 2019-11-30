import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet
from .bilateral_net import BilateralColorNet


class ColorModel(nn.Module):
    def __init__(self, base_net, opt):
        super(ColorModel, self).__init__()
        self.base_net = base_net
        self.bilateral_net = BilateralColorNet()

        self.crit = nn.L1Loss()

    def forward(self, img_input, img_gt, is_inference=False):
        # img_input is (b, 3, h, w)

        '''Commenting this out for now as Tim's model requires image as input'''
        # feat = self.base_net(img_input)  # (b, self.base_net.fc_dim, h//16, w//16)

        # tri-linear sampling

        output = self.bilateral_net(img_input)

        loss = self.crit(output, img_gt)

        if is_inference:
            return loss, output
        return loss


class ModelBuilder(object):
    def __init__(self):
        pass

    @staticmethod
    def build_network(arch='resnet50', pretrained=True):
        if arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            network = Resnet(orig_resnet, fc_dim=512)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            network = Resnet(orig_resnet, fc_dim=2048)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            network = Resnet(orig_resnet, fc_dim=2048)
        else:
            raise Exception('Architecture undefined!')
        '''
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            network = Resnet(orig_resnext, fc_dim=2048)  # we can still use class Resnet
        elif arch == 'se_resnext50_32x4d':
            orig_se_resnext = senet.__dict__['se_resnext50_32x4d']()
            network = SEResNet(orig_se_resnext, fc_dim=2048, num_classes=num_classes)
        elif arch == 'se_resnext101_32x4d':
            orig_se_resnext = senet.__dict__['se_resnext101_32x4d']()
            network = SEResNet(orig_se_resnext, fc_dim=2048, num_classes=num_classes)
        elif arch == 'densenet121':
            orig_densenet = densenet.__dict__['densenet121'](pretrained=pretrained)
            network = DenseNet(orig_densenet, fc_dim=1024, num_classes=num_classes)
        '''

        return network


class Resnet(nn.Module):
    def __init__(self, orig_resnet, fc_dim):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        self.avgpool = orig_resnet.avgpool
        self.fc_dim = fc_dim

    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.squeeze(3).squeeze(2)

        return x
