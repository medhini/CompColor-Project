import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet
from .bilateral_net import BilateralColorNet


class ColorModel(nn.Module):
    def __init__(self, net_enc, net_dec, use_bilateral=True,
                 learn_guide=False, bilateral_depth=8, num_features=256):
        super(ColorModel, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = nn.L1Loss()

        self.bilateral_net = None
        self.guide_net = None
        self.direct_net = None

        self.conv_block = nn.Sequential(
            nn.Conv2d(net_dec.fpn_dim + 1, num_features, kernel_size=3),
            nn.ReLU(),
            resnet.BasicBlock(num_features, num_features),
        )

        if use_bilateral:
            self.bilateral_net = BilateralColorNet(
                num_input_features=num_features,
                bilateral_depth=bilateral_depth)
            if learn_guide:
                self.guide_net = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3),
                    nn.ReLU(),
                    resnet.BasicBlock(64, 64),
                    nn.Conv2d(64, 1, kernel_size=3),
                    nn.Sigmoid(),
                )
        else:
            self.direct_net = nn.Sequential(
                nn.Conv2d(num_features, 2, kernel_size=1),
                nn.Tanh(),
            )

    def forward(self, luma: torch.Tensor, chroma: torch.Tensor,
                is_inference: bool = False, scale: int = 4):
        """Predicts a color Lab image from input luma and computes loss.

        Args:
            luma (torch.Tensor): Tensor of shape [N, 1, H, W].
            chroma (torch.Tensor): Tensor of shape [N, 2, H, W].
            is_inference (bool, optional): Whether to also return the predicted
                image. Defaults to False.
            scale (int, optional): Scale by which to downsample. Defaults to 4.

        Returns:
            Returns loss or tuple of loss and output ab channels.
        """
        inputs = luma.repeat(1, 3, 1, 1)
        # Features are of shape [N, self.decoder.fc_dim, H//4, W//4]
        feat = self.decoder(self.encoder(inputs))

        # Resizes features and inputs according to self.scale.
        size = (luma.shape[2] // scale, luma.shape[3] // scale)
        scaled_luma = F.interpolate(
            luma, size=size, mode='bilinear', align_corners=False)
        scaled_feat = F.interpolate(
            feat, size=size, mode='bilinear', align_corners=False)

        # Concat features with input luma and
        feat = torch.cat((scaled_luma, scaled_feat), dim=1)
        feat = self.conv_block(feat)

        if self.bilateral_net is not None:
            if self.guide_net is not None:
                guide = self.guide_net(luma)
            else:
                guide = luma
            output = self.bilateral_net(guide, feat)
        else:
            output = self.direct_net(feat)

        loss = self.crit(output, chroma)

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
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        return conv_out
