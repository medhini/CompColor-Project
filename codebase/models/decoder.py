import torch
from torch import nn

class PalletEncoder(nn.Module):
    def __init__(self, p_dim_in=12, p_dim_out=32):
        super(PalletEncoder, self).__init__()

        self.p_dim_in = p_dim_in
        self.p_dim_out = p_dim_out

        #scale up pallet to 32 dim
        self.pe1 = nn.Sequential(nn.Linear(self.p_dim_in, 24), 
                                nn.BatchNorm1d(24), nn.ReLU())
        self.pe2 = nn.Sequential(nn.Linear(24, self.p_dim_out), 
                                nn.BatchNorm1d(self.p_dim_out), nn.ReLU())

    def forward(self, pallet):
        x = pallet.view(-1, self.p_dim_in) #(b, 12)
        x = self.pe1(x) #(b, 24)
        x = self.pe2(x) #(b, 32)

        return x

# upernet
class Decoder(nn.Module):
    def __init__(self, fc_dim=2048, pool_scales=(1, 2, 3, 6),
            fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256, use_pallet=False,
            p_dim_in=12, p_dim_out=32):
        super(Decoder, self).__init__()
        self.fpn_dim = fpn_dim

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []
        self.use_pallet = use_pallet

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1)
        )

        if self.use_pallet:
            self.encode_pallet = PalletEncoder(p_dim_in, p_dim_out)
            self.fuse_pallet = nn.Sequential(
                    conv3x3_bn_relu(fpn_dim+p_dim_out, fpn_dim, 1),
                    nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1)
                )

    def forward(self, conv_out, pallet=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]

        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_pallet:
            pe = self.encode_pallet(pallet)
            pe = torch.unsqueeze(pe, 2)
            pe = torch.unsqueeze(pe, 3)
            pe = pe.repeat(1, 1, x.shape[2], x.shape[3]) #(b, 32, 56, 56)
            x = torch.cat((x, pe), 1)
            x = self.fuse_pallet(x)

        return x

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """ 3x3 convolution + BN + relu """
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
