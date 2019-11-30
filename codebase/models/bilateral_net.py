import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """3x3 conv layer followed by leaky ReLU."""

    def __init__(self, num_in_channels: int, num_out_channels: int) -> None:
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size=3,
                      padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(  # pylint: disable=arguments-differ
        self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class DownsampleBlock(nn.Module):
    """Downsample block containing multiple convs followed by a downsample."""

    def __init__(self, num_in_channels: int, num_out_channels: int,
                 num_convs: int = 3) -> None:
        super(DownsampleBlock, self).__init__()

        conv_list = [Conv(num_in_channels, num_out_channels)]
        for _ in range(num_convs - 1):
            conv_list.append(Conv(num_out_channels, num_out_channels))
        self.convs = nn.Sequential(*conv_list)

    def forward(  # pylint: disable=arguments-differ
            self, features: torch.Tensor) -> torch.Tensor:
        features = self.convs(features)
        features = F.interpolate(features, scale_factor=0.5, mode='area')
        return features


class BilateralColorNet(nn.Module):
    """Bilateral neural network for image colorization."""

    def __init__(self, bilateral_depth: int = 8) -> None:
        super(BilateralColorNet, self).__init__()

        self.bilateral_depth = bilateral_depth

        self.encoder = nn.Sequential(
            DownsampleBlock(1, 16),
            DownsampleBlock(16, 32),
            DownsampleBlock(32, 64),
            DownsampleBlock(64, 128),
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(128, 3 * bilateral_depth, kernel_size=1),
            nn.Tanh(),
        )

    def forward(  # pylint: disable=arguments-differ
            self, image: torch.Tensor) -> torch.Tensor:
        """Colorizes a grayscale image using a bilateral neural network.

        Args:
            image (torch.Tensor): Input grayscale image of size [N, 1, H, W].

        Returns:
            torch.Tensor: Output color image of size [N, 3, H, W].
        """
        features = self.encoder(image)
        features = self.last_conv(features)
        depth = image[:, 0, ...] * (self.bilateral_depth - 1)
        return self._trilinear_slice(features, depth)

    @staticmethod
    def _trilinear_slice(
            features: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Trilinearly slices input features according to depth map.

        Args:
            features (torch.Tensor): Input features of size [N, 3 * D, H, W].
            depth (torch.Tensor): Depth map of size [N, H', W'].

        Returns:
            torch.Tensor: Output image of size [N, 3, H', W'].
        """
        # TODO: Use F.grid_sample if trilinear sampling is ever added.
        # https://github.com/pytorch/pytorch/issues/5565

        # Bilinearly upsamples spatial dimensions of features.
        N, H_prime, W_prime = depth.size()  # pylint: disable=invalid-name
        features = F.interpolate(features, size=(H_prime, W_prime),
                                 mode='bilinear', align_corners=False)

        # Adds channel dim of size 3 and depth dim of size 1 to depth map.
        depth = torch.stack([depth] * 3, dim=1)
        depth = depth[:, :, None, :, :]

        # Gets two depths to interpolate between at each position.
        depth_0 = depth.floor().long()
        depth_1 = depth.ceil().long()
        weight = depth % 1.0

        # Linearly interpolates between features at the two depths.
        features = features.view(N, 3, -1, H_prime, W_prime)
        features_0 = torch.gather(features, 2, depth_0)
        features_1 = torch.gather(features, 2, depth_1)
        # TODO: Use torch.lerp once derivative of 'weight' is released.
        # https://github.com/pytorch/pytorch/issues/22444
        features = features_0 + weight * (features_1 - features_0)

        image = features[:, :, 0, :, :]
        return image
