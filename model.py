from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bilateral_net import BilateralColorNet


class ColorModel(nn.Module):
    def __init__(self, gpu_id: Optional[int] = None):
        super(ColorModel, self).__init__()
        self.bilateral_net = BilateralColorNet()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self._load_images(image)
        self.im_prediction = self.bilateral_net(self.im_gray)
        return self.im_prediction

    def compute_loss(self) -> torch.Tensor:
        return nn.L1Loss(self.im_prediction, self.im_label)

    def compute_psnr(self) -> torch.Tensor:
        mse = (self.im_prediction - self.im_label) ** 2.0
        mse = torch.mean(mse, dim=(1, 2, 3))
        psnr = 10 * (1.0 / mse).log10()
        return psnr.mean()

    def get_images(self) -> Dict[str, torch.Tensor]:
        difference = self.im_prediction - self.im_label
        difference = (difference / 2.0 + 0.5).clamp(0.0, 1.0)
        return {
            'Color_Label': self.im_label,
            'Color_Prediction': self.im_prediction,
            'Grayscale_Input': self.im_gray,
            'Difference': difference,
        }

    ############################################################################
    # Private Instance Methods
    ############################################################################

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if self.gpu_id is not None:
            image = image.cuda(device=self.gpu_id)
        return image

    def _load_images(self, image: torch.Tensor) -> None:
        self.im_label = self._prepare_image(image)
        self.im_gray = self._prepare_image(self._rgb_to_gray(image))

    ############################################################################
    # Private Static Methods
    ############################################################################

    @staticmethod
    def _rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
        red = image[:, 0:1, ...]
        green = image[:, 1:2, ...]
        blue = image[:, 2:3, ...]
        gray = 0.30 * red + 0.59 * green + 0.11 * blue
        return gray
