import torch

import torch.nn as nn
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(self, cfg_model, no_of_landmarks):
        super(Unet, self).__init__()
        self.unet = smp.Unet(
            encoder_name=cfg_model.ENCODER_NAME,
            encoder_weights=cfg_model.ENCODER_WEIGHTS,
            decoder_channels=cfg_model.DECODER_CHANNELS,
            in_channels=cfg_model.IN_CHANNELS,
            classes=no_of_landmarks,
        )
        self.temperatures = nn.Parameter(torch.ones(1, no_of_landmarks, 1, 1), requires_grad=False)

    def forward(self, x):
        return self.unet(x)

    def scale(self, x):
        y = x / self.temperatures
        return y


def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


def nll_across_batch(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=(2, 3)))
