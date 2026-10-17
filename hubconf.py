"""PyTorch Hub entry point for scallion lane segmentation models."""

import torch

from unet import UNet


dependencies = ['torch']


def scallion_unet(checkpoint=None, n_channels=3, n_classes=3, bilinear=False,
                 use_dsem=False, reduction=4, use_dacg=False, use_tgsm=False,
                 tgsm_version='enhanced'):
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                 use_dsem=use_dsem, reduction=reduction, use_dacg=use_dacg,
                 use_tgsm=use_tgsm, tgsm_version=tgsm_version)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu')
        state = state.get('model_state_dict', state.get('state_dict', state))
        state = {key: value for key, value in state.items()
                 if isinstance(value, torch.Tensor)
                 and 'total_ops' not in key and 'total_params' not in key}
        model.load_state_dict(state)
    return model
