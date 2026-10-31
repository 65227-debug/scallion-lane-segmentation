from unet import UNet
from utils.runtime import load_model

dependencies = ['torch', 'numpy', 'PIL']


def scallion_unet(checkpoint=None, **kwargs):
    if checkpoint is not None:
        model = load_model(checkpoint)
        for key, value in kwargs.items():
            if model.get_config_summary().get(key) != value:
                raise ValueError(f'Model option does not match checkpoint: {key}')
        return model
    return UNet(**kwargs)
