import argparse
from pathlib import Path

import torch

from unet import UNet


def load_state(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must contain a state dictionary')
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state, dict):
        raise ValueError('Invalid state dictionary')
    cleaned = {}
    for key, value in state.items():
        if key == 'mask_values' or key.endswith(('.total_ops', '.total_params')):
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'Unexpected checkpoint entry: {key}')
        cleaned[key.removeprefix('module.')] = value
    return cleaned


def detect_config(state):
    required = ('inc.conv1.0.weight', 'outc.conv.weight')
    if any(key not in state for key in required):
        raise ValueError('Missing model input/output tensors')
    config = {
        'n_channels': state['inc.conv1.0.weight'].shape[1],
        'n_classes': state['outc.conv.weight'].shape[0],
        'bilinear': 'up1.up.weight' not in state,
        'use_dsem': any('.dir_convs.' in key for key in state),
        'use_dacg': any('.gate_conv.' in key for key in state),
        'use_tgsm': any(key.startswith('tgsm.') for key in state),
        'tgsm_version': 'enhanced',
    }
    if config['use_dsem']:
        tensor = state['down4.maxpool_conv.1.conv2.3.dir_convs.0.weight']
        if tensor.shape[1] % tensor.shape[0]:
            raise ValueError('Cannot infer DSEM channel reduction')
        config['reduction'] = tensor.shape[1] // tensor.shape[0]
    return config


def add_model_arguments(parser):
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--bilinear', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use-dsem', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use-dacg', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use-tgsm', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--reduction', type=int, default=None)
    parser.add_argument('--tgsm-version', choices=['enhanced'], default='enhanced')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')


def load_model(path, device='cpu', args=None):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    state = load_state(path)
    config = detect_config(state)
    if config['n_classes'] != 3 or config['n_channels'] != 3:
        raise ValueError('Expected three RGB channels and three output classes')
    if args is not None:
        if args.classes != config['n_classes']:
            raise ValueError('Class count does not match checkpoint')
        for name in ('bilinear', 'use_dsem', 'use_dacg', 'use_tgsm', 'reduction'):
            value = getattr(args, name, None)
            if value is not None and value != config.get(name, 4):
                raise ValueError(f'Model option does not match checkpoint: {name}')
    model = UNet(**config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
