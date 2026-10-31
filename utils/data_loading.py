from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ('Background', 'Ridge', 'Furrow')
INPUT_SIZE = (512, 512)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def decode_mask(image, encoding='indices'):
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError('Mask must have a single channel')
    labels = (0, 1, 2) if encoding == 'indices' else (0, 85, 170)
    if encoding not in ('indices', 'grayscale'):
        raise ValueError('Mask encoding must be indices or grayscale')
    values = set(np.unique(array).tolist())
    if not values.issubset(set(labels) | {255}):
        raise ValueError(f'Unknown mask labels: {sorted(values - set(labels) - {255})}')
    result = np.full(array.shape, 255, dtype=np.int64)
    for index, value in enumerate(labels):
        result[array == value] = index
    return result


def preprocess_image(image):
    image = image.convert('RGB').resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array.copy())


def list_images(directory, file_list=None):
    root = Path(directory).resolve()
    if file_list:
        text = Path(file_list).read_text(encoding='utf-8')
        names = json.loads(text) if Path(file_list).suffix.lower() == '.json' else text.splitlines()
        if isinstance(names, dict):
            names = names['test']
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError('File list must contain image paths')
        paths = [(root / name.strip()).resolve() for name in names if name.strip()]
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError('Image paths must be inside the input directory')
    else:
        paths = sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Input images must be nonempty and contain no repeated paths')
    if any(not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES for path in paths):
        raise FileNotFoundError('An input image is missing or has an unsupported extension')
    return paths


class BasicDataset(Dataset):
    def __init__(self, images_dir, mask_dir, file_list=None, encoding='indices', mask_suffix=''):
        self.images = list_images(images_dir, file_list)
        self.mask_dir = Path(mask_dir)
        self.encoding = encoding
        self.masks = [self.mask_dir / (path.stem + mask_suffix + '.png') for path in self.images]
        if len({path.stem for path in self.images}) != len(self.images):
            raise ValueError('Image stems must be unique')
        missing = [str(path) for path in self.masks if not path.is_file()]
        if missing:
            raise FileNotFoundError('Missing masks: ' + ', '.join(missing))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as image, Image.open(self.masks[index]) as mask:
            if image.size != mask.size:
                raise ValueError(f'Image/mask dimensions differ: {self.images[index].name}')
            decoded = decode_mask(mask, self.encoding)
            mask_image = Image.fromarray(decoded.astype(np.uint8))
            resized = np.asarray(mask_image.resize(INPUT_SIZE, Image.Resampling.NEAREST), dtype=np.int64)
            return {'image': preprocess_image(image), 'mask': torch.from_numpy(resized.copy()),
                    'name': self.images[index].name}
