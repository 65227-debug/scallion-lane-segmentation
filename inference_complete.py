import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.data_loading import INPUT_SIZE, decode_mask, list_images, preprocess_image
from utils.metrics import SegmentationMetrics
from utils.runtime import add_model_arguments, load_model


def predict_image(model, image, device):
    with torch.inference_mode():
        return model(preprocess_image(image).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '--model-path', required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image', '--input', nargs='+')
    inputs.add_argument('--image-dir')
    parser.add_argument('--mask-dir')
    parser.add_argument('--file-list')
    parser.add_argument('--mask-encoding', choices=['indices', 'grayscale'], default='indices')
    parser.add_argument('--output')
    parser.add_argument('--output-dir', default='inference_results')
    add_model_arguments(parser)
    args = parser.parse_args()
    paths = [Path(name) for name in args.image] if args.image else list_images(args.image_dir, args.file_list)
    if args.file_list and args.image:
        parser.error('--file-list requires --image-dir')
    if args.output and len(paths) != 1:
        parser.error('--output requires one image')
    if len({path.stem for path in paths}) != len(paths):
        parser.error('Input image stems must be unique')
    if args.mask_dir:
        missing = [str(Path(args.mask_dir) / (path.stem + '.png')) for path in paths
                   if not (Path(args.mask_dir) / (path.stem + '.png')).is_file()]
        if missing:
            raise FileNotFoundError('Missing masks: ' + ', '.join(missing))
    model = load_model(args.model, args.device, args)
    metric = SegmentationMetrics()
    records = []
    for path in paths:
        with Image.open(path) as image:
            original_size = image.size
            prediction = predict_image(model, image, args.device)
        output = Path(args.output) if args.output else Path(args.output_dir) / (path.stem + '.png')
        if output.suffix.lower() != '.png':
            raise ValueError('Prediction masks must use PNG format')
        if output.resolve() == path.resolve():
            raise ValueError('Output must differ from input')
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction).save(output)
        if args.mask_dir:
            with Image.open(Path(args.mask_dir) / (path.stem + '.png')) as mask:
                if mask.size != original_size:
                    raise ValueError(f'Image/mask dimensions differ: {path.name}')
                decoded = decode_mask(mask, args.mask_encoding)
                target = np.asarray(Image.fromarray(decoded.astype(np.uint8)).resize(INPUT_SIZE, Image.Resampling.NEAREST))
                metric.update(prediction, target)
        records.append(path.name)
    if args.mask_dir:
        result = metric.compute()
        result.update({'images_evaluated': len(records), 'image_names': records})
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
