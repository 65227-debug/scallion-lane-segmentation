import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset
from utils.metrics import SegmentationMetrics, benchmark, count_parameters
from utils.runtime import add_model_arguments, load_model


def evaluate(model, loader, device):
    metric = SegmentationMetrics()
    names = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch['image'].to(device)).argmax(dim=1)
            metric.update(prediction, batch['mask'])
            names.extend(batch['name'])
    result = metric.compute()
    result.update({'images_evaluated': len(names), 'image_names': names})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--mask-dir', required=True)
    parser.add_argument('--file-list')
    parser.add_argument('--mask-encoding', choices=['indices', 'grayscale'], default='indices')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--timing-iterations', type=int, default=100)
    parser.add_argument('--skip-timing', action='store_true')
    parser.add_argument('--output', default='evaluation.json')
    add_model_arguments(parser)
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device, args)
    dataset = BasicDataset(args.data_dir, args.mask_dir, args.file_list, args.mask_encoding)
    result = evaluate(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False), args.device)
    result['parameters_million'] = count_parameters(model)[0]
    result['parameter_storage_mb'] = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    result['checkpoint_file_mb'] = Path(args.checkpoint).stat().st_size / 1e6
    result['model_config'] = model.get_config_summary()
    result['input_size'] = [512, 512]
    if not args.skip_timing:
        result['timing'] = benchmark(model, args.device, args.timing_iterations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'image_names'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
