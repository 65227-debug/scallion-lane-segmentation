import time

import numpy as np
import torch

from .data_loading import CLASS_NAMES


def as_array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


class SegmentationMetrics:
    def __init__(self, num_classes=3, ignore_index=255, class_names=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = tuple(class_names or CLASS_NAMES)
        if len(self.class_names) != num_classes:
            raise ValueError('Class names and class count differ')
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        pred, target = as_array(pred), as_array(target)
        if pred.ndim == target.ndim + 1:
            pred = pred.argmax(axis=1)
        if pred.shape != target.shape:
            raise ValueError('Prediction and target shapes differ')
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        if not np.equal(pred, np.floor(pred)).all() or not np.equal(target, np.floor(target)).all():
            raise ValueError('Class labels must be integers')
        if np.any((target < 0) | (target >= self.num_classes) | (pred < 0) | (pred >= self.num_classes)):
            raise ValueError('Class label out of range')
        indices = self.num_classes * target.astype(np.int64) + pred.astype(np.int64)
        self.confusion_matrix += np.bincount(indices, minlength=self.num_classes ** 2).reshape(self.confusion_matrix.shape)

    def compute(self):
        matrix = self.confusion_matrix.astype(np.float64)
        actual, predicted, diagonal = matrix.sum(1), matrix.sum(0), matrix.diagonal()
        total = matrix.sum()
        union = actual + predicted - diagonal
        iou = np.divide(diagonal, union, out=np.full_like(diagonal, np.nan), where=union > 0)
        recall = np.divide(diagonal, actual, out=np.full_like(diagonal, np.nan), where=actual > 0)
        observed = diagonal.sum() / total if total else np.nan
        expected = np.dot(actual, predicted) / total ** 2 if total else np.nan
        kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
        def mean(values):
            return float(np.nanmean(values) * 100) if np.isfinite(values).any() else None
        return {
            'confusion_matrix': self.confusion_matrix.tolist(),
            'class_names': list(self.class_names),
            'iou_percent': [float(v * 100) if np.isfinite(v) else None for v in iou],
            'recall_percent': [float(v * 100) if np.isfinite(v) else None for v in recall],
            'foreground_miou_percent': float(iou[1:].mean() * 100) if np.isfinite(iou[1:]).all() else None,
            'all_class_miou_percent': mean(iou),
            'mpa_percent': mean(recall),
            'pixel_accuracy_percent': float(observed * 100) if np.isfinite(observed) else None,
            'kappa': float(kappa) if np.isfinite(kappa) else None,
            'valid_pixels': int(total),
        }


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total / 1e6, trainable / 1e6


def benchmark(model, device, iterations=100, warmup=10):
    if iterations < 1 or warmup < 0:
        raise ValueError('Invalid timing iteration count')
    device = torch.device(device)
    sample = torch.zeros(1, model.n_channels, 512, 512, device=device)
    model.eval()
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize()
        elapsed = []
        for _ in range(iterations):
            synchronize()
            start = time.perf_counter()
            model(sample)
            synchronize()
            elapsed.append(time.perf_counter() - start)
    seconds = float(np.mean(elapsed))
    return {'fps': 1 / seconds, 'inference_ms': seconds * 1000,
            'input_shape': list(sample.shape), 'dtype': str(sample.dtype),
            'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
            'iterations': iterations}


def calculate_class_iou(pred, target, num_classes=3):
    metric = SegmentationMetrics(num_classes)
    metric.update(pred, target)
    return metric.compute()['iou_percent']


def calculate_miou(pred, target, num_classes=3):
    metric = SegmentationMetrics(num_classes)
    metric.update(pred, target)
    return metric.compute()['all_class_miou_percent']


def calculate_mpa(pred, target, num_classes=3):
    metric = SegmentationMetrics(num_classes)
    metric.update(pred, target)
    return metric.compute()['mpa_percent']
