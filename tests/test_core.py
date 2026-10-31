import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from unet import UNet
from unet.tgsm import EnhancedTGSM, EnhancedTopologyConfidence
from unet.unet_parts import DACG_Block, DSEM
from utils.data_loading import BasicDataset, decode_mask, list_images, preprocess_image
from utils.metrics import SegmentationMetrics, benchmark
from utils.runtime import load_model
from navigation import (
    angular_line_error, fit_ground_line, fit_navigation_line, fuse_centres,
    ground_homography, horizontal_line_error, lateral_offset_mm,
    project_ground, ransac_line, scale_intrinsics, segment_median,
)


torch.set_num_threads(2)


def test_mask_labels():
    labels = np.array([[0, 1, 2, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(decode_mask(Image.fromarray(labels)), labels)
    gray = np.array([[0, 85, 170, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(decode_mask(Image.fromarray(gray), 'grayscale'), labels)
    with pytest.raises(ValueError):
        decode_mask(Image.fromarray(gray))


def test_palette_mask_indices():
    image = Image.fromarray(np.array([[0, 1, 2]], dtype=np.uint8)).convert('P')
    image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (768 - 9))
    np.testing.assert_array_equal(decode_mask(image), [[0, 1, 2]])


def test_fixed_input_size():
    image = Image.new('RGB', (2844, 1280), (255, 0, 0))
    tensor = preprocess_image(image)
    assert tensor.shape == (3, 512, 512)
    assert tensor[0].eq(1).all() and tensor[1:].eq(0).all()


def test_dataset_manifest_and_missing_mask(tmp_path):
    images, masks = tmp_path / 'images', tmp_path / 'masks'
    images.mkdir()
    masks.mkdir()
    for name in ('a', 'b'):
        Image.new('RGB', (16, 8)).save(images / f'{name}.png')
        Image.fromarray(np.ones((8, 16), dtype=np.uint8)).save(masks / f'{name}.png')
    manifest = tmp_path / 'split.json'
    manifest.write_text(json.dumps({'test': ['b.png']}))
    dataset = BasicDataset(images, masks, manifest)
    assert len(dataset) == 1 and dataset[0]['name'] == 'b.png'
    assert dataset[0]['mask'].eq(1).all()
    (masks / 'b.png').unlink()
    with pytest.raises(FileNotFoundError):
        BasicDataset(images, masks, manifest)


def test_duplicate_file_list_rejected(tmp_path):
    Image.new('RGB', (3, 3)).save(tmp_path / 'a.png')
    listing = tmp_path / 'test.txt'
    listing.write_text('a.png\na.png\n')
    with pytest.raises(ValueError):
        list_images(tmp_path, listing)


def test_confusion_metrics_and_ignored_pixels():
    matrix = np.array([[4, 1, 0], [0, 3, 1], [1, 0, 2]])
    truth, prediction = [], []
    for i in range(3):
        for j in range(3):
            truth.extend([i] * matrix[i, j])
            prediction.extend([j] * matrix[i, j])
    truth.append(255)
    prediction.append(2)
    metric = SegmentationMetrics()
    metric.update(np.array(prediction), np.array(truth))
    result = metric.compute()
    np.testing.assert_array_equal(result['confusion_matrix'], matrix)
    np.testing.assert_allclose(result['iou_percent'], [100 * 4 / 6, 60, 50])
    assert result['foreground_miou_percent'] == pytest.approx(55)
    assert result['mpa_percent'] == pytest.approx((4 / 5 + 3 / 4 + 2 / 3) / 3 * 100)
    assert result['kappa'] == pytest.approx((9 / 12 - 50 / 144) / (1 - 50 / 144))
    assert result['valid_pixels'] == 12


def test_absent_foreground_is_not_perfect():
    metric = SegmentationMetrics()
    metric.update(np.zeros(4), np.zeros(4))
    assert metric.compute()['foreground_miou_percent'] is None
    json.dumps(metric.compute(), allow_nan=False)


def test_dsem_scaling_before_projection():
    block = DSEM(16).eval()
    block.alpha.data.fill_(0.25)
    x = torch.randn(1, 16, 9, 13)
    captured = {}
    hook = block.fusion.register_forward_pre_hook(lambda _, args: captured.update(input=args[0].detach()))
    with torch.no_grad():
        expected = block.norm(torch.cat([conv(x) for conv in block.dir_convs], 1)) * .25
        block(x)
    hook.remove()
    torch.testing.assert_close(captured['input'], expected)


def test_tgsm_residual_uses_confidence():
    module = EnhancedTGSM(12).eval()
    x = torch.randn(1, 12, 8, 12)
    captured = {}
    hook = module.proj.register_forward_pre_hook(lambda _, args: captured.update(input=args[0].detach()))
    with torch.no_grad():
        output, confidence = module(x)
        torch.testing.assert_close(output, x + .5 * module.proj(confidence))
    hook.remove()
    assert confidence.shape == (1, 1, 8, 12)
    torch.testing.assert_close(captured['input'], confidence)


def test_confidence_pooling_and_bounds():
    module = EnhancedTopologyConfidence()
    x = torch.zeros(1, 4, 5, 7)
    confidence = module(x)
    expected = .5 * torch.sigmoid(torch.tensor(-.3))
    torch.testing.assert_close(confidence, torch.full_like(confidence, expected))
    assert confidence.min() >= 0 and confidence.max() <= 1


def test_dacg_local_and_prior():
    block = DACG_Block(24, 16).eval()
    assert block.local_branch.kernel_size == (1, 1)
    assert block.local_branch.groups == 16
    x = torch.randn(1, 24, 8, 12)
    with torch.no_grad():
        without = block(x)
        block.topo_weight.zero_()
        zero_weight = block(x, torch.rand(1, 1, 2, 3))
    torch.testing.assert_close(without, zero_weight)


def test_direction_modules_gradients():
    tgsm = EnhancedTGSM(16)
    decoder = DACG_Block(24, 16)
    encoder = DSEM(16)
    source = torch.randn(2, 16, 8, 12, requires_grad=True)
    enhanced, confidence = tgsm(encoder(source))
    result = decoder(torch.cat([enhanced, enhanced[:, :8]], dim=1), confidence)
    result.square().mean().backward()
    for parameter in [source, tgsm.proj.weight, tgsm.alpha, decoder.topo_weight, encoder.alpha]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize('dsem,dacg,tgsm', [
    (False, False, False), (False, False, True), (True, False, False),
    (False, True, False), (True, True, False), (True, True, True)])
def test_ablation_output_shapes(dsem, dacg, tgsm):
    model = UNet(use_dsem=dsem, use_dacg=dacg, use_tgsm=tgsm).eval()
    captured = {}
    hook = model.down4.register_forward_hook(lambda _, args, output: captured.update(shape=output.shape))
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 32, 48))
    hook.remove()
    assert output.shape == (1, 3, 32, 48)
    assert captured['shape'][1] == 1024
    assert torch.isfinite(output).all()


def test_checkpoint_roundtrip_and_mismatch(tmp_path):
    model = UNet(use_dsem=False, use_dacg=False, use_tgsm=True).eval()
    path = tmp_path / 'model.pth'
    state = model.state_dict()
    torch.save(state, path)
    restored = load_model(path)
    assert restored.get_config_summary() == model.get_config_summary()
    state['tgsm.proj.weight'] = torch.zeros(1024, 1024, 1, 1)
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        load_model(path)


def test_benchmark_input_size():
    class Probe(torch.nn.Module):
        n_channels = 3
        def forward(self, x):
            assert x.shape == (1, 3, 512, 512)
            return x
    result = benchmark(Probe(), 'cpu', iterations=2, warmup=1)
    assert result['fps'] * result['inference_ms'] == pytest.approx(1000)


def test_width_completion_and_missing_ridge():
    rows = np.arange(512)
    left = np.full(512, np.nan)
    right = np.full(512, np.nan)
    ridge = np.full(512, np.nan)
    left[100:301] = 100
    right[100:301] = 200
    left[70:100] = 100
    left[40:70] = 100
    ridge[80:90] = 170
    centres, quality = fuse_centres(left, right, ridge)
    np.testing.assert_allclose(centres[100:301], 150)
    np.testing.assert_allclose(centres[70:80], 150)
    np.testing.assert_allclose(centres[80:90], 160)
    assert np.isnan(centres[40:70]).all()
    assert quality[100] == 2 and quality[80] == 1


def test_short_support_falls_back_to_ridge():
    left = np.full(512, np.nan)
    right = left.copy()
    left[:20], right[:20] = 100, 200
    left[20:30] = 100
    centres, _ = fuse_centres(left, right, np.full(512, 170))
    assert centres[10] == 150 and centres[25] == 170


def test_smoothing_does_not_bridge_gaps():
    values = np.array([0, 0, 100, np.nan, 500, 500])
    np.testing.assert_allclose(segment_median(values), [0, 0, 0, np.nan, 500, 500], equal_nan=True)


def test_ransac_fit_and_empty_input():
    rows = np.arange(512)
    centres = .1 * rows + 200
    centres[::7] += 40
    result = ransac_line(centres)
    assert result['slope'] == pytest.approx(.1)
    assert result['intercept'] == pytest.approx(200)
    assert 1 <= result['trials'] <= 184
    assert ransac_line([np.nan, 1, np.nan]) is None


def test_navigation_bilateral_fit():
    rows = np.arange(512)
    result = fit_navigation_line(100 + .1 * rows, 300 + .1 * rows, np.full(512, np.nan))
    assert result['line']['slope'] == pytest.approx(.1, abs=1e-5)
    assert result['line']['intercept'] == pytest.approx(200, abs=.01)


def test_camera_matrix_and_ground_metrics():
    angle = np.pi / 6
    intrinsic = scale_intrinsics([[686.3, 0, 640], [0, 1229.4, 640], [0, 0, 1]], 1280, 1280)
    basis = np.array([[1, 0, 0], [0, -np.cos(angle), -np.sin(angle)],
                      [0, -np.sin(angle), np.cos(angle)]])
    homography = ground_homography(intrinsic, basis, [0, .8, 0])
    np.testing.assert_allclose(homography, [[274.52, 221.702503, 102.4],
        [0, -24.177497, 443.101322], [0, .8660254, .4]], atol=1e-6)
    centre = {'slope': 0, 'intercept': 256}
    points, valid = project_ground(centre, homography)
    fitted = fit_ground_line(points[:, valid])
    assert abs(fitted['intercept']) < 1e-10
    assert angular_line_error(centre, centre, homography) < 1e-10
    assert lateral_offset_mm({'slope': .75, 'intercept': .01}) == pytest.approx(8)
    assert horizontal_line_error(centre, {'slope': 0, 'intercept': 256.85}) == pytest.approx(.85)


@pytest.mark.parametrize('script', ['predict.py', 'eval_fixed.py', 'inference_complete.py',
    'inference_compatible_fixed.py', 'detect_config.py', 'navigation.py'])
def test_cli_help(script):
    result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
