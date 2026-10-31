import argparse
import json
import math
from pathlib import Path

import numpy as np


def fuse_centres(left, right, ridge, width=512):
    left, right, ridge = [np.asarray(value, dtype=np.float64).copy() for value in (left, right, ridge)]
    if left.ndim != 1 or left.shape != right.shape or left.shape != ridge.shape:
        raise ValueError('Boundary arrays must be one-dimensional and equally sized')
    rows = np.arange(len(left), dtype=np.float64)
    for array in (left, right, ridge):
        array[~np.isfinite(array) | (array < 0) | (array > width - 1)] = np.nan
    left_valid, right_valid = np.isfinite(left), np.isfinite(right)
    bilateral = left_valid & right_valid & (left < right)
    centres = np.full_like(rows, np.nan)
    quality = np.zeros(len(rows), dtype=np.int64)
    centres[bilateral] = (left[bilateral] + right[bilateral]) / 2
    quality[bilateral] = 2
    support = rows[bilateral]
    eligible = np.zeros(len(rows), dtype=bool)
    predicted_width = np.full_like(rows, np.nan)
    if len(support) >= 30 and np.ptp(support) >= 100:
        coefficients = np.linalg.lstsq(np.column_stack([support, np.ones_like(support)]),
                                       (right - left)[bilateral], rcond=None)[0]
        predicted_width = coefficients[0] * rows + coefficients[1]
        eligible = (rows >= support.min() - 30) & (rows <= support.max() + 30)
        eligible &= np.isfinite(predicted_width) & (predicted_width > 0)
    unilateral = left_valid ^ right_valid
    completed = np.where(left_valid, left + predicted_width / 2, right - predicted_width / 2)
    accepted = unilateral & eligible & np.isfinite(completed) & (completed >= 0) & (completed <= width - 1)
    centres[accepted] = completed[accepted]
    with_ridge = accepted & np.isfinite(ridge)
    centres[with_ridge] = (completed[with_ridge] + ridge[with_ridge]) / 2
    quality[accepted] = 1
    fallback = ~bilateral & ~accepted & np.isfinite(ridge)
    centres[fallback] = ridge[fallback]
    return centres, quality


def segment_median(centres):
    values = np.asarray(centres, dtype=np.float64)
    result = np.full_like(values, np.nan)
    valid_rows = np.flatnonzero(np.isfinite(values))
    for segment in np.split(valid_rows, np.flatnonzero(np.diff(valid_rows) != 1) + 1):
        for index, row in enumerate(segment):
            result[row] = np.median(values[segment[max(0, index - 2):index + 3]])
    return result


def ransac_line(centres, max_trials=184, threshold=3.0, probability=0.99, seed=42):
    values = np.asarray(centres, dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(values)).astype(np.float64)
    if len(rows) < 2:
        return None
    x = values[rows.astype(int)]
    rng = np.random.default_rng(seed)
    best = None
    best_count = 0
    best_residual = np.inf
    trials, limit = 0, max_trials
    while trials < limit:
        first, second = rng.choice(len(rows), 2, replace=False)
        trials += 1
        slope = (x[second] - x[first]) / (rows[second] - rows[first])
        intercept = x[first] - slope * rows[first]
        residual = np.abs(x - (slope * rows + intercept))
        inliers = residual <= threshold
        count = int(inliers.sum())
        score = float(residual[inliers].sum())
        if count > best_count or (count == best_count and score < best_residual):
            best, best_count, best_residual = inliers, count, score
            fraction = count / len(rows)
            if fraction >= 1:
                limit = trials
            elif fraction > 0:
                requirement = math.ceil(math.log1p(-probability) / math.log1p(-fraction ** 2))
                limit = min(limit, max(trials, requirement))
    if best is None or best_count < 2:
        return None
    slope, intercept = np.linalg.lstsq(np.column_stack([rows[best], np.ones(best_count)]), x[best], rcond=None)[0]
    return {'slope': float(slope), 'intercept': float(intercept),
            'inlier_rows': rows[best].astype(int).tolist(), 'trials': trials,
            'valid_rows': len(rows)}


def fit_navigation_line(left, right, ridge):
    centres, quality = fuse_centres(left, right, ridge)
    smoothed = segment_median(centres)
    return {'line': ransac_line(smoothed), 'centres': centres,
            'smoothed_centres': smoothed, 'quality': quality}


def scale_intrinsics(intrinsics, original_width, original_height):
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or original_width <= 0 or original_height <= 0 or not np.isfinite(matrix).all():
        raise ValueError('Invalid camera geometry')
    return np.diag([512 / original_width, 512 / original_height, 1]) @ matrix


def ground_homography(intrinsics, world_to_camera, camera_center):
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    basis = np.asarray(world_to_camera, dtype=np.float64)
    center = np.asarray(camera_center, dtype=np.float64)
    if intrinsic.shape != (3, 3) or basis.shape != (3, 3) or center.shape != (3,):
        raise ValueError('Invalid matrix dimensions')
    matrix = intrinsic @ np.column_stack([basis[:, 0], basis[:, 2], -basis @ center])
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Singular or nonfinite homography')
    return matrix


def project_ground(line, homography, rows=None, forward=(1.0, 4.0), transverse=(-0.8, 0.8)):
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid homography')
    rows = np.arange(512, dtype=np.float64) if rows is None else np.asarray(rows, dtype=np.float64)
    image_x = line['slope'] * rows + line['intercept']
    homogeneous = np.linalg.solve(matrix, np.vstack([image_x, rows, np.ones_like(rows)]))
    denominator = homogeneous[2]
    ground = np.divide(homogeneous[:2], denominator, out=np.full_like(homogeneous[:2], np.nan), where=np.abs(denominator) > 1e-12)
    depth = matrix[2] @ np.vstack([ground, np.ones_like(rows)])
    valid = np.isfinite(ground).all(axis=0) & (depth > 0)
    valid &= (image_x >= 0) & (image_x <= 511) & (rows >= 0) & (rows <= 511)
    valid &= (ground[0] >= transverse[0]) & (ground[0] <= transverse[1])
    valid &= (ground[1] >= forward[0]) & (ground[1] <= forward[1])
    return ground, valid


def fit_ground_line(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 2:
        return None
    if not np.isfinite(points).all() or np.ptp(points[1]) < 1e-9:
        return None
    slope, intercept = np.linalg.lstsq(np.column_stack([points[1], np.ones(points.shape[1])]), points[0], rcond=None)[0]
    return {'slope': float(slope), 'intercept': float(intercept)}


def horizontal_line_error(predicted, reference):
    rows = np.arange(512, dtype=np.float64)
    return float(np.abs((predicted['slope'] - reference['slope']) * rows +
                       predicted['intercept'] - reference['intercept']).mean())


def angular_line_error(predicted, reference, homography):
    pred_points, pred_valid = project_ground(predicted, homography)
    ref_points, ref_valid = project_ground(reference, homography)
    common = pred_valid & ref_valid
    pred_ground = fit_ground_line(pred_points[:, common])
    ref_ground = fit_ground_line(ref_points[:, common])
    if pred_ground is None or ref_ground is None:
        return None
    return abs(math.degrees(math.atan(pred_ground['slope']) - math.atan(ref_ground['slope'])))


def lateral_offset_mm(ground_line):
    return 1000 * abs(ground_line['intercept']) / math.hypot(1, ground_line['slope'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundaries', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with np.load(args.boundaries, allow_pickle=False) as arrays:
        result = fit_navigation_line(arrays['left'], arrays['right'], arrays['ridge'])
    for key in ('centres', 'smoothed_centres'):
        result[key] = [float(value) if np.isfinite(value) else None for value in result[key]]
    result['quality'] = result['quality'].tolist()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
