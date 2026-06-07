"""Validation utilities: compute precision, recall, mAP, and confusion matrix.

This module provides a CLI wrapper `validate_model_cli()` which runs a validation
using Ultralytics' `model.val()` when available, and computes a confusion matrix
by running predictions on the validation set and matching to ground-truth labels.
"""
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from utils import latest_run_dir
import glob
import cv2


ROOT = Path(__file__).parent


def parse_label_file(label_path):
    # YOLO txt format: class x_center y_center w h (normalized)
    items = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                items.append(cls)
    return items


def prompt_existing_path(prompt: str, default: str) -> str:
    value = input(prompt).strip() or default
    while not Path(value).exists():
        print(f"File not found: {value}")
        value = input(prompt).strip() or default
    return value


def prompt_existing_yaml(prompt: str, default: str) -> str:
    value = input(prompt).strip() or default
    while True:
        path = Path(value)
        if not path.exists():
            print(f"File not found: {value}")
        elif path.suffix.lower() not in ['.yaml', '.yml']:
            print(f"Not a YAML file: {value}")
        else:
            return str(path)
        value = input(prompt).strip() or default


def validate_model_cli():
    print('\n== Validasi Model ==')
    weights = input('Path to weights [runs/best_*.pt or runs/best.pt]: ').strip() or ''
    if not weights:
        # try to pick most recent best under runs/
        run = latest_run_dir(ROOT / 'runs')
        if run and (run / 'weights' / 'best.pt').exists():
            weights = str(run / 'weights' / 'best.pt')
        else:
            print('No weights provided and no best.pt found.')
            return

    data_yaml = prompt_existing_yaml('Path to data.yaml [data.yaml]: ', str(ROOT / 'data.yaml'))

    print('Validating with weights:', weights)
    model = YOLO(weights)

    def extract_metrics(result):
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        if hasattr(result, 'metrics'):
            return getattr(result, 'metrics')
        if isinstance(result, (list, tuple)) and result:
            first = result[0]
            if isinstance(first, dict):
                return first
            if hasattr(first, 'metrics'):
                return getattr(first, 'metrics')
        return {}

    # Try ultralytics built-in validation first (returns metrics)
    try:
        metrics = model.val(data=data_yaml, verbose=True)
        print('Validation metrics (Ultralytics):')
        stats = extract_metrics(metrics)
        for key in ['metrics/precision', 'metrics/recall', 'metrics/mAP50', 'metrics/mAP50-95']:
            if key in stats:
                print(f"{key}: {stats[key]:.4f}")
        if 'precision' in stats:
            print(f"Precision: {stats['precision']:.4f}")
        if 'recall' in stats:
            print(f"Recall: {stats['recall']:.4f}")
        if 'mAP50' in stats:
            print(f"mAP50: {stats['mAP50']:.4f}")
        if 'mAP50-95' in stats:
            print(f"mAP50-95: {stats['mAP50-95']:.4f}")
    except Exception as e:
        print('Built-in validation failed or returned non-dict:', e)

    # Build confusion matrix by running model on validation images
    # Expect dataset layout: dataset/valid/images and dataset/valid/labels
    valid_images = list(Path('dataset') .glob('valid/images/**/*.*'))
    y_true = []
    y_pred = []

    if not valid_images:
        print('No images found under dataset/valid/images - skipping confusion matrix.')
        return

    print('Computing confusion matrix on validation set (this may take time)...')
    for img_path in valid_images:
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        lbl_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
        gt_classes = parse_label_file(lbl_path) if lbl_path.exists() else []

        preds = model.predict(source=str(img_path), conf=0.25, verbose=False)
        # preds is a list of Results; take classes of highest conf detection if any
        det_classes = []
        try:
            res = preds[0]
            boxes = res.boxes
            if boxes is not None:
                for b in boxes:
                    det_cls = int(b.cls.cpu().numpy())
                    det_classes.append(det_cls)
        except Exception:
            det_classes = []

        # For simple confusion matrix: if multiple GT or pred, extend with labels
        if not gt_classes and not det_classes:
            continue
        if not gt_classes:
            # all predictions are false positives: consider true label = -1
            for p in det_classes:
                y_true.append(-1)
                y_pred.append(p)
        elif not det_classes:
            for g in gt_classes:
                y_true.append(g)
                y_pred.append(-1)
        else:
            # pair up first N
            N = max(len(gt_classes), len(det_classes))
            for i in range(N):
                tg = gt_classes[i] if i < len(gt_classes) else -1
                tp = det_classes[i] if i < len(det_classes) else -1
                y_true.append(tg)
                y_pred.append(tp)

    if not y_true:
        print('No targets/predictions collected; cannot compute confusion matrix.')
        return

    labels = sorted(set([v for v in y_true + y_pred if v >= 0]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print('\nConfusion Matrix (rows=true, cols=pred):')
    print(cm)

    # Attempt simple summary metrics
    try:
        p = precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = recall_score(y_true, y_pred, average='macro', zero_division=0)
        print(f"Precision (macro): {p:.4f}")
        print(f"Recall (macro): {r:.4f}")
    except Exception as e:
        print('Could not compute precision/recall summary:', e)


if __name__ == '__main__':
    validate_model_cli()
