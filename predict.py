"""Prediction utilities for single image and folder.

Draws bounding boxes with labels and confidence and saves results to
`hasil_prediksi/` as images and a small CSV summary.
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
INPUT_DIR = ROOT / 'input_images'
INPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_class_names():
    data_yaml = ROOT / 'data.yaml'
    if not data_yaml.exists():
        return []
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names', [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return names


def draw_boxes(img, boxes, class_names=None):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        label = f"{cls}:{conf:.2f}"
        if class_names and cls < len(class_names):
            label = f"{class_names[cls]}:{conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img


def make_prediction_rows(results, class_names):
    rows = []
    if results is None:
        return rows
    res = results[0]
    boxes = res.boxes
    if boxes is not None:
        for b in boxes:
            cls = int(b.cls[0])
            confs = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()
            class_name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            rows.append({
                'class_id': cls,
                'class_name': class_name,
                'confidence': confs,
                'x1': xyxy[0],
                'y1': xyxy[1],
                'x2': xyxy[2],
                'y2': xyxy[3],
            })
    return sorted(rows, key=lambda x: x['confidence'], reverse=True)


def show_prediction_dashboard(image_path, rows, output_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Gambar')

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    if rows:
        top = rows[0]
        text = [
            'HASIL PREDIKSI',
            '',
            f"Aksara: {top['class_name']}",
            f"Confidence: {top['confidence']:.4f} ({top['confidence']*100:.2f}%)",
            '',
            'TOP 5 PREDICTIONS:',
        ]
        for i, row in enumerate(rows[:5], start=1):
            text.append(f"{i}. {row['class_name']}: {row['confidence']:.4f}")
    else:
        text = [
            'HASIL PREDIKSI',
            '',
            'Tidak ada deteksi.',
            '',
            'Coba turunkan threshold atau cek kembali gambar input.',
        ]
    ax2.text(0, 1, '\n'.join(text), va='top', fontsize=12, family='monospace')
    fig.tight_layout()
    plt.savefig(output_path)
    plt.show()


def predict_image(weights, image_path, conf=0.25, save_dir=None):
    class_names = load_class_names()
    model = YOLO(weights)
    thresholds = [conf, 0.1, 0.05, 0.01]
    results = None
    rows = []

    for threshold in thresholds:
        results = model.predict(source=str(image_path), conf=threshold, imgsz=640, max_det=20, verbose=False)
        rows = make_prediction_rows(results, class_names)
        if rows:
            break

    if not rows and results is not None:
        # Jika tetap tidak ada hasil, coba dengan augmentasi untuk deteksi yang lebih sensitif.
        results = model.predict(source=str(image_path), conf=0.01, imgsz=640, max_det=20, augment=True, verbose=False)
        rows = make_prediction_rows(results, class_names)

    img = cv2.imread(str(image_path))
    boxes = results[0].boxes if results else None
    img_out = draw_boxes(img, boxes if boxes is not None else [], class_names=class_names)
    if save_dir is None:
        save_dir = ROOT / 'hasil_prediksi' / 'single'
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / Path(image_path).name
    cv2.imwrite(str(out_path), img_out)

    if rows:
        df = pd.DataFrame([{
            'image': Path(image_path).name,
            'class_id': row['class_id'],
            'class_name': row['class_name'],
            'confidence': row['confidence'],
            'x1': row['x1'],
            'y1': row['y1'],
            'x2': row['x2'],
            'y2': row['y2'],
        } for row in rows])
        df.to_csv(Path(save_dir) / (Path(image_path).stem + '_preds.csv'), index=False)

    return out_path, rows


def predict_folder(weights, folder_path, conf=0.25, save_dir=None):
    p = Path(folder_path)
    images = [x for x in p.glob('*') if x.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    saved = []
    for im in images:
        out, _ = predict_image(weights, im, conf=conf, save_dir=save_dir)
        saved.append(out)
    return saved


def predict_image_cli():
    print('\n== Prediksi 1 Gambar ==')
    weights = input('Path to weights [runs/best_*.pt]: ').strip() or 'runs/best.pt'
    default_image = ''
    default_files = [x for x in INPUT_DIR.iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if default_files:
        default_image = str(default_files[0])
    prompt = f'Path to image [input_images/]: '
    img = input(prompt).strip() or default_image
    if not img:
        print('Tidak ada gambar dipilih dan folder input_images kosong.')
        print('Silakan letakkan file .jpg atau .png di folder input_images atau masukkan path lengkap.')
        return
    out, rows = predict_image(weights, img)
    if rows:
        print('Deteksi ditemukan:')
        for row in rows[:5]:
            print(f"- {row['class_name']} (class {row['class_id']}): {row['confidence']:.4f}")
    else:
        print('Tidak ada deteksi pada gambar ini. Coba gunakan gambar lain atau latih model lebih lanjut.')

    report_path = ROOT / 'hasil_prediksi' / 'single' / (Path(img).stem + '_result.png')
    show_prediction_dashboard(img, rows, report_path)
    print('Hasil annotasi disimpan di:', out)
    print('Dashboard prediksi disimpan di:', report_path)


def predict_folder_cli():
    print('\n== Prediksi Folder Gambar ==')
    weights = input('Path to weights [runs/best_*.pt]: ').strip() or 'runs/best.pt'
    folder = input('Path to folder images: ').strip()
    if not folder:
        print('Tidak ada folder dipilih.')
        return
    outs = predict_folder(weights, folder)
    print('Selesai. Hasil disimpan untuk', len(outs), 'gambar')


if __name__ == '__main__':
    predict_image_cli()
