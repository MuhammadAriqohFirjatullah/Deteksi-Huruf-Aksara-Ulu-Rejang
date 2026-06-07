"""Reporting utilities: parse YOLO `results.csv` and create plots + summary.

Generates plots for training loss, validation loss, mAP50, precision, recall,
and writes a short auto-report to `laporan/` including dataset counts and
performance numbers.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dataset_info import collect_classes, count_images
import yaml


ROOT = Path(__file__).parent


def find_latest_results_csv(runs_root: Path = None):
    if runs_root is None:
        runs_root = ROOT / 'runs'
    runs = [p for p in runs_root.iterdir() if p.is_dir()]
    runs_sorted = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs_sorted:
        csv = r / 'results.csv'
        if csv.exists():
            return csv
    return None


def plot_metrics(results_csv: Path, out_dir: Path):
    df = pd.read_csv(results_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to plot common columns if present
    if 'epoch' in df.columns and 'box' in df.columns:
        plt.figure()
        if 'box' in df.columns:
            plt.plot(df['epoch'], df['box'], label='box_loss')
        if 'cls' in df.columns:
            plt.plot(df['epoch'], df['cls'], label='cls_loss')
        plt.legend()
        plt.title('Training losses')
        plt.savefig(out_dir / 'train_loss.png')

    # validation mAP
    if 'mAP50' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['mAP50'], label='mAP50')
        plt.legend()
        plt.title('mAP50 over epochs')
        plt.savefig(out_dir / 'mAP50.png')

    # Precision & Recall
    if 'precision' in df.columns or 'recall' in df.columns:
        plt.figure()
        if 'precision' in df.columns:
            plt.plot(df['epoch'], df['precision'], label='Precision')
        if 'recall' in df.columns:
            plt.plot(df['epoch'], df['recall'], label='Recall')
        plt.legend()
        plt.title('Precision & Recall')
        plt.savefig(out_dir / 'prec_recall.png')


def generate_report_cli():
    print('\n== Generate Report ==')
    ds = ROOT / 'dataset'
    classes = collect_classes(ds)
    n_train = count_images(ds / 'train') if (ds / 'train').exists() else 0
    n_valid = count_images(ds / 'valid') if (ds / 'valid').exists() else 0
    n_test = count_images(ds / 'test') if (ds / 'test').exists() else 0

    csv = find_latest_results_csv()
    laporan_dir = ROOT / 'laporan'
    laporan_dir.mkdir(parents=True, exist_ok=True)

    if csv:
        print('Found results.csv at', csv)
        try:
            plot_metrics(csv, laporan_dir)
            df = pd.read_csv(csv)
            # Try to extract final metrics
            last = df.iloc[-1]
            prec = last.get('precision', None)
            rec = last.get('recall', None)
            mAP50 = last.get('mAP50', None)
            mAP5095 = last.get('mAP50-95', None)
        except Exception as e:
            print('Could not read/plot results.csv:', e)
            prec = rec = mAP50 = mAP5095 = None
    else:
        print('No results.csv found under runs/. Skipping metric plots.')
        prec = rec = mAP50 = mAP5095 = None

    # Write a short markdown report
    report_md = laporan_dir / 'report.md'
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write('# Laporan Performa Model\n\n')
        f.write(f'* Jumlah kelas: {len(classes)}\n')
        f.write(f'* Jumlah gambar train: {n_train}\n')
        f.write(f'* Jumlah gambar valid: {n_valid}\n')
        f.write(f'* Jumlah gambar test: {n_test}\n')
        f.write(f'* Precision: {prec}\n')
        f.write(f'* Recall: {rec}\n')
        f.write(f'* mAP50: {mAP50}\n')
        f.write(f'* mAP50-95: {mAP5095}\n')
        f.write('\n## Kesimpulan\n')
        if mAP50 is not None and mAP50 > 0.5:
            f.write('Model menunjukkan performa baik pada mAP50 > 0.5.\n')
        else:
            f.write('Model memerlukan peningkatan — periksa augmentasi, epochs, dan data balance.\n')

    print('Report saved to', report_md)


if __name__ == '__main__':
    generate_report_cli()
