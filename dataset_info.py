"""Dataset inspection utilities.

Counts number of classes and images per split. Assumes YOLO detection dataset organized as:
dataset/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/

Also creates a `data.yaml` if missing with paths for the organized dataset.
"""
from pathlib import Path
import yaml


ROOT = Path(__file__).parent


def collect_classes(dataset_root: Path):
    # For YOLO detection datasets, class names are stored in data.yaml.
    data_yaml = dataset_root.parent / 'data.yaml'
    if data_yaml.exists():
        try:
            import yaml
            content = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
            if isinstance(content, dict) and 'names' in content:
                return list(content['names'])
        except Exception:
            pass

    # Fallback: scan label files for class IDs under train/labels/.
    labels_dir = dataset_root / 'train' / 'labels'
    classes = set()
    if labels_dir.exists():
        for p in sorted(labels_dir.rglob('*.txt')):
            with open(p, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            classes.add(int(parts[0]))
                        except ValueError:
                            pass
    return [str(c) for c in sorted(classes)]


def count_images(split_dir: Path):
    # Count image files under split_dir/images
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    image_root = split_dir / 'images'
    if not image_root.exists():
        return 0
    for p in image_root.rglob('*'):
        if p.suffix.lower() in exts:
            count += 1
    return count


def dataset_info_cli():
    print('\n== Informasi Dataset ==')
    ds = ROOT / 'dataset'
    if not ds.exists():
        print('Folder `dataset/` tidak ditemukan in project root.')
        return

    classes = collect_classes(ds)
    n_classes = len(classes)
    n_train = count_images(ds / 'train') if (ds / 'train').exists() else 0
    n_valid = count_images(ds / 'valid') if (ds / 'valid').exists() else 0
    n_test = count_images(ds / 'test') if (ds / 'test').exists() else 0

    print(f'Jumlah kelas: {n_classes}')
    print(f'Jumlah gambar train: {n_train}')
    print(f'Jumlah gambar valid: {n_valid}')
    print(f'Jumlah gambar test: {n_test}')

    # Create data.yaml if missing
    data_yaml = ROOT / 'data.yaml'
    if not data_yaml.exists():
        print('Membuat data.yaml otomatis berdasarkan folder kelas...')
        content = {
            'path': 'dataset',
            'train': 'dataset/train',
            'val': 'dataset/valid',
            'test': 'dataset/test',
            'nc': n_classes,
            'names': classes,
        }
        with open(data_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(content, f, allow_unicode=True)
        print('data.yaml dibuat di:', data_yaml)
    else:
        print('data.yaml sudah ada di project root.')


if __name__ == '__main__':
    dataset_info_cli()
