
from ultralytics import YOLO
from pathlib import Path
from utils import latest_run_dir, copy_weights_from_run
import time

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).parent


def get_default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return '0'
    return 'cpu'


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


def prompt_device(prompt: str, default: str = 'auto') -> str:
    current_default = default
    while True:
        value = input(f'{prompt} [{current_default}]: ').strip() or current_default
        if value.lower() == 'auto':
            detected = get_default_device()
            print(f'Auto-selected device: {detected}')
            return detected
        if value.lower() == 'cpu':
            return 'cpu'
        if torch is None:
            print('Torch tidak ditemukan, hanya CPU yang tersedia.')
            current_default = 'cpu'
            continue
        if not torch.cuda.is_available():
            if value.isdigit():
                print('CUDA tidak tersedia di lingkungan ini. Input numerik GPU tidak dapat digunakan.')
            else:
                print('CUDA tidak tersedia di lingkungan ini. Gunakan "cpu" atau instal PyTorch dengan CUDA.')
            current_default = 'cpu'
            continue
        return value


def train_model():
    print('\n== Train Model ==')
    data_yaml = prompt_existing_yaml('Path to data.yaml [data.yaml]: ', str(ROOT / 'data.yaml'))
    epochs = input('Epochs [50]: ').strip() or '50'
    model_name = prompt_existing_path('Initial model file [yolo11n.pt]: ', str(ROOT / 'yolo11n.pt'))
    device = prompt_device('Device', 'auto')
    batch = input('Batch size [4]: ').strip() or '4'
    imgsz = input('Image size [416]: ').strip() or '416'

    try:
        epochs = int(epochs)
    except Exception:
        epochs = 50

    try:
        batch = int(batch)
    except Exception:
        batch = 4

    try:
        imgsz = int(imgsz)
    except Exception:
        imgsz = 416

    print(f"Starting training: model={model_name}, data={data_yaml}, epochs={epochs}, device={device}, batch={batch}, imgsz={imgsz}")
    run_name = f"train_{int(time.time())}"

    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        device=device,
        batch=batch,
        imgsz=imgsz,
        project=str(ROOT / 'runs'),
        name=run_name,
    )

    run_dir = latest_run_dir(ROOT / 'runs')
    if run_dir:
        best, last = copy_weights_from_run(run_dir)
        print('Training finished. Run folder:', run_dir)
        if best:
            print('Saved best weights to:', best)
        if last:
            print('Saved last weights to:', last)
    else:
        print('No run folder found after training.')


if __name__ == '__main__':
    train_model()
