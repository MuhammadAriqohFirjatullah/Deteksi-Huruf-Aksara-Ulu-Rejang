import os
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

# Folder default untuk menyimpan model dan hasil prediksi
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
PREDICTIONS_DIR = ROOT_DIR / "predictions"
DATA_YAML = ROOT_DIR / "data.yaml"
DATA_RESOLVED_YAML = ROOT_DIR / "data_resolved.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def ensure_directories() -> None:
    """Buat folder models dan predictions otomatis jika belum ada."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def safe_int_input(prompt: str, default: int) -> int:
    """Menerima input integer dengan default bila Enter ditekan."""
    while True:
        value = input(prompt).strip()
        if not value:
            return default
        if value.isdigit():
            return int(value)
        try:
            number = int(value)
            return number
        except ValueError:
            print("Input tidak valid. Masukkan angka atau tekan Enter untuk default.")


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Coba beberapa cara resolve path relatif atau absolut dari data.yaml."""
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate

    candidate = (base_dir / path_str).resolve()
    if candidate.exists():
        return candidate

    candidate2 = (base_dir / Path(path_str).name).resolve()
    if candidate2.exists():
        return candidate2

    cleaned = path_str.replace("..\\", "").replace("../", "").lstrip("./\\")
    candidate3 = (base_dir / cleaned).resolve()
    if candidate3.exists():
        return candidate3

    candidate4 = (ROOT_DIR / cleaned).resolve()
    if candidate4.exists():
        return candidate4

    return candidate


def prepare_data_yaml() -> Path:
    """Baca data.yaml dan perbaiki path train/val/test bila perlu."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"File data.yaml tidak ditemukan di {DATA_YAML}")

    with open(DATA_YAML, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("data.yaml tidak valid. Harus berisi dictionary yaml.")

    changed = False
    for key in ("train", "val", "test"):
        path_value = data.get(key)
        if not path_value:
            continue
        resolved = resolve_path(str(path_value), DATA_YAML.parent)
        if resolved.exists():
            new_value = str(resolved)
            if new_value != str(path_value):
                data[key] = new_value
                changed = True

    if changed or not DATA_RESOLVED_YAML.exists():
        with open(DATA_RESOLVED_YAML, "w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False)
        return DATA_RESOLVED_YAML

    return DATA_YAML


def load_data_yaml() -> dict:
    """Muat isi data.yaml menjadi dictionary."""
    config_path = prepare_data_yaml()
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def count_images(folder: Path) -> int:
    """Hitung jumlah gambar di folder dan subfolder."""
    if not folder.exists() or not folder.is_dir():
        return 0
    count = 0
    for path in folder.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            count += 1
    return count


def get_dataset_path(key: str, data: dict) -> Path:
    """Dapatkan jalur dataset dari konfigurasi data.yaml dengan fallback ke struktur folder.
    """
    if key in data:
        candidate = resolve_path(str(data[key]), DATA_YAML.parent)
        if candidate.exists():
            return candidate

    fallback = ROOT_DIR / key
    if fallback.exists():
        return fallback

    fallback_images = ROOT_DIR / key / "images"
    if fallback_images.exists():
        return fallback_images

    return ROOT_DIR / key


def print_dataset_info() -> None:
    """Tampilkan informasi dataset berdasarkan data.yaml dan struktur folder."""
    try:
        data = load_data_yaml()
    except Exception as exc:
        print(f"ERROR: Gagal membaca data.yaml. {exc}")
        return

    names = data.get("names") or []
    if isinstance(names, dict):
        class_list = [names[key] for key in sorted(names, key=int)]
    else:
        class_list = list(names)

    train_path = get_dataset_path("train", data)
    val_path = get_dataset_path("val", data)
    test_path = get_dataset_path("test", data)

    print("\n# ==================================================")
    print("INFORMASI DATASET")
    print(f"Lokasi data.yaml : {DATA_YAML}")
    print(f"Jumlah kelas      : {len(class_list)}")
    print("Nama kelas        :")
    for label in class_list:
        print(f"  - {label}")
    print(f"Train images      : {count_images(train_path)} ({train_path})")
    print(f"Valid images      : {count_images(val_path)} ({val_path})")
    print(f"Test images       : {count_images(test_path)} ({test_path})")
    print("# ==================================================\n")


def get_class_distribution(folder: Path) -> dict[str, int]:
    """Hitung jumlah gambar per kelas dalam dataset klasifikasi folder."""
    distribution = {}
    if not folder.exists() or not folder.is_dir():
        return distribution

    for class_dir in sorted(folder.iterdir()):
        if class_dir.is_dir():
            count = sum(1 for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())
            distribution[class_dir.name] = count

    return distribution


def print_classifier_dataset_info() -> None:
    """Tampilkan informasi dataset klasifikasi Kaganga jika tersedia."""
    classifier_root = ROOT_DIR / "classifier_dataset"
    if not classifier_root.exists() or not classifier_root.is_dir():
        print("ERROR: Folder classifier_dataset tidak ditemukan.")
        return

    print("\n# ==================================================")
    print("INFORMASI CLASSIFIER DATASET")
    for split in ("train", "valid", "test"):
        split_folder = classifier_root / split
        if not split_folder.exists() or not split_folder.is_dir():
            print(f"{split.capitalize()} dataset tidak ditemukan: {split_folder}")
            continue

        distribution = get_class_distribution(split_folder)
        total_images = sum(distribution.values())
        class_count = len(distribution)

        print(f"\n{split.capitalize()} dataset:")
        print(f"  Lokasi        : {split_folder}")
        print(f"  Kelas         : {class_count}")
        print(f"  Total gambar  : {total_images}")
        if class_count > 0:
            smallest = min(distribution.values())
            largest = max(distribution.values())
            print(f"  Jumlah per kelas: terkecil={smallest}, terbesar={largest}")
            print("  Contoh kelas: ")
            example_classes = list(distribution.keys())[:10]
            for name in example_classes:
                print(f"    - {name}: {distribution[name]}")

    print("# ==================================================\n")


def find_best_checkpoint(training_dir: Path) -> Path:
    """Cari checkpoint best.pt di folder training."""
    if training_dir.exists():
        for path in training_dir.rglob("best.pt"):
            if path.is_file():
                return path
    return MODELS_DIR / "best.pt"


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_metrics(metrics) -> None:
    """Tampilkan metrik hasil training atau validasi."""
    if metrics is None or not hasattr(metrics, "box"):
        print("Tidak ada metrik yang dapat ditampilkan.")
        return

    box_metric = metrics.box
    precision = getattr(box_metric, "mp", lambda: None)()
    recall = getattr(box_metric, "mr", lambda: None)()
    map50 = getattr(box_metric, "map50", lambda: None)()
    map50_95 = getattr(box_metric, "map", lambda: None)()

    print("\n# ==================================================")
    print("HASIL METRIK")
    if precision is not None:
        print(f"Precision   : {format_percent(precision)}")
    if recall is not None:
        print(f"Recall      : {format_percent(recall)}")
    if map50 is not None:
        print(f"mAP50       : {format_percent(map50)}")
    if map50_95 is not None:
        print(f"mAP50-95    : {format_percent(map50_95)}")
    print("# ==================================================\n")


def train_model() -> None:
    """Mulai training model YOLOv11 menggunakan data.yaml."""
    try:
        data_path = prepare_data_yaml()
    except Exception as exc:
        print(f"ERROR: Gagal menyiapkan konfigurasi dataset. {exc}")
        return

    epochs = safe_int_input("Masukkan jumlah epoch (default 10): ", 10)
    imgsz = safe_int_input("Masukkan image size (default 416): ", 416)
    device = "cuda" if YOLO("yolo11n.pt").device == "cuda" else "cpu"

    print("\nMulai training model dengan YOLOv11... Harap tunggu.")
    try:
        model = YOLO("yolo11n.pt")
        # Tanyakan apakah ingin meniru augmentasi Roboflow
        use_rf = input("Samakan augmentasi seperti Roboflow? (Y/n): ").strip().lower()
        augment_kwargs = {}
        if use_rf in ("", "y", "yes"):
            # Pengaturan augmentasi berdasarkan informasi Roboflow pengguna
            # Rotation ±10°, Brightness ±10%, Blur ≤1.5 px, Noise ≤1.96%
            # YOLOv11 parameter augmentasi yang benar:
            augment_kwargs = {
                "degrees": 10,        # Rotation ±10°
                "hsv_v": 0.1,        # Brightness ±10%
                "flipud": 0.0,       # No vertical flip
                "fliplr": 0.0,       # No horizontal flip
                "mosaic": 1.0,       # Enable mosaic augmentation
            }

        train_kwargs = {
            "data": str(data_path),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": 16,           # Batch size: 16, 32, 64 sesuai GPU
            "project": str(MODELS_DIR),
            "name": "training",
            "exist_ok": True,
            "device": device,
        }
        
        # Merge augmentation kwargs jika ada
        if augment_kwargs:
            train_kwargs.update(augment_kwargs)
        
        metrics = model.train(**train_kwargs)
    except Exception as exc:
        print(f"ERROR: Training gagal. {exc}")
        return

    training_dir = MODELS_DIR / "training"
    best_checkpoint = find_best_checkpoint(training_dir)
    if best_checkpoint.exists():
        saved_best = MODELS_DIR / "best.pt"
        try:
            shutil.copy(best_checkpoint, saved_best)
            print(f"Model terbaik disimpan di: {saved_best}")
        except Exception as exc:
            print(f"Peringatan: Gagal menyalin best.pt. {exc}")
    else:
        print("Peringatan: best.pt tidak ditemukan setelah training.")

    print_metrics(metrics)


def validate_model() -> None:
    """Validasi model best.pt dengan dataset valid."""
    model_path = MODELS_DIR / "best.pt"
    if not model_path.exists():
        print("ERROR: Model terbaik belum tersedia. Jalankan training terlebih dahulu.")
        return

    try:
        data_path = prepare_data_yaml()
    except Exception as exc:
        print(f"ERROR: Gagal menyiapkan konfigurasi dataset. {exc}")
        return

    try:
        model = YOLO(str(model_path))
        metrics = model.val(data=str(data_path), imgsz=416, batch=16, device="cuda" if YOLO("yolo11n.pt").device == "cuda" else "cpu")
    except Exception as exc:
        print(f"ERROR: Validasi gagal. {exc}")
        return

    print_metrics(metrics)


def predict_one_image() -> None:
    """Prediksi satu gambar dan simpan hasil ke folder predictions."""
    image_input = input("Masukkan path/nama gambar: ").strip().strip('"')
    if not image_input:
        print("ERROR: Path gambar tidak boleh kosong.")
        return

    image_path = Path(image_input)
    if not image_path.exists() or not image_path.is_file():
        print(f"ERROR: Gambar tidak ditemukan: {image_path}")
        return

    model_path = MODELS_DIR / "best.pt"
    if not model_path.exists():
        print("ERROR: Model terbaik belum tersedia. Jalankan training terlebih dahulu.")
        return

    try:
        model = YOLO(str(model_path))
        results = model.predict(
            source=str(image_path),
            save=True,
            project=str(PREDICTIONS_DIR),
            name="single",
            exist_ok=True,
            imgsz=640,
        )
    except Exception as exc:
        print(f"ERROR: Prediksi gagal. {exc}")
        return

    if not results:
        print("ERROR: Tidak ada hasil prediksi.")
        return

    result = results[0]
    classes = []
    if hasattr(result, "boxes") and result.boxes is not None:
        boxes = result.boxes
        class_ids = boxes.cls.tolist() if len(boxes.cls) else []
        confidences = boxes.conf.tolist() if len(boxes.conf) else []
        for idx, conf in zip(class_ids, confidences):
            label = result.names[int(idx)] if idx in result.names else str(int(idx))
            classes.append((label, float(conf)))

    save_path = Path(result.save_dir) / image_path.name
    print("\n# ==================================================")
    print("HASIL DETEKSI")
    if classes:
        for label, conf in classes:
            print(f"Kelas      : {label}")
            print(f"Confidence : {conf * 100:.2f}%")
    else:
        print("Kelas      : Tidak ada deteksi")
        print("Confidence : -")
    print(f"Lokasi     : {save_path}")
    print("# ==================================================\n")


def predict_folder_images() -> None:
    """Prediksi seluruh gambar dalam folder dan simpan hasil ke folder predictions."""
    folder_input = input("Masukkan path folder gambar: ").strip().strip('"')
    if not folder_input:
        print("ERROR: Path folder tidak boleh kosong.")
        return

    folder_path = Path(folder_input)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"ERROR: Folder tidak ditemukan: {folder_path}")
        return

    model_path = MODELS_DIR / "best.pt"
    if not model_path.exists():
        print("ERROR: Model terbaik belum tersedia. Jalankan training terlebih dahulu.")
        return

    image_files = [p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    if not image_files:
        print(f"ERROR: Tidak ada gambar yang ditemukan di folder: {folder_path}")
        return

    try:
        model = YOLO(str(model_path))
        model.predict(
            source=str(folder_path),
            save=True,
            project=str(PREDICTIONS_DIR),
            name="folder",
            exist_ok=True,
            imgsz=640,
        )
        print(f"Prediksi semua gambar selesai. Hasil disimpan di: {PREDICTIONS_DIR / 'folder'}")
    except Exception as exc:
        print(f"ERROR: Prediksi folder gagal. {exc}")


def print_menu() -> None:
    print("\n# ==================================================")
    print("SISTEM DETEKSI AKSARA KAGANGA YOLOv11")
    print("\n1. Training Model")
    print("2. Validasi Model")
    print("3. Prediksi 1 Gambar")
    print("4. Prediksi Folder Gambar")
    print("5. Informasi Dataset")
    print("6. Informasi Classifier Dataset")
    print("7. Keluar")
    print("# ==================================================")


def main() -> None:
    ensure_directories()
    while True:
        print_menu()
        choice = input("Masukkan pilihan: ").strip()
        if choice == "1":
            train_model()
        elif choice == "2":
            validate_model()
        elif choice == "3":
            predict_one_image()
        elif choice == "4":
            predict_folder_images()
        elif choice == "5":
            print_dataset_info()
        elif choice == "6":
            print_classifier_dataset_info()
        elif choice == "7":
            print("Keluar dari aplikasi.")
            break
        else:
            print("Pilihan tidak valid. Masukkan angka 1 sampai 7.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")
    except Exception as general_exc:
        print(f"ERROR tidak terduga: {general_exc}")
