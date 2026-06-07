"""Main CLI application for Aksara Ulu Rejang detection project.

Provides a terminal menu to train, validate, predict, inspect dataset,
and generate reports. Uses Ultralytics YOLO (YOLO11n) as requested.

Comments are intentionally verbose to help with thesis explanations.
"""
import sys
from utils import ensure_dirs
from train import train_model
from validate import validate_model_cli
from predict import predict_image_cli, predict_folder_cli
from dataset_info import dataset_info_cli
from report import generate_report_cli


ensure_dirs()


def main_menu():
    menu = [
        "Train Model",
        "Validasi Model",
        "Prediksi 1 Gambar",
        "Prediksi Folder Gambar",
        "Informasi Dataset",
        "Informasi Performa Model (Laporan)",
        "Keluar",
    ]

    while True:
        print("\n=== Aksara Ulu Rejang (Kaganga) — YOLO11 Project ===")
        for i, m in enumerate(menu, 1):
            print(f"{i}. {m}")

        try:
            choice = int(input("Pilih menu (angka): "))
        except Exception:
            print("Input tidak valid. Coba lagi.")
            continue

        if choice == 1:
            train_model()
        elif choice == 2:
            validate_model_cli()
        elif choice == 3:
            predict_image_cli()
        elif choice == 4:
            predict_folder_cli()
        elif choice == 5:
            dataset_info_cli()
        elif choice == 6:
            generate_report_cli()
        elif choice == 7:
            print("Keluar. Terima kasih.")
            sys.exit(0)
        else:
            print("Pilihan tidak tersedia.")


if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print('\nProgram dihentikan. Sampai jumpa.')
        sys.exit(0)


