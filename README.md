# Sistem Deteksi Aksara Ulu Rejang (Kaganga) dengan YOLOv11

Sistem deteksi tulisan Aksara Ulu Rejang (Kaganga) menggunakan model deep learning YOLOv11 dengan dataset dari Roboflow yang mencakup 253 kelas huruf.

---

## 👥 Anggota Kelompok

| No. | Nama | NIM |
|-----|------|-----|
| 1 | Muhammad Ariqoh Firjatullah | G1A023033 |
| 2 | Muhammad Jaka | G1A023042 |
| 3 | Muhammad Dizi Valgiyos | G1A023072 |

---

## 📊 Performa Model

Hasil pelatihan model pada dataset Aksara Ulu Rejang:

- **mAP (Mean Average Precision)**: 98.9%
- **Precision**: 97.5%
- **Recall**: 99.1%
- **Jumlah Kelas**: 253 (a sampai yu)

### Augmentasi Data

Model dilatih dengan augmentasi berikut:

- Rotasi ±10°
- Brightness ±10%
- Blur ≤1.5 px
- Noise ≤1.96%

---

## 📁 Struktur Project

```
Aksara Ulu Rejang.v1i.yolov11/
├── app.py                          # Aplikasi utama
├── data.yaml                       # Konfigurasi dataset YOLO
├── data_resolved.yaml              # Konfigurasi dataset resolved
├── requirements.txt                # Dependensi Python
├── yolo11n.pt                      # Pre-trained model YOLOv11 Nano
├── README.md                       # File dokumentasi ini
├── README.roboflow.txt             # Informasi dataset dari Roboflow
├── README.dataset.txt              # Informasi metadata dataset
├── README.results.txt              # Hasil pelatihan
├── train/                          # Dataset training (images & labels)
│   ├── images/
│   └── labels/
├── valid/                          # Dataset validasi
│   ├── images/
│   └── labels/
├── test/                           # Dataset test
│   ├── images/
│   └── labels/
├── classifier_dataset/             # Dataset untuk klasifikasi (opsional)
│   ├── train/
│   ├── valid/
│   └── test/
├── models/                         # Folder output model
│   ├── training/                   # Hasil training
│   └── best.pt                     # Model terbaik
└── predictions/                    # Folder hasil prediksi
    └── single/                     # Hasil prediksi single image
    └── folder/                     # Hasil prediksi folder
```

---

## 🚀 Quick Start

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
python app.py
```

### 3. Menu Utama

```
# ==================================================
SISTEM DETEKSI AKSARA KAGANGA YOLOv11

1. Training Model
2. Validasi Model
3. Prediksi 1 Gambar
4. Prediksi Folder Gambar
5. Informasi Dataset
6. Informasi Classifier Dataset
7. Keluar
# ==================================================
```

---

## 📋 Panduan Penggunaan

### 1. **Training Model**

Pilih menu `1` untuk melatih model baru:

```
Masukkan jumlah epoch (default 10): 10
Masukkan image size (default 416): 416
Samakan augmentasi seperti Roboflow? (Y/n): Y
```

**Parameter:**
- **Epoch**: Jumlah iterasi training (default: 10)
- **Image Size**: Ukuran input gambar (default: 416, sesuai Roboflow)
- **Augmentation**: Gunakan augmentasi Roboflow (default: Y)

**Output:** Model terbaik akan disimpan di `models/best.pt`

---

### 2. **Validasi Model**

Pilih menu `2` untuk memvalidasi model pada dataset validasi.

**Output:** Metrik performa (Precision, Recall, mAP50, mAP50-95)

---

### 3. **Prediksi 1 Gambar**

Pilih menu `3` untuk mendeteksi tulisan Kaganga pada satu gambar:

```
Masukkan path/nama gambar: path/to/image.jpg
```

**Output:**
- Nama kelas yang terdeteksi
- Confidence score
- Gambar dengan bounding box disimpan di `predictions/single/`

---

### 4. **Prediksi Folder Gambar**

Pilih menu `4` untuk mendeteksi semua gambar dalam folder:

```
Masukkan path folder gambar: path/to/folder
```

**Output:** Semua hasil prediksi disimpan di `predictions/folder/`

---

### 5. **Informasi Dataset**

Pilih menu `5` untuk melihat:
- Jumlah kelas (253)
- Nama-nama kelas
- Jumlah gambar per split (train/valid/test)

---

### 6. **Informasi Classifier Dataset**

Pilih menu `6` untuk melihat statistik `classifier_dataset/`:
- Jumlah gambar per kelas
- Distribusi data
- Contoh nama kelas

---

## 📦 Requirements

Lihat `requirements.txt`:

```
ultralytics==8.3.45
torch
torchvision
PyYAML
opencv-python
numpy
```

Instal dengan:

```bash
pip install -r requirements.txt
```

---

## 🎯 Dataset

Dataset diperoleh dari **Roboflow**:

- **Nama**: Aksara Ulu Rejang - v1
- **Jumlah Gambar**: 1,669
- **Format**: YOLOv11
- **Ukuran Input**: 416x416 (Stretched)
- **Augmentasi**: 3 versi per gambar
- **URL**: https://universe.roboflow.com/muhammad-ariqoh-firjatullah/aksara-ulu-rejang-kyn9c

### Kelas (253 huruf)

a, ah, an, ang, ar, aw, ay, ba, bah, ban, bang, bar, baw, bay, be, bi, bo, bu, ca, cah, can, cang, car, caw, cay, ce, ci, co, cu, da, dah, dan, dang, dar, daw, day, de, di, do, du, e, ga, gah, gan, gang, gar, gaw, gay, ge, gi, go, gu, ha, hah, han, hang, har, haw, hay, he, hi, ho, hu, i, ja, jah, jan, jang, jar, jaw, jay, je, ji, jo, ju, ka, kah, kan, kang, kar, kaw, kay, ke, ki, ko, ku, la, lah, lan, lang, lar, law, lay, le, li, lo, lu, ma, mah, man, mang, mar, maw, may, mba, mbah, mban, mbang, mbar, mbaw, mbay, mbe, mbi, mbo, mbu, me, mi, mo, mu, na, nah, nan, nang, nar, naw, nay, nda, ndah, ndan, ndang, ndar, ndaw, nday, nde, ndi, ndo, ndu, ne, nga, ngah, ngan, ngang, ngar, ngaw, ngay, nge, ngga, nggah, nggan, nggang, nggar, nggaw, nggay, ngge, nggi, nggo, nggu, ngi, ngo, ngu, ni, nja, njah, njan, njang, njar, njaw, njay, nje, nji, njo, nju, no, nu, nya, nyah, nyan, nyang, nyar, nyaw, nyay, nye, nyi, nyo, nyu, o, pa, pah, pan, pang, par, paw, pay, pe, pi, po, pu, ra, rah, ran, rang, rar, raw, ray, re, ri, ro, ru, sa, sah, san, sang, sar, saw, say, se, si, so, su, ta, tah, tan, tang, tar, taw, tay, te, ti, to, tu, u, wa, wah, wan, wang, war, waw, way, we, wi, wo, wu, ya, yah, yan, yang, yar, yaw, yay, ye, yi, yo, yu

---

## 🔧 Teknologi

- **Framework**: Ultralytics YOLOv11
- **Bahasa**: Python 3.8+
- **Deep Learning**: PyTorch
- **Data Annotation**: YOLO format
- **Source Dataset**: Roboflow

---

## 📈 Hasil Training

Setelah pelatihan, model akan menghasilkan:

- **Metrik**:
  ```
  Precision   : 97.5%
  Recall      : 99.1%
  mAP50       : [nilai]
  mAP50-95    : 98.9%
  ```

- **File Model**: `models/best.pt`
- **Training Logs**: `models/training/`
- **Weights History**: `models/training/weights/`

---

## 🐛 Troubleshooting

### 1. GPU Out of Memory

Jika training gagal karena GPU memory penuh:

```bash
# Ubah batch size di app.py dari 16 ke 8 atau 4
batch=8  # atau 4
```

### 2. Data Path Not Found

Pastikan struktur folder sesuai dengan `data.yaml`:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

### 3. Model Tidak Terbaca

Pastikan `yolo11n.pt` ada di root folder project.

---

## 📄 License

Dataset dan project ini menggunakan lisensi **CC BY 4.0** (Roboflow).

---

## 🔗 Referensi

- Dokumentasi YOLOv11: https://docs.ultralytics.com
- Roboflow Universe: https://universe.roboflow.com
- GitHub Ultralytics: https://github.com/ultralytics/ultralytics

---

## 📞 Kontak

Untuk pertanyaan atau masalah, hubungi anggota tim:
- Muhammad Ariqoh Firjatullah (G1A023033)
- Muhammad Jaka (G1A023042)
- Muhammad Dizi Valgiyos (G1A023072)

---

**Last Updated**: Juni 2026
