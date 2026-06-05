#!/bin/bash
# Script untuk push project ke GitHub
# Repository: Deteksi-Huruf-Aksara-Ulu-Rejang
# Account: MuhammadAriqohFirjatullah

echo "======================================================"
echo "PUSH KE GITHUB: Deteksi-Huruf-Aksara-Ulu-Rejang"
echo "======================================================"
echo ""

# 1. Initialize git repository (jika belum ada)
echo "[1/6] Initializing git repository..."
git init

# 2. Configure git (opsional, uncomment jika belum konfigurasi)
# git config user.email "your-email@example.com"
# git config user.name "Your Name"

# 3. Add all files
echo "[2/6] Adding all files..."
git add .

# 4. Create initial commit
echo "[3/6] Creating initial commit..."
git commit -m "Initial commit: Sistem deteksi Aksara Ulu Rejang dengan YOLOv11 (mAP 98.9%)"

# 5. Rename branch to main (jika belum)
echo "[4/6] Renaming branch to main..."
git branch -M main

# 6. Add remote origin
echo "[5/6] Adding remote origin..."
git remote add origin https://github.com/MuhammadAriqohFirjatullah/Deteksi-Huruf-Aksara-Ulu-Rejang.git

# 7. Push ke GitHub
echo "[6/6] Pushing to GitHub..."
git push -u origin main

echo ""
echo "======================================================"
echo "✅ SELESAI! Project berhasil di-push ke GitHub"
echo "======================================================"
echo "Repository: https://github.com/MuhammadAriqohFirjatullah/Deteksi-Huruf-Aksara-Ulu-Rejang"
echo ""
