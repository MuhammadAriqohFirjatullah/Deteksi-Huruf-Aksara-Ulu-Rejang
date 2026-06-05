# Script untuk push project ke GitHub (Windows PowerShell)
# Repository: Deteksi-Huruf-Aksara-Ulu-Rejang
# Account: MuhammadAriqohFirjatullah

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "PUSH KE GITHUB: Deteksi-Huruf-Aksara-Ulu-Rejang" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Initialize git repository (jika belum ada)
Write-Host "[1/6] Initializing git repository..." -ForegroundColor Green
git init

# 2. Configure git (opsional, uncomment jika belum konfigurasi)
# git config user.email "your-email@example.com"
# git config user.name "Your Name"

# 3. Add all files
Write-Host "[2/6] Adding all files..." -ForegroundColor Green
git add .

# 4. Create initial commit
Write-Host "[3/6] Creating initial commit..." -ForegroundColor Green
git commit -m "Initial commit: Sistem deteksi Aksara Ulu Rejang dengan YOLOv11 (mAP 98.9%)"

# 5. Rename branch to main (jika belum)
Write-Host "[4/6] Renaming branch to main..." -ForegroundColor Green
git branch -M main

# 6. Add remote origin
Write-Host "[5/6] Adding remote origin..." -ForegroundColor Green
git remote add origin https://github.com/MuhammadAriqohFirjatullah/Deteksi-Huruf-Aksara-Ulu-Rejang.git

# 7. Push ke GitHub
Write-Host "[6/6] Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "✅ SELESAI! Project berhasil di-push ke GitHub" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host "Repository: https://github.com/MuhammadAriqohFirjatullah/Deteksi-Huruf-Aksara-Ulu-Rejang" -ForegroundColor Cyan
Write-Host ""
