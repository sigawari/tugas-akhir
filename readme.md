# 📘 Reka Cipta Sistem Penerjemah Bahasa Isyarat Indonesia (BISINDO) ke Teks

Proyek ini merupakan bagian dari **Tugas Akhir** dengan fokus pada penerjemahan gerakan **Bahasa Isyarat Indonesia (BISINDO)** menjadi teks menggunakan **deep learning**.  
Tahap awal penelitian ini dilakukan secara bertahap (preliminary) untuk memahami pipeline data, pemrosesan landmark dengan **MediaPipe**, serta eksplorasi model berbasis **RNN (LSTM & GRU)**.

---

## 🎯 Research Objectives

- Investigate the feasibility of **ResNet-2D as a spatial-only model** for BISINDO recognition.
- Design a **multichannel landmark representation** that embeds motion information without explicit temporal modeling.
- Analyze the impact of different **landmark combinations** (pose, hands, face) on recognition performance.
- Provide empirical evidence that **2D CNNs can learn temporal patterns implicitly** through structured spatial inputs.

---

## 🧩 Progres Preliminary

### 1️⃣ Tahap Pertama: NumPy

- Landmark dari MediaPipe disimpan dalam bentuk **array NumPy**.
- Setiap sequence (gerakan isyarat) direpresentasikan sebagai matriks berisi koordinat landmark.
- Tujuan utama: **validasi pipeline data** sebelum menyimpan ke format lain.

### 2️⃣ Tahap Kedua: JSON

![Phase of Data Collecting](json_collect.png)

- Data sequence disimpan dalam format **JSON** agar lebih mudah dibaca dan diinspeksi.
- Struktur JSON mencakup:
  - Metadata (id video, fps, jumlah frame, jumlah landmark).
  - Frame-by-frame landmark (pose, face, tangan kiri, tangan kanan).
- Proses pengumpulan data dilakukan dengan menyimpan sequence gerakan dalam format JSON.
- Setiap file JSON mewakili satu video gesture dengan struktur yang telah ditentukan.
- Tujuan utama: **mempersiapkan dataset standar** untuk pelatihan model serta mengumpulkan dataset yang cukup untuk eksplorasi model.
- File JSON yang dihasilkan dapat digunakan langsung untuk eksplorasi model.

---

## 🔮 Rencana Selanjutnya

- **Model Awal:** LSTM digunakan sebagai baseline untuk memproses sequence gesture → teks.
- **Model Lanjutan:** Mengeksplorasi **GRU** untuk membandingkan performa dan efisiensi.
- Evaluasi dilakukan berdasarkan **akurasi penerjemahan** serta **kecepatan inferensi**.

---

## ✨ Catatan

- Dataset saat ini masih dalam tahap awal (gesture sederhana seperti _halo_, _terima kasih_).
- Format penyimpanan akan terus dieksplorasi hingga didapat format optimal untuk pelatihan model.
- Dokumentasi ini akan terus diperbarui seiring perkembangan proyek.
