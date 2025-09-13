# 📘 Reka Cipta Sistem Penerjemah Bahasa Isyarat Indonesia (BISINDO) ke Teks

Proyek ini merupakan bagian dari **Tugas Akhir** dengan fokus pada penerjemahan gerakan **Bahasa Isyarat Indonesia (BISINDO)** menjadi teks menggunakan **deep learning**.  
Tahap awal penelitian ini dilakukan secara bertahap (preliminary) untuk memahami pipeline data, pemrosesan landmark dengan **MediaPipe**, serta eksplorasi model berbasis **RNN (LSTM & GRU)**.

---

## 🎯 Tujuan Utama

- Membuat sistem yang mampu menangkap gerakan isyarat melalui kamera.
- Mengekstrak landmark wajah, tangan, dan pose menggunakan **MediaPipe Holistic**.
- Melatih model berbasis **RNN** untuk menerjemahkan gestur menjadi teks.

---

## 🧩 Progres Preliminary

### 1️⃣ Tahap Pertama: NumPy

- Landmark dari MediaPipe disimpan dalam bentuk **array NumPy**.
- Setiap sequence (gerakan isyarat) direpresentasikan sebagai matriks berisi koordinat landmark.
- Tujuan utama: **validasi pipeline data** sebelum menyimpan ke format lain.

### 2️⃣ Tahap Kedua: JSON

- Data sequence disimpan dalam format **JSON** agar lebih mudah dibaca dan diinspeksi.
- Struktur JSON mencakup:
  - Metadata (id video, fps, jumlah frame, jumlah landmark).
  - Frame-by-frame landmark (pose, face, tangan kiri, tangan kanan).
- Tujuan utama: **mempersiapkan dataset standar** untuk pelatihan model.

---

## 🔮 Rencana Selanjutnya

- **Model Awal:** LSTM digunakan sebagai baseline untuk memproses sequence gesture → teks.
- **Model Lanjutan:** Mengeksplorasi **GRU** untuk membandingkan performa dan efisiensi.
- Evaluasi dilakukan berdasarkan **akurasi penerjemahan** serta **kecepatan inferensi**.

---

## 📂 Struktur Proyek (sementara)

lstm
┣ asl_dataset
┃ ┣ iloveyou.mp4
┃ ┣ nicetomeetyou.mp4
┃ ┗ thankyou.mp4
┣ data
┃ ┣ halo
┃ ┃ ┣ sequence_1.json
┃ ┃ ┗ sequence_1.npy
┃ ┣ terima_kasih
┃ ┃ ┣ sequence_1.json
┃ ┃ ┗ sequence_1.npy
┃ ┣ halo_combined.json
┃ ┣ halo_combined.npy
┃ ┣ terima_kasih_combined.json
┃ ┗ terima_kasih_combined.npy
┣ MP_DatawithVids
┃ ┣ halo
┃ ┃ ┣ halo_sequence_1.mp4
┃ ┃ ┣ halo_sequence_2.mp4
┃ ┃ ┣ sequence_1.npy
┃ ┃ ┗ sequence_2.npy
┃ ┗ terima_kasih
┣ .python-version
┣ coba.py
┣ dataColnp.py
┣ data_collecting.py
┣ data_collection_w_vid.py
┣ data_labelling.py
┣ lstm.ipynb
┣ modelling.py
┣ MP_Data.zip
┗ README.md

---

## ✨ Catatan

- Dataset saat ini masih dalam tahap awal (gesture sederhana seperti _halo_, _terima kasih_).
- Format penyimpanan akan terus dieksplorasi hingga didapat format optimal untuk pelatihan model.
- Dokumentasi ini akan terus diperbarui seiring perkembangan proyek.
