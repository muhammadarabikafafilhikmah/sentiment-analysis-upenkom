# Dashboard Analisis Sentimen UPENKOM BKD Provinsi Jawa Tengah

Project Kerja Praktik (KP) — analisis sentimen tanggapan peserta Uji Penilaian Kompetensi ASN (UPENKOM) di BKD Provinsi Jawa Tengah, menggunakan pendekatan **hybrid**: Lexicon-based Analysis dan Machine Learning (Naive Bayes), disajikan dalam dashboard interaktif berbasis Streamlit.

## Deskripsi

Project ini bertujuan menganalisis sentimen peserta terhadap layanan penilaian kompetensi UPENKOM BKD Provinsi Jawa Tengah, berdasarkan tanggapan terbuka pada kuesioner peserta (studi kasus: PNS BKD Jawa Tengah), untuk membantu mengidentifikasi area layanan yang perlu ditingkatkan.

## Latar Belakang

Peserta uji kompetensi mengisi kuesioner setelah mengikuti proses penilaian, termasuk pertanyaan terbuka mengenai kesulitan yang dihadapi, harapan, serta saran/kritik. Karena jumlah tanggapan cukup banyak, dilakukan analisis sentimen otomatis untuk mengelompokkan tanggapan menjadi positif/negatif, sehingga penyelenggara dapat lebih cepat menangkap pola kepuasan peserta.

## Dataset

- **Sumber**: Kuesioner tanggapan terbuka peserta penilaian kompetensi UPENKOM periode Januari–Maret 2022 (198 respons, 42 kolom pada data mentah)
- **Kolom yang dianalisis**: Timestamp, instrumen yang dianggap sulit, harapan peserta, serta saran/kritik peserta

> **Catatan privasi**: Dataset mentah kuesioner berisi data pribadi responden (nama, instansi asal, dan jabatan) yang merupakan pegawai negeri sipil (PNS) di lingkungan Pemerintah Provinsi Jawa Tengah. **Dataset asli tidak disertakan dalam repository ini** untuk melindungi privasi responden. Yang disertakan hanya hasil analisis teragregasi (statistik, grafik, dan contoh teks yang telah diproses tanpa identitas). Jika Anda ingin mereproduksi analisis ini, siapkan dataset Anda sendiri dengan struktur kolom yang sama, atau hubungi penulis untuk versi data yang telah dianonimkan.

## Metodologi

Analisis mengikuti tahapan berikut (kerangka CRISP-DM):

1. **Business Understanding** — menentukan tujuan analisis sentimen terhadap layanan UPENKOM
2. **Data Understanding** — eksplorasi struktur dan kualitas data kuesioner
3. **Data Preparation** — preprocessing teks: case folding, cleaning (URL, tanda baca, angka), tokenizing, stopword removal
4. **Modeling** —
   - **Lexicon-based**: pelabelan sentimen menggunakan kamus sentimen bahasa Indonesia buatan sendiri (kata positif/negatif dengan bobot)
   - **Machine Learning**: TF-IDF/Count Vectorizer + Multinomial Naive Bayes, dengan resampling untuk menyeimbangkan kelas minoritas sebelum training
5. **Evaluation** — accuracy, confusion matrix, precision/recall/F1 per kelas, cross-validation, serta perbandingan tingkat kesesuaian (*agreement*) antara hasil Lexicon dan Machine Learning
6. **Analisis Mendalam** — kata kunci teratas per sentimen, dan tren sentimen berdasarkan waktu

## Hasil & Evaluasi Model

| Metrik | Nilai |
|---|---|
| Akurasi model (Naive Bayes) | 92.75% |
| Akurasi rata-rata cross-validation (5-fold) | 87.41% (± 3.60%) |
| Agreement Lexicon vs Machine Learning | 96.97% |

**Classification Report:**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Negatif | 0.9688 | 0.8857 | 0.9254 | 35 |
| Positif | 0.8919 | 0.9706 | 0.9296 | 34 |

**Distribusi sentimen (Machine Learning):** 173 positif (87.37%), 25 negatif (12.63%) dari 198 total tanggapan.

Visualisasi pendukung tersedia di folder `assets/`: confusion matrix, perbandingan hasil Lexicon vs ML, tren sentimen bulanan, dan word cloud kata kunci.

## Dashboard Aplikasi

Dashboard dibangun dengan Streamlit, menampilkan ringkasan hasil analisis (bukan inferensi real-time terhadap data baru), mencakup:

- Ringkasan statistik dataset
- Perbandingan hasil Lexicon-based vs Machine Learning
- Tren sentimen berdasarkan waktu
- Analisis kata kunci per sentimen
- Contoh proses preprocessing teks

### Tampilan Aplikasi
![Tampilan Dashboard](<img width="827" height="402" alt="image" src="https://github.com/user-attachments/assets/824ff349-1918-4631-8151-0bb425982b20" />
)

## Teknologi yang Digunakan

- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF/Count Vectorizer, Multinomial Naive Bayes, cross-validation)
- Matplotlib, Seaborn, Plotly
- WordCloud
- Streamlit (dashboard)

## Struktur Project

```text
sentiment-analysis-upenkom/
├── README.md
├── requirements.txt
├── notebooks/
│   └── Analisis_Sentiment.ipynb
├── app/
│   └── sentiment_app.py
└── assets/
    ├── confusion_matrix_sentimen.png
    ├── perbandingan_sentimen.png
    ├── tren_sentimen.png
    └── wordcloud_sentimen.png
```

## Cara Menjalankan

1. Clone repository ini
2. Install dependency: `pip install -r requirements.txt`
3. Jalankan dashboard: `streamlit run app/sentiment_app.py`
4. Buka `http://localhost:8501` di browser

Untuk menjalankan ulang analisis dari awal, buka `notebooks/Analisis_Sentiment.ipynb` (memerlukan dataset kuesioner Anda sendiri, lihat bagian Dataset).

## Batasan

- Kamus sentimen (lexicon) dibangun manual dan bersifat sederhana, belum tervalidasi secara linguistik formal
- Dashboard menampilkan hasil analisis yang telah dihitung sebelumnya (pre-computed), bukan inferensi model secara real-time
- Ukuran dataset relatif kecil (198 respons) sehingga hasil evaluasi model perlu diinterpretasikan dengan hati-hati

## Penulis

Muhammad Arabi Kafafil Hikmah — Program Studi Sains Data, Universitas Teknologi Yogyakarta.
Project ini dikerjakan sebagai bagian dari mata kuliah Kerja Praktik.

## Catatan Privasi & Etika Data

Dataset asli yang digunakan dalam Kerja Praktik ini merupakan data internal instansi pemerintah yang bersifat rahasia/terbatas. Repository ini hanya membagikan kode, metodologi, dan hasil analisis teragregasi untuk keperluan portofolio akademik — bukan data mentah responden.
