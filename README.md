# Analisis Sentimen Menggunakan SVM (Support Vector Machine)

## Gambaran Umum

Proyek ini mengimplementasikan model analisis sentimen menggunakan **Support Vector Machine (SVM)** untuk mengklasifikasikan ulasan film menjadi dua kelas:  
- **Positif (disukai)**  
- **Negatif (tidak disukai)**

Model dilatih menggunakan dataset berlabel `Training.txt` yang berisi kalimat-kalimat dengan label `1` (disukai) dan `0` (tidak disukai).

## Metode

### Preprocessing Data

Data teks diproses melalui langkah-langkah berikut:

- Mengubah semua teks menjadi huruf kecil (lowercase).
- Melakukan tokenisasi kata menggunakan ekspresi reguler sederhana untuk menangkap token berupa huruf.
- Melakukan **lemmatization** pada setiap token menggunakan `WordNetLemmatizer` dari NLTK untuk mengubah kata ke bentuk dasarnya (misalnya, "berlari" menjadi "lari").

### Ekstraksi Fitur

Menggunakan **TF-IDF Vectorizer** untuk mengubah teks yang sudah diproses menjadi representasi numerik berdasarkan bobot pentingnya kata dalam dokumen relatif terhadap seluruh korpus.

### Pelatihan Model

- Model yang digunakan adalah **SVM dengan kernel linear** dari pustaka scikit-learn (`sklearn.svm.SVC`).
- Dataset dibagi menjadi data latih (80%) dan data uji (20%).
- Model dilatih menggunakan data latih dan dievaluasi performanya menggunakan data uji.

## Evaluasi

Model dievaluasi menggunakan:

- **Classification report**: mengukur precision, recall, dan f1-score.
- **Confusion matrix** yang divisualisasikan menggunakan heatmap dari seaborn.

## Contoh Prediksi

Model diuji pada kalimat contoh seperti:

- "the da vinci code is awesome"  
- "the movie was boring and bad"

Untuk melihat hasil prediksi sentimen dari kalimat tersebut.

## Hasil

- Model menunjukkan akurasi yang baik dengan performa seimbang antara precision dan recall pada data uji.
- Confusion matrix memperlihatkan distribusi prediksi yang benar dan salah secara visual.

## Kesimpulan

Penggunaan **SVM dengan TF-IDF** dan preprocessing yang baik seperti lemmatization efektif dalam melakukan analisis sentimen pada data teks.  
Lemmatization membantu mengurangi variasi kata sehingga model lebih mudah mengenali pola dan konteks dalam teks.

## Cara Menjalankan Kode

1. Pastikan Python 3 dan paket-paket berikut sudah terpasang :

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn

2. Unduh data wordnet NLTK dengan menjalankan Python dan memasukkan:
import nltk
nltk.download('wordnet')

3. Jalankan skrip:
python svm.py
