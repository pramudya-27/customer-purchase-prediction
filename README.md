# Dokumentasi Ujian Praktik: Prediksi Pembelian Pelanggan

![Ilustrasi](image.png)

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem prediksi pembelian bagi **RetailTech Solutions**, sebuah platform *e-commerce* internasional. Sistem ini menggunakan model *machine learning* untuk memprediksi pelanggan mana yang paling mungkin melakukan pembelian berdasarkan perilaku penjelajahan (browsing) mereka. Hasil dari proyek ini diharapkan dapat membantu strategi pertumbuhan dan peningkatan pendapatan perusahaan.

## Kebutuhan Data
Proyek ini menggunakan beberapa kumpulan data, antara lain:
- `raw_customer_data.csv`: Data mentah sesi pelanggan yang memerlukan pembersihan.
- `model_data.csv`: Dataset yang digunakan untuk tahapan rekayasa fitur (feature engineering).
- `input_model_features.csv`: Fitur yang sudah dipersiapkan untuk melatih model.
- `validation_features.csv`: Data validasi untuk menguji prediksi dari model yang telah dilatih.

## Tahapan Proyek

### 1. Pembersihan Data (Data Cleaning)
Data mentah dari `raw_customer_data.csv` diproses untuk menangani nilai yang hilang (missing values) dengan kriteria berikut:
* **time_spent**: Diisi dengan nilai median.
* **pages_viewed**: Diisi dengan nilai rata-rata (mean).
* **basket_value**: Diisi dengan angka 0.
* **device_type**: Diisi dengan label "Unknown".
* **customer_type**: Diisi dengan label "New".
Hasil dari proses ini disimpan ke dalam DataFrame bernama `clean_data`.

### 2. Rekayasa Fitur (Feature Engineering)
Data hasil pembersihan awal (`model_data.csv`) dipersiapkan untuk dimasukkan ke dalam *neural network*:
* **Scaling**: Fitur numerik (`time_spent`, `pages_viewed`, `basket_value`) diubah skalanya menggunakan `MinMaxScaler` agar berada dalam rentang 0-1.
* **One-Hot Encoding**: Fitur kategorikal (`device_type`, `customer_type`) diubah menjadi format matriks biner menggunakan fungsi `pd.get_dummies`.
Hasil dari transformasi ini disimpan dalam DataFrame `model_feature_set`.

### 3. Pembuatan dan Pelatihan Model (Neural Network)
Sebuah *neural network* dibangun menggunakan library **PyTorch** dengan arsitektur berikut:
* **Hidden Layer**: Memiliki 1 lapisan tersembunyi dengan 8 unit dan fungsi aktivasi ReLU.
* **Output Layer**: Memiliki 1 unit keluaran dengan fungsi aktivasi Sigmoid untuk melakukan klasifikasi biner (0 atau 1).
* **Training**: Model (`purchase_model`) dilatih menggunakan data `input_model_features.csv` dengan *Binary Cross Entropy Loss* dan pengoptimal *Adam*.
* **Validation**: Model digunakan untuk memprediksi data dari `validation_features.csv`. Nilai probabilitas di atas atau sama dengan 0.5 diklasifikasikan sebagai 1 (pembelian), dan sisanya sebagai 0. Hasil akhir disimpan di dalam DataFrame `validation_predictions`.
