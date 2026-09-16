# NoiseClean - Filtering / Noise Reduction

Aplikasi interaktif untuk demonstrasi pengurangan noise pada sinyal audio. Aplikasi dapat membuat sinyal uji atau membaca file WAV, menambahkan white Gaussian noise, lalu membandingkan hasil tiga metode filter:

- Wiener Filter (metode utama)
- Chebyshev Type I Low-pass Filter (metode utama)
- Moving Average dan Median Filter (metode pembanding)

## Menjalankan aplikasi

1. Pastikan Python 3.10+ terpasang.
2. Dari folder proyek, instal dependensi:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Jalankan:

   ```powershell
   python -m streamlit run app.py
   ```

4. Browser akan membuka halaman aplikasi. Gunakan panel kiri untuk memilih sumber sinyal, besar noise, dan parameter filter.

## Catatan

- File input harus berformat WAV. Jika stereo, kedua kanal dirata-ratakan menjadi mono agar analisis konsisten.
- Untuk memperlihatkan metrik SNR secara objektif, aplikasi memakai sinyal awal sebagai referensi sebelum noise ditambahkan.
- FFT hanya digunakan untuk menampilkan spektrum frekuensi; bukan algoritma filtering utama pada aplikasi ini.
- Sinyal hasil dapat diputar dan diunduh sebagai WAV.
