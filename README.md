<h1 align="center">Tugas Besar 1 IF3270 Pembelajaran Mesin</h1>
<h1 align="center">Kelompok 48</h3>
<h3 align="center">FEEDFORWARD NEURAL NETWORK (FFNN)</p>

## Table of Contents

- [Abstraksi](#abstraksi)
- [Cara set up dan run](#cara-set-up-dan-run)
- [Pembagian Tugas](#pembagian-tugas)

## Abstraksi
Repository ini berisi kode yang digunakan untuk melakukan simulasi FFNN. Di dalam repository ini terdapat kode yang dibangun dari _scratch_ untuk mensimulasikan jaringan Neural Network. Terdapat beberapa metode inisialisasi, fungsi aktivasi yang juga diimplementasikan secara manual. Inti dari FFNN, terdapat 2 aksi yang terjadi pada FFNN, yaitu forward dan backward. Untuk eksperimen kali ini, dataset yang digunakan adalah `mnist`. Selain, pembangunan model secara polos, terdapat juga penambahan fitur seperti kesempatan model untuk melakukan normalisasi menggunakan RMSNorm dan regularisasi L1 dan L2. Terdapat juga beberapa jenis inisialisasi bobot yang dapat digunakan. Inti dari repository ini adalah untuk memenuhi spesifikasi Tugas Besar 1 IF3270 Pembelajaran Mesin terkait dengan FFNN.

## Cara set up dan run
1. Pastikan Python terinstal pada komputer yang akan menjalankan program di repository ini. Download atau instalasi Python dapat dilihat [disini](https://www.python.org/downloads/).
2. Kemudian clone repository ini dengan cara seperti berikut.
```
git clone https://github.com/DrwnRstnly/48-IF3270-FFNN.git
cd 48-IF3270-FFNN
```
3. Lalu, gunakan code editor kesukaanmu yang dapat support notebook (.ipynb). Contohnya, `vscode`.
4. Navigasi workspace pada file `main.ipynb` dan jalankan menggunakan fitur `Run All` yang tersedia atau melakukan command `Ctrl` + `Enter` pada setiap bloknya.
5. Seharusnya, program notebook sudah berjalan. **Ingat untuk menjalankan notebook mulai dari blok yang teratas, untuk mencegah error pada blok dibawahnya.**

**Note: Ingat untuk melakukan pip install terhadap library yang tidak tersedia pada environment Anda, dengan menggunakan command `pip install <nama_modul>`**

## Pembagian Tugas
This project was developed by:
| NIM      | Nama                    | Kontribusi                                                                                                                                                                                                               |
|----------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 13522045 | Elbert Chailes           | Pembuatan model FFNN, inisialisasi bobot, regularisasi, normalisasi, laporan                                                          |
| 13522047 | Farel Winalda    | Pembuatan model FFNN, inisialisasi bobot, regularisasi, normalisasi, laporan                                                          |
| 13522115 | Derwin Rustanly    | Pembuatan model FFNN, inisialisasi bobot, regularisasi, normalisasi, laporan                                                          |
