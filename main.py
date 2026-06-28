# =============================================================================
# SISTEM KLASIFIKASI KELAYAKAN PENERIMA BANTUAN
# Perbandingan Algoritma Naive Bayes dan Decision Tree C4.5
# Dataset: Data Penerima Bantuan DPPKB Kabupaten Gorontalo
# =============================================================================

# --- Import Library yang Dibutuhkan ---
import pandas as pd                        # Untuk manipulasi dan analisis data
import numpy as np                         # Untuk komputasi numerik
import io                                  # Untuk membaca data dari string
import re                                  # Untuk ekspresi reguler (regex)
import joblib                               # Untuk menyimpan model ke file .pkl
from pathlib import Path                    # Untuk path file yang aman lintas OS
import warnings                            # Untuk mengelola pesan peringatan
warnings.filterwarnings('ignore')          # Sembunyikan peringatan agar output lebih bersih

from sklearn.model_selection import (
    train_test_split,                      # Untuk membagi data train dan test
    cross_val_score,                       # Untuk evaluasi model dengan cross-validation
    StratifiedKFold                        # Untuk cross-validation dengan distribusi kelas yang seimbang
)
from sklearn.pipeline import make_pipeline  # Pipeline agar scaler CV tidak mengubah scaler produksi
from sklearn.naive_bayes import GaussianNB # Algoritma Naive Bayes untuk fitur numerik
from sklearn.tree import (
    DecisionTreeClassifier,                # Algoritma Decision Tree (C4.5 menggunakan criterion='entropy')
    export_text                            # Untuk menampilkan struktur pohon keputusan
)
from sklearn.preprocessing import (
    LabelEncoder,                          # Untuk mengubah kategori teks menjadi angka
    StandardScaler                         # Untuk normalisasi fitur numerik (khusus Naive Bayes)
)
from sklearn.metrics import (
    accuracy_score,                        # Akurasi prediksi
    classification_report,                 # Laporan presisi, recall, F1-score
    confusion_matrix                       # Matriks konfusi untuk evaluasi detail
)


# =============================================================================
# BAGIAN 1: DATA MENTAH
# Dataset penerima bantuan dalam format CSV
# =============================================================================

DATA_MENTAH = pd.read_csv("datasets/baznas-2.csv")

# =============================================================================
# BAGIAN 2: FUNGSI PEMBERSIHAN DATA
# Fungsi-fungsi untuk memproses dan membersihkan data mentah
# =============================================================================

def bersihkan_angka(nilai):
    """
    Mengubah nilai pendapatan dari format string Indonesia menjadi float.
    Contoh: '1.500.000' -> 1500000.0
             '0' -> 0.0
             '' atau NaN -> 0.0
    """
    if pd.isna(nilai) or str(nilai).strip() == '':
        return 0.0                              # Nilai kosong dianggap 0
    nilai_str = str(nilai).strip()
    # Hapus titik pemisah ribuan pada format angka Indonesia
    nilai_str = nilai_str.replace('.', '').replace(',', '.')
    try:
        return float(nilai_str)
    except:
        return 0.0                              # Jika gagal parsing, kembalikan 0


def normalisasi_teks(nilai):
    """
    Normalisasi teks: ubah ke huruf kecil, hapus spasi berlebih.
    Fungsi ini penting untuk menghomogenkan variasi penulisan dalam dataset.
    """
    if pd.isna(nilai):
        return ''
    return str(nilai).strip().lower()           # Potong spasi + ubah ke lowercase


def normalisasi_dinding(nilai):
    """
    Menyeragamkan variasi penulisan jenis dinding menjadi 5 kategori standar:
    - tembok    : material permanen berbasis bata/beton
    - semi      : material setengah permanen
    - bilik     : material bambu atau kayu ringan
    - bambu     : bambu murni
    - lainnya   : selain kategori di atas
    """
    v = normalisasi_teks(nilai)
    if any(x in v for x in ['tembok', 'beton']):
        return 'tembok'
    elif 'semi' in v:
        return 'semi'
    elif any(x in v for x in ['bilik', 'kayu']):
        return 'bilik'
    elif 'bambu' in v:
        return 'bambu'
    else:
        return 'lainnya'


def normalisasi_lantai(nilai):
    """
    Menyeragamkan variasi penulisan jenis lantai menjadi 4 kategori:
    - keramik   : lantai ubin/keramik (kualitas lebih baik)
    - semen     : lantai semen/plester
    - tanah     : lantai tanah (kualitas paling rendah)
    - panggung  : lantai rumah panggung (kayu)
    - lainnya   : selain kategori di atas
    """
    v = normalisasi_teks(nilai)
    if 'keramik' in v or 'kermaik' in v:       # Tangani typo 'kermaik'
        return 'keramik'
    elif 'semen' in v:
        return 'semen'
    elif 'tanah' in v:
        return 'tanah'
    elif 'panggung' in v:
        return 'panggung'
    else:
        return 'lainnya'


def normalisasi_luas(nilai):
    """
    Menyeragamkan variasi penulisan luas tempat tinggal menjadi 4 kategori:
    - besar         : rumah besar
    - sedang        : rumah sedang
    - kecil         : rumah kecil
    - sangat kecil  : rumah sangat kecil
    """
    v = normalisasi_teks(nilai)
    if 'besar' in v:
        return 'besar'
    elif 'sedang' in v:
        return 'sedang'
    elif 'sangat kecil' in v:
        return 'sangat kecil'
    elif 'kecil' in v:
        return 'kecil'
    else:
        return 'sedang'                         # Default ke 'sedang' bila tidak dikenali


def normalisasi_pendidikan(nilai):
    """
    Menyeragamkan dan mengelompokkan tingkat pendidikan ke 6 kategori:
    - tidak sekolah     : belum/tidak sekolah
    - sd                : tamat atau tidak tamat SD
    - sltp              : setingkat SMP
    - slta              : setingkat SMA
    - diploma           : D1-D3
    - sarjana           : D4/S1 ke atas
    """
    v = normalisasi_teks(nilai)
    if any(x in v for x in ['belum tamat', 'belum tamat sd', 'tidak sekolah']):
        return 'tidak sekolah'
    elif 'tamat sd' in v or (v == 'sd'):
        return 'sd'
    elif any(x in v for x in ['sltp', 'smp', 'sederajat']):
        # Bedakan SLTP dan SLTA yang sama-sama mengandung 'sederajat'
        if 'slta' not in v and 'sma' not in v:
            return 'sltp'
        else:
            return 'slta'
    elif any(x in v for x in ['slta', 'sma', 'pelajar', 'mahasiswa']):
        return 'slta'
    elif any(x in v for x in ['diploma i', 'diploma ii', 'diploma iii', 'd1', 'd2', 'd3']):
        return 'diploma'
    elif any(x in v for x in ['diploma iv', 'strata', 's1', 's2', 's3']):
        return 'sarjana'
    else:
        return 'slta'                           # Default ke SLTA bila tidak dikenali


def normalisasi_pekerjaan(nilai):
    """
    Mengelompokkan jenis usaha/pekerjaan ke 8 kategori umum:
    - perdagangan   : pedagang/wiraswasta
    - pertanian     : petani/nelayan
    - jasa          : tukang, transportasi, buruh
    - karyawan      : karyawan swasta
    - pns           : Pegawai Negeri Sipil
    - irt           : Ibu Rumah Tangga / URT
    - tidak bekerja : tidak ada pekerjaan
    - lainnya       : selain kategori di atas
    """
    v = normalisasi_teks(nilai)
    if any(x in v for x in ['dagang', 'wiraswasta', 'usaha']):
        return 'perdagangan'
    elif any(x in v for x in ['petani', 'nelayan', 'tani']):
        return 'pertanian'
    elif any(x in v for x in ['tukang', 'transportasi', 'buruh', 'bekerja']):
        return 'jasa'
    elif 'karyawan' in v:
        return 'karyawan'
    elif any(x in v for x in ['pns', 'pegawai negeri']):
        return 'pns'
    elif any(x in v for x in ['irt', 'urt', 'ibu rumah']):
        return 'irt'
    elif v in ['', '-']:
        return 'tidak bekerja'
    else:
        return 'lainnya'


def normalisasi_aset(nilai):
    """
    Menyeragamkan kepemilikan aset ke nilai biner:
    - 0 : tidak ada aset
    - 1 : memiliki aset
    """
    v = normalisasi_teks(nilai)
    return 0 if 'tidak ada' in v or v == '' else 1


def hitung_jumlah_anggota(jumlah_tanggungan):
    """
    Jumlah tanggungan pada data mewakili anggota keluarga yang ditanggung.
    Kepala keluarga tetap dihitung agar indikator per kapita lebih stabil.
    """
    return max(1, int(jumlah_tanggungan)) + 1


def skor_kondisi_rumah(luas_tempat, jenis_dinding, jenis_lantai):
    """
    Skor 0-9 untuk menggambarkan kemampuan ekonomi dari kondisi tempat tinggal.
    Nilai besar berarti kondisi rumah makin mapan.
    """
    skor_luas = {
        'sangat kecil': 0,
        'kecil': 1,
        'sedang': 2,
        'besar': 3,
    }.get(luas_tempat, 2)
    skor_dinding = {
        'bambu': 0,
        'bilik': 1,
        'semi': 2,
        'tembok': 3,
        'lainnya': 1,
    }.get(jenis_dinding, 1)
    skor_lantai = {
        'tanah': 0,
        'panggung': 1,
        'semen': 2,
        'keramik': 3,
        'lainnya': 1,
    }.get(jenis_lantai, 1)
    return skor_luas + skor_dinding + skor_lantai


def hitung_skor_kemampuan_ekonomi(penghasilan_total, jumlah_tanggungan,
                                  kepemilikan_aset, jenis_lantai,
                                  jenis_dinding, luas_tempat):
    """
    Menggabungkan sinyal pendapatan, aset, tanggungan, dan rumah.
    Skor ini dipakai sebagai fitur agar model melihat kemampuan ekonomi
    secara lebih representatif daripada pola kategorikal mentah saja.
    """
    kebutuhan = hitung_kebutuhan_dasar(
        jumlah_tanggungan, jenis_lantai, jenis_dinding, luas_tempat
    )
    anggota = hitung_jumlah_anggota(jumlah_tanggungan)
    pendapatan_per_anggota = penghasilan_total / anggota if anggota > 0 else 0
    rasio = penghasilan_total / kebutuhan if kebutuhan > 0 else 0
    skor_rumah = skor_kondisi_rumah(luas_tempat, jenis_dinding, jenis_lantai)

    skor = 0
    if rasio >= 1.25:
        skor += 2
    elif rasio >= 1.0:
        skor += 1

    if pendapatan_per_anggota >= 1_000_000:
        skor += 2
    elif pendapatan_per_anggota >= 750_000:
        skor += 1

    if int(kepemilikan_aset) == 1:
        skor += 1
    if jumlah_tanggungan <= 2 and penghasilan_total >= 2_000_000:
        skor += 1
    if skor_rumah >= 8:
        skor += 2
    elif skor_rumah >= 6:
        skor += 1

    return skor


# =============================================================================
# BAGIAN 3: FUNGSI LABELING (PENENTUAN STATUS FAKIR/MISKIN)
# Berdasarkan kriteria yang diberikan dalam soal
# =============================================================================

def hitung_kebutuhan_dasar(jumlah_tanggungan, jenis_lantai, jenis_dinding, luas_tempat):
    """
    Mengestimasi kebutuhan dasar bulanan dalam rupiah berdasarkan:
    - Jumlah tanggungan keluarga
    - Kualitas tempat tinggal (lantai, dinding, luas)
    
    Asumsi dasar:
    - Per orang: Rp 500.000/bulan (kebutuhan pokok minimum)
    - Biaya tambahan berdasarkan kualitas tempat tinggal
    
    Returns:
        float: estimasi kebutuhan dasar per bulan (Rupiah)
    """
    # Kebutuhan dasar per anggota keluarga (termasuk kepala keluarga)
    anggota = hitung_jumlah_anggota(jumlah_tanggungan)
    kebutuhan_per_orang = 500_000               # Rp 500.000/orang/bulan

    # Faktor biaya tempat tinggal berdasarkan luas (sewa/perawatan estimasi)
    biaya_tempat = {
        'sangat kecil': 100_000,
        'kecil': 150_000,
        'sedang': 250_000,
        'besar': 400_000
    }.get(luas_tempat, 200_000)                 # Default sedang jika tidak dikenali

    # Faktor kondisi dinding (dinding buruk = biaya perbaikan lebih besar)
    faktor_dinding = {
        'tembok': 0,                            # Sudah permanen, tidak perlu biaya tambahan
        'semi': 50_000,                         # Perlu perbaikan periodik
        'bilik': 100_000,                       # Rentan rusak, butuh biaya lebih
        'bambu': 150_000,                       # Paling rentan
        'lainnya': 75_000
    }.get(jenis_dinding, 75_000)

    # Total estimasi kebutuhan dasar
    total = (anggota * kebutuhan_per_orang) + biaya_tempat + faktor_dinding
    return total


def tentukan_status(penghasilan_total, jumlah_tanggungan, pendidikan,
                    jenis_lantai, jenis_dinding, luas_tempat,
                    kepemilikan_aset):
    """
    Menentukan status kesejahteraan berdasarkan perbandingan
    penghasilan dengan kebutuhan dasar.
    
    Kriteria:
    - FAKIR  : penghasilan < 50% kebutuhan dasar
                (penghasilan sangat tidak mencukupi kebutuhan minimum)
    - MISKIN : 50% <= penghasilan < 100% kebutuhan dasar
                (ada penghasilan tapi masih kurang)
    - MAMPU  : penghasilan >= 100% kebutuhan dasar
                (penghasilan mencukupi kebutuhan dasar)

    Koreksi kemampuan ekonomi:
    - Pendapatan cukup, tanggungan rendah, memiliki aset, dan rumah mapan
      tidak dilabeli Fakir/Miskin walaupun pola historis dataset mengarah ke sana.
    
    Returns:
        str: 'Fakir', 'Miskin', atau 'Mampu'
    """
    kebutuhan = hitung_kebutuhan_dasar(
        jumlah_tanggungan, jenis_lantai, jenis_dinding, luas_tempat
    )

    rasio = penghasilan_total / kebutuhan if kebutuhan > 0 else 0
    anggota = hitung_jumlah_anggota(jumlah_tanggungan)
    pendapatan_per_anggota = penghasilan_total / anggota if anggota > 0 else 0
    skor_rumah = skor_kondisi_rumah(luas_tempat, jenis_dinding, jenis_lantai)
    skor_kemampuan = hitung_skor_kemampuan_ekonomi(
        penghasilan_total,
        jumlah_tanggungan,
        kepemilikan_aset,
        jenis_lantai,
        jenis_dinding,
        luas_tempat,
    )

    indikator_mampu_kuat = (
        penghasilan_total >= 2_500_000
        and jumlah_tanggungan <= 2
        and int(kepemilikan_aset) == 1
        and skor_rumah >= 8
    )
    indikator_mampu_umum = (
        skor_kemampuan >= 5
        or (rasio >= 1.15 and int(kepemilikan_aset) == 1 and skor_rumah >= 6)
        or (pendapatan_per_anggota >= 1_000_000 and skor_rumah >= 7)
    )

    if indikator_mampu_kuat or indikator_mampu_umum:
        return 'Mampu'

    if rasio < 0.5:
        # Penghasilan kurang dari setengah kebutuhan dasar -> FAKIR
        return 'Fakir'
    elif rasio < 1.0:
        # Penghasilan ada tapi tidak cukup memenuhi kebutuhan -> MISKIN
        return 'Miskin'
    else:
        # Penghasilan cukup atau lebih -> MAMPU (tidak layak dapat bantuan)
        return 'Mampu'


def tentukan_kelayakan(status):
    """
    Menentukan kelayakan penerima bantuan berdasarkan status kesejahteraan.

    Kriteria:
    - LAYAK     : status 'Fakir' atau 'Miskin' (membutuhkan bantuan)
    - TIDAK LAYAK: status 'Mampu' (tidak membutuhkan bantuan)

    Args:
        status (str): hasil dari tentukan_status(), yaitu 'Fakir', 'Miskin', atau 'Mampu'

    Returns:
        str: 'Layak' atau 'Tidak Layak'
    """
    if status in ('Fakir', 'Miskin'):
        return 'Layak'
    else:
        return 'Tidak Layak'


# =============================================================================
# BAGIAN 4: LOAD DAN PRA-PEMROSESAN DATA
# =============================================================================

def load_dan_proses_data():
    """
    Membaca data mentah, membersihkan, mentransformasi, dan membuat label kelas.
    
    Returns:
        pd.DataFrame: DataFrame yang sudah bersih dan siap untuk pemodelan
    """
    print("=" * 60)
    print("MEMUAT DAN MEMPROSES DATA")
    print("=" * 60)

    # ---- 4.1 Baca data dari string CSV ----
    df = pd.read_csv('datasets/baznas-3.csv', dtype=str)  # Baca semua kolom sebagai string
    print(f"Data mentah dimuat: {len(df)} baris, {len(df.columns)} kolom")

    # Hapus baris yang seluruhnya kosong (baris header/footer kosong)
    df = df.dropna(how='all')
    print(f"Setelah hapus baris kosong: {len(df)} baris")

    # Hapus baris header yang tersisip di tengah file CSV.
    sebelum_header_ganda = len(df)
    df = df[
        df['NAMA'].fillna('').str.strip().str.upper().ne('NAMA')
        & df['SUAMI'].fillna('').str.strip().str.upper().ne('SUAMI')
    ].copy()
    if len(df) != sebelum_header_ganda:
        print(f"Setelah hapus header ganda: {len(df)} baris")

    # ---- 4.2 Bersihkan kolom nama (hapus spasi berlebih) ----
    df['NAMA'] = df['NAMA'].str.strip()

    # ---- 4.3 Konversi kolom pendapatan ke numerik ----
    # Kolom SUAMI dan ISTRI berisi pendapatan bulanan masing-masing
    df['SUAMI'] = df['SUAMI'].apply(bersihkan_angka)
    df['ISTRI'] = df['ISTRI'].apply(bersihkan_angka)
    df['TOTAL_PENDAPATAN'] = df['SUAMI'] + df['ISTRI']  # Total pendapatan keluarga

    # ---- 4.4 Normalisasi kolom kategorikal ----
    df['JENIS DINDING']          = df['JENIS DINDING'].apply(normalisasi_dinding)
    df['JENIS LANTAI']           = df['JENIS LANTAI'].apply(normalisasi_lantai)
    df['LUAS TEMPAT TINGGAL']    = df['LUAS TEMPAT TINGGAL'].apply(normalisasi_luas)
    df['PENDIDIKAN']             = df['PENDIDIKAN'].apply(normalisasi_pendidikan)
    df['JENIS USAHA/PEKERJAAN']  = df['JENIS USAHA/PEKERJAAN'].apply(normalisasi_pekerjaan)
    df['KEPEMILIKAN ASET']       = df['KEPEMILIKAN ASET'].apply(normalisasi_aset)

    # ---- 4.5 Bersihkan kolom JUMLAH TANGGUNGAN ----
    # Beberapa baris memiliki nilai non-numerik (misal: 'KEBAKARAN', 'PUTING BELIUNG')
    # Nilai seperti itu diasumsikan sebagai kasus khusus, diisi dengan median
    df['JUMLAH TANGGUNGAN'] = pd.to_numeric(df['JUMLAH TANGGUNGAN'], errors='coerce')
    median_tanggungan = df['JUMLAH TANGGUNGAN'].median()
    df['JUMLAH TANGGUNGAN'] = df['JUMLAH TANGGUNGAN'].fillna(median_tanggungan)
    df['JUMLAH TANGGUNGAN'] = df['JUMLAH TANGGUNGAN'].astype(int)

    # ---- 4.6 Buat label kelas (STATUS) ----
    # Menentukan apakah tiap orang termasuk Fakir, Miskin, atau Mampu
    df['STATUS'] = df.apply(
        lambda row: tentukan_status(
            row['TOTAL_PENDAPATAN'],
            row['JUMLAH TANGGUNGAN'],
            row['PENDIDIKAN'],
            row['JENIS LANTAI'],
            row['JENIS DINDING'],
            row['LUAS TEMPAT TINGGAL'],
            row['KEPEMILIKAN ASET']
        ),
        axis=1
    )

    # ---- 4.7 Buat fitur turunan ekonomi ----
    # Fitur ini membantu model membedakan keluarga rentan dari keluarga yang
    # memiliki pendapatan, aset, dan kondisi rumah yang relatif mapan.
    df['JUMLAH_ANGGOTA_KELUARGA'] = df['JUMLAH TANGGUNGAN'].apply(hitung_jumlah_anggota)
    df['KEBUTUHAN_DASAR'] = df.apply(
        lambda row: hitung_kebutuhan_dasar(
            row['JUMLAH TANGGUNGAN'],
            row['JENIS LANTAI'],
            row['JENIS DINDING'],
            row['LUAS TEMPAT TINGGAL']
        ),
        axis=1
    )
    df['PENDAPATAN_PER_ANGGOTA'] = df['TOTAL_PENDAPATAN'] / df['JUMLAH_ANGGOTA_KELUARGA']
    df['RASIO_PENDAPATAN_KEBUTUHAN'] = df['TOTAL_PENDAPATAN'] / df['KEBUTUHAN_DASAR']
    df['SKOR_KONDISI_RUMAH'] = df.apply(
        lambda row: skor_kondisi_rumah(
            row['LUAS TEMPAT TINGGAL'],
            row['JENIS DINDING'],
            row['JENIS LANTAI']
        ),
        axis=1
    )
    df['SKOR_KEMAMPUAN_EKONOMI'] = df.apply(
        lambda row: hitung_skor_kemampuan_ekonomi(
            row['TOTAL_PENDAPATAN'],
            row['JUMLAH TANGGUNGAN'],
            row['KEPEMILIKAN ASET'],
            row['JENIS LANTAI'],
            row['JENIS DINDING'],
            row['LUAS TEMPAT TINGGAL']
        ),
        axis=1
    )

    # ---- 4.8 Buat kolom KELAYAKAN berdasarkan STATUS ----
    # Fakir & Miskin -> Layak; Mampu -> Tidak Layak
    df['KELAYAKAN'] = df['STATUS'].apply(tentukan_kelayakan)

    print(f"\nDistribusi kelas (STATUS):")
    print(df['STATUS'].value_counts())          # Tampilkan jumlah per kelas

    print(f"\nDistribusi Kelayakan Penerima Bantuan:")
    print(df['KELAYAKAN'].value_counts())
    total = len(df)
    layak = (df['KELAYAKAN'] == 'Layak').sum()
    tidak_layak = (df['KELAYAKAN'] == 'Tidak Layak').sum()
    print(f"  Layak     : {layak} orang ({layak/total*100:.1f}%)")
    print(f"  Tidak Layak: {tidak_layak} orang ({tidak_layak/total*100:.1f}%)")

    return df


# =============================================================================
# BAGIAN 5: TRANSFORMASI FITUR (ENCODING)
# Mengubah semua fitur kategorik menjadi angka agar bisa diproses model ML
# =============================================================================

def siapkan_fitur(df):
    """
    Memilih fitur yang relevan dan mengubahnya menjadi representasi numerik
    menggunakan LabelEncoder (setiap nilai unik mendapat angka berbeda).
    
    Fitur yang digunakan:
    - TOTAL_PENDAPATAN     : pendapatan gabungan suami+istri
    - JUMLAH TANGGUNGAN    : jumlah anggota keluarga yang ditanggung
    - JENIS USAHA/PEKERJAAN: jenis pekerjaan (dikodekan)
    - KEPEMILIKAN ASET     : 0 = tidak ada, 1 = ada
    - LUAS TEMPAT TINGGAL  : ukuran rumah (dikodekan)
    - JENIS DINDING        : material dinding (dikodekan)
    - JENIS LANTAI         : material lantai (dikodekan)
    - PENDAPATAN_PER_ANGGOTA, RASIO, dan skor ekonomi sebagai indikator kemampuan
    
    Returns:
        X (np.array): matriks fitur
        y (np.array): array label kelas
        nama_fitur (list): nama kolom fitur
        label_encoder_y (LabelEncoder): encoder untuk kelas output
    """
    print("\n" + "=" * 60)
    print("TRANSFORMASI FITUR")
    print("=" * 60)

    # Pilih kolom fitur yang akan digunakan sebagai prediktor
    kolom_fitur = [
        'TOTAL_PENDAPATAN',
        'JUMLAH TANGGUNGAN',
        'JENIS USAHA/PEKERJAAN',
        'KEPEMILIKAN ASET',
        'LUAS TEMPAT TINGGAL',
        'JENIS DINDING',
        'JENIS LANTAI',
        'JUMLAH_ANGGOTA_KELUARGA',
        'KEBUTUHAN_DASAR',
        'PENDAPATAN_PER_ANGGOTA',
        'RASIO_PENDAPATAN_KEBUTUHAN',
        'SKOR_KONDISI_RUMAH',
        'SKOR_KEMAMPUAN_EKONOMI',
    ]

    # Kolom kategorik yang perlu diubah ke angka
    kolom_kategorik = [
        'JENIS USAHA/PEKERJAAN',
        'LUAS TEMPAT TINGGAL',
        'JENIS DINDING',
        'JENIS LANTAI',
    ]

    df_fitur = df[kolom_fitur].copy()           # Salin subset fitur

    # Enkode setiap kolom kategorik menggunakan LabelEncoder
    encoders = {}                               # Simpan encoder agar bisa dipakai saat prediksi
    for kol in kolom_kategorik:
        le = LabelEncoder()
        df_fitur[kol] = le.fit_transform(df_fitur[kol].astype(str))
        encoders[kol] = le                      # Simpan encoder per kolom
        print(f"  {kol}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Enkode label target (STATUS: Fakir/Miskin/Mampu)
    le_y = LabelEncoder()
    y = le_y.fit_transform(df['STATUS'])
    print(f"\n  Label kelas: {dict(zip(le_y.classes_, le_y.transform(le_y.classes_)))}")

    X = df_fitur.values                         # Konversi ke array numpy
    print(f"\nBentuk matriks fitur X: {X.shape}")
    print(f"Jumlah kelas unik: {len(le_y.classes_)}")

    return X, y, kolom_fitur, le_y, encoders


# =============================================================================
# BAGIAN 6: TRAINING DAN EVALUASI MODEL
# =============================================================================

def latih_dan_evaluasi(X, y, nama_fitur, le_y):
    """
    Melatih kedua model (Naive Bayes dan Decision Tree C4.5),
    mengevaluasi performa, dan membandingkan hasilnya.
    
    Args:
        X: matriks fitur (numpy array)
        y: array label kelas (numpy array)
        nama_fitur: nama kolom fitur
        le_y: LabelEncoder untuk kelas output
    """

    # ---- 6.1 Split data: 80% training, 20% testing ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,          # 20% untuk pengujian
        random_state=42,        # Seed agar hasil reproducible
        stratify=y              # Jaga proporsi kelas pada split
    )
    print(f"\n{'=' * 60}")
    print("TRAINING & EVALUASI MODEL")
    print(f"{'=' * 60}")
    print(f"Ukuran data latih : {X_train.shape[0]} sampel")
    print(f"Ukuran data uji   : {X_test.shape[0]} sampel")

    # ---- 6.2 Naive Bayes ----
    # GaussianNB cocok untuk fitur numerik yang diasumsikan berdistribusi normal
    # Untuk Naive Bayes, fitur perlu dinormalisasi agar tidak ada fitur yang
    # mendominasi karena skala nilainya jauh lebih besar (mis. pendapatan vs dinding)
    print(f"\n{'─' * 60}")
    print("  [MODEL 1] NAIVE BAYES")
    print(f"{'─' * 60}")

    scaler = StandardScaler()                   # Normalisasi fitur (Z-score)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)   # Gunakan scaler yang sama dari training

    model_nb = GaussianNB()
    model_nb.fit(X_train_scaled, y_train)       # Latih model Naive Bayes

    y_pred_nb = model_nb.predict(X_test_scaled) # Prediksi pada data uji
    akurasi_nb = accuracy_score(y_test, y_pred_nb)

    print(f"  Akurasi       : {akurasi_nb:.4f} ({akurasi_nb*100:.2f}%)")
    print(f"\n  Laporan Klasifikasi:")
    print(classification_report(
        y_test, y_pred_nb,
        target_names=le_y.classes_,
        zero_division=0             # Hindari error saat ada kelas tanpa prediksi
    ))

    print(f"  Matriks Konfusi:")
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    print(f"  Kelas: {le_y.classes_}")
    print(cm_nb)

    # Cross-validation Naive Bayes (5-fold) untuk estimasi performa yang lebih stabil
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_nb_model = make_pipeline(StandardScaler(), GaussianNB())
    cv_nb = cross_val_score(cv_nb_model, X, y, cv=cv, scoring='accuracy')
    print(f"\n  Cross-Validation (5-fold): {cv_nb.mean():.4f} ± {cv_nb.std():.4f}")

    # ---- 6.3 Decision Tree C4.5 ----
    # Scikit-learn mengimplementasikan C4.5 menggunakan criterion='entropy'
    # (Information Gain, yang merupakan dasar dari C4.5)
    # max_depth=5 mencegah overfitting pada dataset kecil
    print(f"\n{'─' * 60}")
    print("  [MODEL 2] DECISION TREE C4.5 (criterion=entropy)")
    print(f"{'─' * 60}")

    model_dt = DecisionTreeClassifier(
        criterion='entropy',    # C4.5 menggunakan Information Gain (entropy)
        max_depth=5,            # Batasi kedalaman pohon agar tidak overfit
        min_samples_split=5,    # Minimal 5 sampel untuk membelah node
        min_samples_leaf=2,     # Minimal 2 sampel di setiap daun
        random_state=42         # Seed untuk reproducibility
    )
    model_dt.fit(X_train, y_train)              # Latih model Decision Tree

    y_pred_dt = model_dt.predict(X_test)        # Prediksi pada data uji
    akurasi_dt = accuracy_score(y_test, y_pred_dt)

    print(f"  Akurasi       : {akurasi_dt:.4f} ({akurasi_dt*100:.2f}%)")
    print(f"\n  Laporan Klasifikasi:")
    print(classification_report(
        y_test, y_pred_dt,
        target_names=le_y.classes_,
        zero_division=0
    ))

    print(f"  Matriks Konfusi:")
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    print(f"  Kelas: {le_y.classes_}")
    print(cm_dt)

    # Cross-validation Decision Tree (5-fold)
    cv_dt = cross_val_score(model_dt, X, y, cv=cv, scoring='accuracy')
    print(f"\n  Cross-Validation (5-fold): {cv_dt.mean():.4f} ± {cv_dt.std():.4f}")

    # ---- 6.4 Tampilkan Struktur Pohon Keputusan ----
    print(f"\n{'─' * 60}")
    print("  STRUKTUR POHON KEPUTUSAN (max 3 level pertama):")
    print(f"{'─' * 60}")
    struktur_pohon = export_text(
        model_dt,
        feature_names=nama_fitur,   # Gunakan nama fitur agar mudah dibaca
        max_depth=3                  # Hanya tampilkan 3 level teratas
    )
    print(struktur_pohon)

    # ---- 6.5 Kepentingan Fitur (Decision Tree) ----
    print(f"{'─' * 60}")
    print("  KEPENTINGAN FITUR (Decision Tree):")
    print(f"{'─' * 60}")
    importances = model_dt.feature_importances_
    # Urutkan dari yang paling penting ke yang kurang penting
    urutan = np.argsort(importances)[::-1]
    for i in urutan:
        print(f"  {nama_fitur[i]:<35}: {importances[i]:.4f}")

    # ---- 6.6 Perbandingan Akhir ----
    print(f"\n{'=' * 60}")
    print("  RINGKASAN PERBANDINGAN ALGORITMA")
    print(f"{'=' * 60}")
    print(f"  {'Algoritma':<30} {'Akurasi Test':<15} {'CV Mean':<15} {'CV Std'}")
    print(f"  {'-'*70}")
    print(f"  {'Naive Bayes':<30} {akurasi_nb*100:<15.2f}% {cv_nb.mean()*100:<15.2f}% {cv_nb.std()*100:.2f}%")
    print(f"  {'Decision Tree C4.5':<30} {akurasi_dt*100:<15.2f}% {cv_dt.mean()*100:<15.2f}% {cv_dt.std()*100:.2f}%")

    # Tentukan pemenang berdasarkan cross-validation mean (lebih representatif)
    if cv_nb.mean() > cv_dt.mean():
        print(f"\n  >> Naive Bayes lebih unggul berdasarkan Cross-Validation.")
        best_algorithm = "naive_bayes"
    elif cv_dt.mean() > cv_nb.mean():
        print(f"\n  >> Decision Tree C4.5 lebih unggul berdasarkan Cross-Validation.")
        best_algorithm = "decision_tree"
    else:
        print(f"\n  >> Kedua algoritma memiliki performa yang setara.")
        best_algorithm = "decision_tree"

    # ---- Ringkasan Kelayakan pada Data Uji ----
    print(f"\n{'=' * 60}")
    print("  RINGKASAN KELAYAKAN PENERIMA BANTUAN (DATA UJI)")
    print(f"{'=' * 60}")

    # Kembalikan label asli ke nama kelas
    kelas_asli  = le_y.inverse_transform(y_test)
    kelas_nb    = le_y.inverse_transform(y_pred_nb)
    kelas_dt    = le_y.inverse_transform(y_pred_dt)

    # Konversi ke kelayakan
    kelayakan_asli = [tentukan_kelayakan(s) for s in kelas_asli]
    kelayakan_nb   = [tentukan_kelayakan(s) for s in kelas_nb]
    kelayakan_dt   = [tentukan_kelayakan(s) for s in kelas_dt]

    total_uji = len(y_test)

    for label_algoritma, kelayakan_pred in [
        ("Naive Bayes",       kelayakan_nb),
        ("Decision Tree C4.5", kelayakan_dt),
    ]:
        layak     = kelayakan_pred.count('Layak')
        tdk_layak = kelayakan_pred.count('Tidak Layak')
        print(f"\n  [{label_algoritma}]")
        print(f"    Layak      : {layak} orang ({layak/total_uji*100:.1f}%)")
        print(f"    Tidak Layak: {tdk_layak} orang ({tdk_layak/total_uji*100:.1f}%)")

    # Akurasi kelayakan (bukan akurasi per kelas, tapi akurasi Layak vs Tidak Layak)
    from sklearn.metrics import accuracy_score as acc
    akurasi_kel_nb = acc(kelayakan_asli, kelayakan_nb)
    akurasi_kel_dt = acc(kelayakan_asli, kelayakan_dt)
    print(f"\n  Akurasi Kelayakan (Layak/Tidak Layak):")
    print(f"    Naive Bayes      : {akurasi_kel_nb*100:.2f}%")
    print(f"    Decision Tree C4.5: {akurasi_kel_dt*100:.2f}%")

    metrics = {
        "naive_bayes": {
            "accuracy_test": float(akurasi_nb),
            "cv_mean": float(cv_nb.mean()),
            "cv_std": float(cv_nb.std()),
        },
        "decision_tree": {
            "accuracy_test": float(akurasi_dt),
            "cv_mean": float(cv_dt.mean()),
            "cv_std": float(cv_dt.std()),
        },
        "best_algorithm": best_algorithm,
        "selection_metric": "cv_mean",
    }

    comparison_summary = build_comparison_summary(metrics)

    return model_nb, model_dt, scaler, metrics, comparison_summary


def build_comparison_summary(metrics):
    """
    Menyusun ringkasan analisis perbandingan algoritma agar mudah dipakai
    ulang di backend API maupun frontend.
    """
    selection_metric = metrics.get("selection_metric", "cv_mean")

    rows = []
    for key, label in [
        ("naive_bayes", "Naive Bayes"),
        ("decision_tree", "Decision Tree C4.5"),
    ]:
        model_metrics = metrics.get(key, {})
        selection_score = float(model_metrics.get(selection_metric, model_metrics.get("accuracy_test", 0.0)))
        rows.append({
            "key": key,
            "nama_algoritma": label,
            "akurasi_test": float(model_metrics.get("accuracy_test", 0.0)),
            "cv_mean": float(model_metrics.get("cv_mean", 0.0)),
            "cv_std": float(model_metrics.get("cv_std", 0.0)),
            "selection_score": selection_score,
        })

    rows = sorted(rows, key=lambda item: item["selection_score"], reverse=True)
    winner = rows[0] if rows else {}
    runner_up = rows[1] if len(rows) > 1 else {}
    score_gap = float(winner.get("selection_score", 0.0) - runner_up.get("selection_score", 0.0)) if runner_up else 0.0

    selection_label = "Cross-Validation Mean" if selection_metric == "cv_mean" else selection_metric.replace("_", " ").title()
    if winner:
        summary_text = (
            f"{winner['nama_algoritma']} unggul berdasarkan {selection_label} "
            f"dengan selisih {score_gap:.4f} dibanding pesaing terdekat."
        )
    else:
        summary_text = "Ringkasan perbandingan belum tersedia."

    return {
        "selection_metric": selection_metric,
        "selection_label": selection_label,
        "winner_key": winner.get("key"),
        "winner_label": winner.get("nama_algoritma"),
        "runner_up_key": runner_up.get("key"),
        "score_gap": score_gap,
        "rows": rows,
        "summary": summary_text,
        "recommendation": (
            f"Gunakan {winner['nama_algoritma']} sebagai model utama."
            if winner
            else "Gunakan model dengan performa terbaik yang tersedia."
        ),
    }


# =============================================================================
# BAGIAN 7: FUNGSI PREDIKSI DATA BARU
# Memungkinkan pengguna memprediksi status seseorang secara manual
# =============================================================================

def prediksi_baru(model_nb, model_dt, scaler, le_y, encoders, nama_fitur):
    """
    Contoh penggunaan model untuk memprediksi status individu baru
    yang belum ada dalam dataset.
    
    Args:
        model_nb: model Naive Bayes yang sudah dilatih
        model_dt: model Decision Tree yang sudah dilatih
        scaler: StandardScaler yang sudah di-fit pada data latih
        le_y: LabelEncoder untuk label kelas
        encoders: dictionary LabelEncoder per kolom kategorik
        nama_fitur: urutan fitur yang sama dengan data training
    """
    print(f"\n{'=' * 60}")
    print("  CONTOH PREDIKSI DATA BARU")
    print(f"{'=' * 60}")

    # ---- Definisi data individu baru yang ingin diprediksi ----
    # Format sama seperti fitur training
    contoh = {
        'TOTAL_PENDAPATAN'      : 2_500_000,    # Total pendapatan gabungan: Rp 2.500.000
        'JUMLAH TANGGUNGAN'     : 1,            # Tanggungan rendah
        'JENIS USAHA/PEKERJAAN' : 'karyawan',   # Pekerjaan stabil
        'KEPEMILIKAN ASET'      : 1,            # Memiliki aset
        'LUAS TEMPAT TINGGAL'   : 'besar',      # Rumah besar
        'JENIS DINDING'         : 'tembok',     # Dinding permanen
        'JENIS LANTAI'          : 'keramik',    # Lantai ubin/keramik
    }

    contoh['JUMLAH_ANGGOTA_KELUARGA'] = hitung_jumlah_anggota(contoh['JUMLAH TANGGUNGAN'])
    contoh['KEBUTUHAN_DASAR'] = hitung_kebutuhan_dasar(
        contoh['JUMLAH TANGGUNGAN'],
        contoh['JENIS LANTAI'],
        contoh['JENIS DINDING'],
        contoh['LUAS TEMPAT TINGGAL']
    )
    contoh['PENDAPATAN_PER_ANGGOTA'] = contoh['TOTAL_PENDAPATAN'] / contoh['JUMLAH_ANGGOTA_KELUARGA']
    contoh['RASIO_PENDAPATAN_KEBUTUHAN'] = contoh['TOTAL_PENDAPATAN'] / contoh['KEBUTUHAN_DASAR']
    contoh['SKOR_KONDISI_RUMAH'] = skor_kondisi_rumah(
        contoh['LUAS TEMPAT TINGGAL'],
        contoh['JENIS DINDING'],
        contoh['JENIS LANTAI']
    )
    contoh['SKOR_KEMAMPUAN_EKONOMI'] = hitung_skor_kemampuan_ekonomi(
        contoh['TOTAL_PENDAPATAN'],
        contoh['JUMLAH TANGGUNGAN'],
        contoh['KEPEMILIKAN ASET'],
        contoh['JENIS LANTAI'],
        contoh['JENIS DINDING'],
        contoh['LUAS TEMPAT TINGGAL']
    )

    vektor = []
    for kol in nama_fitur:
        nilai = contoh[kol]
        if kol in encoders:
            # Jika nilai tidak dikenali encoder, gunakan nilai paling umum
            try:
                nilai = encoders[kol].transform([str(nilai)])[0]
            except ValueError:
                nilai = 0                       # Fallback ke 0 jika tidak dikenali
        vektor.append(nilai)

    X_baru = np.array([vektor])                 # Bentuk: (1, n_fitur)

    # Prediksi menggunakan Naive Bayes (perlu dinormalisasi dulu)
    X_baru_scaled = scaler.transform(X_baru)
    pred_nb = le_y.inverse_transform(model_nb.predict(X_baru_scaled))[0]
    prob_nb = model_nb.predict_proba(X_baru_scaled)[0]

    # Prediksi menggunakan Decision Tree (tidak perlu dinormalisasi)
    pred_dt = le_y.inverse_transform(model_dt.predict(X_baru))[0]
    prob_dt = model_dt.predict_proba(X_baru)[0]

    print(f"  Data input:")
    for k, v in contoh.items():
        print(f"    {k:<30}: {v}")

    # Tentukan kelayakan berdasarkan prediksi masing-masing model
    kelayakan_nb = tentukan_kelayakan(pred_nb)
    kelayakan_dt = tentukan_kelayakan(pred_dt)

    print(f"\n  Hasil Prediksi:")
    print(f"  {'Algoritma':<25} {'Prediksi':<12} {'Kelayakan':<15} {'Probabilitas per kelas'}")
    print(f"  {'-'*80}")
    print(f"  {'Naive Bayes':<25} {pred_nb:<12} {kelayakan_nb:<15} {dict(zip(le_y.classes_, prob_nb.round(3)))}")
    print(f"  {'Decision Tree C4.5':<25} {pred_dt:<12} {kelayakan_dt:<15} {dict(zip(le_y.classes_, prob_dt.round(3)))}")

    print(f"\n  Kesimpulan:")
    # Ambil keputusan berdasarkan majority vote kedua model
    if kelayakan_nb == kelayakan_dt:
        print(f"  >> Kedua model sepakat: individu ini {kelayakan_nb.upper()} sebagai penerima bantuan.")
    else:
        print(f"  >> Model berbeda pendapat:")
        print(f"     - Naive Bayes     : {kelayakan_nb}")
        print(f"     - Decision Tree   : {kelayakan_dt}")
        print(f"  >> Disarankan tinjau ulang secara manual.")


# =============================================================================
# BAGIAN 8: TAMPILKAN SAMPEL DATASET YANG SUDAH DIPROSES
# =============================================================================

def tampilkan_sampel(df):
    """
    Menampilkan 10 baris pertama dataset yang sudah diproses beserta
    kolom yang relevan untuk verifikasi visual.
    """
    print(f"\n{'=' * 60}")
    print("  SAMPEL DATASET YANG SUDAH DIPROSES (10 baris)")
    print(f"{'=' * 60}")

    kolom_tampil = [
        'NAMA', 'TOTAL_PENDAPATAN', 'JUMLAH TANGGUNGAN',
        'JENIS USAHA/PEKERJAAN', 'LUAS TEMPAT TINGGAL',
        'JENIS DINDING', 'JENIS LANTAI', 'PENDIDIKAN', 'STATUS', 'KELAYAKAN'
    ]

    # Atur agar semua kolom tampil saat print
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 20)

    print(df[kolom_tampil].head(10).to_string(index=False))


# =============================================================================
# BAGIAN 9: EXPORT MODEL KE FILE PKL
# =============================================================================

def export_model_artifacts(model_nb, model_dt, scaler, le_y, encoders, nama_fitur, metrics, comparison_summary):
    """
    Menyimpan model dan artefak pendukung ke folder artifacts/.

    Artefak yang disimpan:
    - model_nb_status.pkl      : model Naive Bayes
    - model_dt_status.pkl      : model Decision Tree
    - scaler_nb.pkl            : StandardScaler untuk Naive Bayes
    - label_encoder_y.pkl      : encoder label target
    - feature_encoders.pkl     : encoder per kolom kategorik
    - feature_names.pkl        : daftar nama fitur
    - model_metrics.pkl        : ringkasan evaluasi model
    - comparison_summary.pkl   : ringkasan perbandingan algoritma
    - model_bundle.pkl        : bundle lengkap dalam satu file
    """
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    artifacts = {
        "model_nb_status.pkl": model_nb,
        "model_dt_status.pkl": model_dt,
        "scaler_nb.pkl": scaler,
        "label_encoder_y.pkl": le_y,
        "feature_encoders.pkl": encoders,
        "feature_names.pkl": nama_fitur,
        "model_metrics.pkl": metrics,
        "comparison_summary.pkl": comparison_summary,
    }

    for filename, obj in artifacts.items():
        joblib.dump(obj, output_dir / filename)

    bundle = {
        "model_nb": model_nb,
        "model_dt": model_dt,
        "scaler": scaler,
        "label_encoder_y": le_y,
        "feature_encoders": encoders,
        "feature_names": nama_fitur,
        "metrics": metrics,
        "comparison_summary": comparison_summary,
    }
    joblib.dump(bundle, output_dir / "model_bundle.pkl")

    print(f"\n{'=' * 60}")
    print("  MODEL BERHASIL DI-EKSPOR KE FILE PKL")
    print(f"{'=' * 60}")
    for filename in artifacts.keys():
        print(f"  - {output_dir / filename}")
    print(f"  - {output_dir / 'model_bundle.pkl'}")


# =============================================================================
# BAGIAN 10: MAIN - TITIK MASUK PROGRAM
# =============================================================================

if __name__ == '__main__':
    """
    Alur eksekusi program:
    1. Load dan proses data mentah
    2. Tampilkan sampel dataset
    3. Siapkan fitur (encoding)
    4. Latih dan evaluasi kedua model
    5. Contoh prediksi data baru
    6. Simpan artefak model ke file .pkl
    """

    # Langkah 1: Load dan proses data
    df = load_dan_proses_data()

    # Langkah 2: Tampilkan sampel untuk verifikasi
    tampilkan_sampel(df)

    # Langkah 3: Siapkan fitur untuk modeling
    X, y, nama_fitur, le_y, encoders = siapkan_fitur(df)

    # Langkah 4: Latih dan evaluasi model
    model_nb, model_dt, scaler, metrics, comparison_summary = latih_dan_evaluasi(X, y, nama_fitur, le_y)

    # Langkah 5: Contoh prediksi individu baru
    prediksi_baru(model_nb, model_dt, scaler, le_y, encoders, nama_fitur)

    # Langkah 6: Simpan model dan artefak ke file .pkl
    export_model_artifacts(model_nb, model_dt, scaler, le_y, encoders, nama_fitur, metrics, comparison_summary)

    print(f"\n{'=' * 60}")
    print("  SELESAI")
    print(f"{'=' * 60}")
