import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# 2. KONFIGURASI
# ==========================================================
FILE_PATH = "datasets/baznas-5.csv"
TARGET = "KELAS TARGET"
RIWAYAT = "RIWAYAT PENERIMA BANTUAN (0-3)"
RANDOM_STATE = 42

# ==========================================================
# 3. MEMBACA DATASET
# ==========================================================
df = pd.read_csv(FILE_PATH)

print("Jumlah data:", df.shape)
print("\nKolom:")
print(df.columns.tolist())

print("\nDistribusi kelas:")
print(df[TARGET].value_counts())

print("\nMissing value:")
print(df.isnull().sum())


# ==========================================================
# 4. DATA CLEANING
# ==========================================================

# Merapikan spasi pada kolom teks
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# Konversi kolom pendapatan menjadi numerik
df["SUAMI"] = pd.to_numeric(df["SUAMI"], errors="coerce").fillna(0)
df["ISTRI"] = pd.to_numeric(df["ISTRI"], errors="coerce").fillna(0)

# Konversi jumlah tanggungan
df["JUMLAH TANGGUNGAN"] = pd.to_numeric(
    df["JUMLAH TANGGUNGAN"],
    errors="coerce"
).fillna(0)

# Pastikan riwayat bantuan hanya pada skala 0 sampai 3
df['RIWAYAT'] = pd.to_numeric(
    df['RIWAYAT'],
    errors="coerce"
).fillna(0)

df['RIWAYAT'] = df['RIWAYAT'].clip(0, 3).astype(int)

# Menghapus duplikasi berdasarkan Nomor KK
# df = df.drop_duplicates(subset=["NO KARTU KELUARGA"])

# Menghapus data tanpa kelas target
df = df.dropna(subset=["RIWAYAT"]).copy()

print("Jumlah data setelah cleaning:", df.shape)


df.head()  # Tampilkan 5 baris pertama dataset setelah cleaning


# ==========================================================
# 5. FEATURE ENGINEERING
# ==========================================================

# Total pendapatan keluarga
df["TOTAL_PENDAPATAN"] = df["SUAMI"] + df["ISTRI"]

# Pendapatan per kapita
df["PENDAPATAN_PER_KAPITA"] = (
    df["TOTAL_PENDAPATAN"] /
    (df["JUMLAH TANGGUNGAN"] + 1)
)

# Kategori pendapatan
def kategori_pendapatan(nilai):
    if nilai <= 850_000:
        return "Sangat Rendah"
    elif nilai <= 1_700_000:
        return "Rendah"
    elif nilai <= 3_400_000:
        return "Sedang"
    else:
        return "Tinggi"

# Kategori jumlah tanggungan
def kategori_tanggungan(nilai):
    if nilai <= 1:
        return "Rendah"
    elif nilai <= 4:
        return "Sedang"
    else:
        return "Tinggi"

df["KATEGORI_TOTAL_PENDAPATAN"] = (
    df["TOTAL_PENDAPATAN"]
    .apply(kategori_pendapatan)
)

df["KATEGORI_PENDAPATAN_PER_KAPITA"] = (
    df["PENDAPATAN_PER_KAPITA"]
    .apply(kategori_pendapatan)
)

df["KATEGORI_TANGGUNGAN"] = (
    df["JUMLAH TANGGUNGAN"]
    .apply(kategori_tanggungan)
)

# Kategori riwayat bantuan
df["KATEGORI_RIWAYAT"] = df['RIWAYAT'].map({
    0: "Belum Pernah",
    1: "1 Kali",
    2: "2 Kali",
    3: "3 Kali atau Lebih"
})

print(
    df[
        [
            "TOTAL_PENDAPATAN",
            "PENDAPATAN_PER_KAPITA",
            "KATEGORI_TOTAL_PENDAPATAN",
            "KATEGORI_PENDAPATAN_PER_KAPITA",
            "KATEGORI_TANGGUNGAN",
            "KATEGORI_RIWAYAT"
        ]
    ].head()
)

# ==========================================================
# 6. MENENTUKAN FITUR MODEL
# ==========================================================

# Catatan: "KEPEMILIKAN ASET" TIDAK dipakai sebagai fitur pada kedua
# algoritma (Naive Bayes & C4.5) karena nilainya menentukan KELAS TARGET
# 100% (data leakage). Kolomnya tetap ditampilkan di form web, hanya
# saja tidak dijadikan variabel pembelajaran model.

fitur_tanpa_riwayat = [
    "JENIS USAHA/PEKERJAAN",
    "LUAS TEMPAT TINGGAL",
    "JENIS DINDING",
    "JENIS LANTAI",
    "PENDIDIKAN",
    "KATEGORI_TOTAL_PENDAPATAN",
    "KATEGORI_PENDAPATAN_PER_KAPITA",
    "KATEGORI_TANGGUNGAN"
]

fitur_dengan_riwayat = fitur_tanpa_riwayat + [
    "KATEGORI_RIWAYAT"
]


# ==========================================================
# 7. FUNGSI EVALUASI
# ==========================================================

def evaluasi_model(nama_model, y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["TIDAK LAYAK", "LAYAK"]
    )

    tn, fp, fn, tp = cm.ravel()

    hasil = {
        "Model": nama_model,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision_LAYAK": round(
            precision_score(
                y_true,
                y_pred,
                pos_label="LAYAK",
                zero_division=0
            ),
            4
        ),
        "Recall_LAYAK": round(
            recall_score(
                y_true,
                y_pred,
                pos_label="LAYAK",
                zero_division=0
            ),
            4
        ),
        "F1_LAYAK": round(
            f1_score(
                y_true,
                y_pred,
                pos_label="LAYAK",
                zero_division=0
            ),
            4
        ),
        "True_Negative": int(tn),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "True_Positive": int(tp)
    }

    print(f"\n===== {nama_model} =====")

    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            cm,
            index=["Aktual TIDAK LAYAK", "Aktual LAYAK"],
            columns=["Prediksi TIDAK LAYAK", "Prediksi LAYAK"]
        )
    )

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return hasil


# ==========================================================
# 8. MODEL NAIVE BAYES + CROSS VALIDATION
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import (
    OrdinalEncoder,
    FunctionTransformer
)

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import CategoricalNB


def tambah_satu(X):
    """
    Mengubah kategori tidak dikenal dari -1 menjadi 0
    dan kategori yang dikenal dimulai dari 1.
    """
    return X.astype(int) + 1


def jalankan_naive_bayes(df_model, fitur, nama_skenario):

    X = df_model[fitur].fillna("Tidak Diketahui").astype(str)
    y = df_model[TARGET].astype(str)

    # ======================================================
    # CROSS VALIDATION
    # ======================================================

    # Pipeline diperlukan agar encoder dilatih ulang
    # pada setiap fold dan mencegah data leakage.
    pipeline_cv = Pipeline([
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        ),
        (
            "tambah_satu",
            FunctionTransformer(
                tambah_satu,
                validate=False
            )
        ),
        (
            "naive_bayes",
            CategoricalNB(alpha=1.0)
        )
    ])

    stratified_kfold = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),

        "precision": make_scorer(
            precision_score,
            pos_label="LAYAK",
            zero_division=0
        ),

        "recall": make_scorer(
            recall_score,
            pos_label="LAYAK",
            zero_division=0
        ),

        "f1": make_scorer(
            f1_score,
            pos_label="LAYAK",
            zero_division=0
        )
    }

    hasil_cv = cross_validate(
        estimator=pipeline_cv,
        X=X,
        y=y,
        cv=stratified_kfold,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    print("\n" + "=" * 65)
    print(f"5-FOLD CROSS-VALIDATION NAIVE BAYES - {nama_skenario}")
    print("=" * 65)

    for fold in range(5):
        print(
            f"Fold {fold + 1}: "
            f"Accuracy={hasil_cv['test_accuracy'][fold]:.4f} | "
            f"Precision={hasil_cv['test_precision'][fold]:.4f} | "
            f"Recall={hasil_cv['test_recall'][fold]:.4f} | "
            f"F1-Score={hasil_cv['test_f1'][fold]:.4f}"
        )

    print("\nRata-rata Cross-Validation:")

    print(
        f"Accuracy  : "
        f"{hasil_cv['test_accuracy'].mean():.4f} "
        f"± {hasil_cv['test_accuracy'].std():.4f}"
    )

    print(
        f"Precision : "
        f"{hasil_cv['test_precision'].mean():.4f} "
        f"± {hasil_cv['test_precision'].std():.4f}"
    )

    print(
        f"Recall    : "
        f"{hasil_cv['test_recall'].mean():.4f} "
        f"± {hasil_cv['test_recall'].std():.4f}"
    )

    print(
        f"F1-Score  : "
        f"{hasil_cv['test_f1'].mean():.4f} "
        f"± {hasil_cv['test_f1'].std():.4f}"
    )

    # ======================================================
    # HOLDOUT TEST 80:20
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Encoding kategori menjadi angka
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train_encoded = (
        encoder.fit_transform(X_train).astype(int) + 1
    )

    X_test_encoded = (
        encoder.transform(X_test).astype(int) + 1
    )

    # Training Naive Bayes
    model_nb = CategoricalNB(alpha=1.0)

    model_nb.fit(
        X_train_encoded,
        y_train
    )

    # Prediksi
    y_pred = model_nb.predict(X_test_encoded)

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    urutan_kelas = [
        "TIDAK LAYAK",
        "LAYAK"
    ]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=urutan_kelas
    )

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Tidak Layak", "Layak"],
        yticklabels=["Tidak Layak", "Layak"]
    )


    

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        f"Confusion Matrix Naive Bayes - {nama_skenario}"
    )

    plt.tight_layout()
    plt.show()

    # ======================================================
    # PROBABILITAS PREDIKSI
    # ======================================================

    probabilitas = model_nb.predict_proba(
        X_test_encoded
    )

    # Menentukan posisi kelas berdasarkan model.classes_
    indeks_tidak_layak = np.where(
        model_nb.classes_ == "TIDAK LAYAK"
    )[0][0]

    indeks_layak = np.where(
        model_nb.classes_ == "LAYAK"
    )[0][0]

    hasil = evaluasi_model(
        f"Naive Bayes - {nama_skenario}",
        y_test,
        y_pred
    )

    hasil_prediksi = X_test.copy()

    hasil_prediksi["AKTUAL"] = y_test.values
    hasil_prediksi["PREDIKSI"] = y_pred

    hasil_prediksi["PROBABILITAS_TIDAK_LAYAK"] = (
        probabilitas[:, indeks_tidak_layak]
    )

    hasil_prediksi["PROBABILITAS_LAYAK"] = (
        probabilitas[:, indeks_layak]
    )

    # Return tetap sama seperti kode awal
    return hasil, hasil_prediksi, model_nb, encoder



hasil_nb_tanpa, pred_nb_tanpa, model_nb_tanpa, encoder_nb_tanpa = (
    jalankan_naive_bayes(
        df,
        fitur_tanpa_riwayat,
        "Tanpa Riwayat Bantuan"
    )
)

# Naive Bayes dengan riwayat bantuan
hasil_nb_dengan, pred_nb_dengan, model_nb_dengan, encoder_nb_dengan = (
    jalankan_naive_bayes(
        df,
        fitur_dengan_riwayat,
        "Dengan Riwayat Bantuan"
    )
)


# ==========================================================
# 9. FUNGSI ENTROPY DAN C4.5 GAIN RATIO
# ==========================================================

def entropy(y):
    proporsi = y.value_counts(normalize=True)
    return -(proporsi * np.log2(proporsi)).sum()


class C45Classifier:
    """
    Implementasi C4.5 sederhana.
    Menggunakan Gain Ratio dan pre-pruning.
    Semua fitur harus berbentuk kategori.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=5,
        min_gain_ratio=0.01
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_ratio = min_gain_ratio

    def gain_ratio(self, data, feature):
        entropy_awal = entropy(data["_target"])
        total_data = len(data)

        entropy_setelah_split = 0
        split_info = 0

        for _, subset in data.groupby(feature, dropna=False):
            proporsi = len(subset) / total_data

            entropy_setelah_split += (
                proporsi * entropy(subset["_target"])
            )

            if proporsi > 0:
                split_info -= proporsi * np.log2(proporsi)

        information_gain = entropy_awal - entropy_setelah_split

        if split_info == 0:
            return 0

        return information_gain / split_info

    def build_tree(self, data, features, depth=0):
        target = data["_target"]
        kelas_default = target.mode().iloc[0]

        # Kondisi berhenti
        if target.nunique() == 1:
            return {
                "type": "leaf",
                "label": kelas_default
            }

        if len(features) == 0:
            return {
                "type": "leaf",
                "label": kelas_default
            }

        if depth >= self.max_depth:
            return {
                "type": "leaf",
                "label": kelas_default
            }

        if len(data) < self.min_samples_split:
            return {
                "type": "leaf",
                "label": kelas_default
            }

        # Menghitung Gain Ratio seluruh fitur
        gain_ratios = {
            feature: self.gain_ratio(data, feature)
            for feature in features
        }

        fitur_terbaik = max(gain_ratios, key=gain_ratios.get)
        nilai_gain_ratio = gain_ratios[fitur_terbaik]

        # Jika Gain Ratio terlalu kecil, jadikan leaf
        if nilai_gain_ratio < self.min_gain_ratio:
            return {
                "type": "leaf",
                "label": kelas_default
            }

        node = {
            "type": "node",
            "feature": fitur_terbaik,
            "default": kelas_default,
            "branches": {}
        }

        fitur_sisa = [
            feature
            for feature in features
            if feature != fitur_terbaik
        ]

        for nilai, subset in data.groupby(
            fitur_terbaik,
            dropna=False
        ):
            subset_baru = subset.drop(columns=[fitur_terbaik])

            node["branches"][nilai] = self.build_tree(
                subset_baru,
                fitur_sisa,
                depth + 1
            )

        return node

    def fit(self, X, y):
        data_train = X.copy().reset_index(drop=True)
        data_train["_target"] = pd.Series(y).reset_index(drop=True)

        self.tree_ = self.build_tree(
            data_train,
            list(X.columns)
        )

        return self

    def predict_one(self, row, node):
        if node["type"] == "leaf":
            return node["label"]

        feature = node["feature"]
        nilai = row[feature]

        if nilai not in node["branches"]:
            return node["default"]

        return self.predict_one(
            row,
            node["branches"][nilai]
        )

    def predict(self, X):
        return np.array([
            self.predict_one(row, self.tree_)
            for _, row in X.iterrows()
        ])
    


    # ==========================================================
# 10. MENJALANKAN C4.5
# ==========================================================

def jalankan_c45(df_model, fitur, nama_skenario):

    X = df_model[fitur].fillna("Tidak Diketahui").astype(str)
    y = df_model[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model_c45 = C45Classifier(
        max_depth=5,
        min_samples_split=5,
        min_gain_ratio=0.01
    )

    model_c45.fit(X_train, y_train)

    y_pred = model_c45.predict(X_test)

    hasil = evaluasi_model(
        f"C4.5 - {nama_skenario}",
        y_test,
        y_pred
    )

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Tidak Layak', 'Layak'],
        yticklabels=['Tidak Layak', 'Layak']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix C4.5 - {nama_skenario}')
    plt.tight_layout()
    plt.show()

    hasil_prediksi = X_test.copy()
    hasil_prediksi["AKTUAL"] = y_test.values
    hasil_prediksi["PREDIKSI"] = y_pred

    return hasil, hasil_prediksi, model_c45


# C4.5 tanpa riwayat bantuan
hasil_c45_tanpa, pred_c45_tanpa, model_c45_tanpa = (
    jalankan_c45(
        df,
        fitur_tanpa_riwayat,
        "Tanpa Riwayat Bantuan"
    )
)

# C4.5 dengan riwayat bantuan
hasil_c45_dengan, pred_c45_dengan, model_c45_dengan = (
    jalankan_c45(
        df,
        fitur_dengan_riwayat,
        "Dengan Riwayat Bantuan"
    )
)