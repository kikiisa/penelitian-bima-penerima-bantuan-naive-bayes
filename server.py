from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "baznas-4.csv"
ARTIFACT_PATH = BASE_DIR / "artifacts" / "model_naive_bayes_baznas.pkl"

TARGET = "KELAS TARGET"
RANDOM_STATE = 42

FEATURES = [
    "JENIS USAHA/PEKERJAAN",
    "KEPEMILIKAN ASET",
    "LUAS TEMPAT TINGGAL",
    "JENIS DINDING",
    "JENIS LANTAI",
    "PENDIDIKAN",
    "KATEGORI_TOTAL_PENDAPATAN",
    "KATEGORI_PENDAPATAN_PER_KAPITA",
    "KATEGORI_TANGGUNGAN",
    "KATEGORI_RIWAYAT",
]

RAW_CATEGORICAL_FEATURES = [
    "JENIS USAHA/PEKERJAAN",
    "KEPEMILIKAN ASET",
    "LUAS TEMPAT TINGGAL",
    "JENIS DINDING",
    "JENIS LANTAI",
    "PENDIDIKAN",
]

INPUT_PAYLOAD_FIELDS = [
    *RAW_CATEGORICAL_FEATURES,
    "TOTAL_PENDAPATAN",
    "JUMLAH TANGGUNGAN",
    "RIWAYAT",
]

RAW_FIELD_ALIASES = {
    "JENIS USAHA/PEKERJAAN": ["jenis_usaha_pekerjaan", "JENIS USAHA/PEKERJAAN"],
    "KEPEMILIKAN ASET": ["kepemilikan_aset", "KEPEMILIKAN ASET"],
    "LUAS TEMPAT TINGGAL": ["luas_tempat_tinggal", "LUAS TEMPAT TINGGAL"],
    "JENIS DINDING": ["jenis_dinding", "JENIS DINDING"],
    "JENIS LANTAI": ["jenis_lantai", "JENIS LANTAI"],
    "PENDIDIKAN": ["pendidikan", "PENDIDIKAN"],
}

TANGGUNGAN_NUMBER_ALIASES = ["jumlah_tanggungan", "JUMLAH TANGGUNGAN"]
TOTAL_PENDAPATAN_ALIASES = ["total_pendapatan", "TOTAL_PENDAPATAN", "TOTAL PENDAPATAN"]
RIWAYAT_ALIASES = ["riwayat", "RIWAYAT", "RIWAYAT PENERIMAAN BANTUAN"]
RIWAYAT_MAPPING = {
    0: "Belum Pernah",
    1: "1 Kali",
    2: "2 Kali",
    3: "3 Kali atau Lebih",
}


app = FastAPI(
    title="Klasifikasi Mustahik BAZNAS",
    description="Prediksi KELAS TARGET menggunakan fitur dengan riwayat dari main.ipynb.",
    version="3.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def kategori_pendapatan(nilai: float) -> str:
    if nilai <= 850_000:
        return "Sangat Rendah"
    if nilai <= 1_700_000:
        return "Rendah"
    if nilai <= 3_400_000:
        return "Sedang"
    return "Tinggi"


def _parse_total_pendapatan(value: Any) -> float:
    if isinstance(value, bool):
        raise HTTPException(
            status_code=422,
            detail="TOTAL_PENDAPATAN harus berupa angka, bukan boolean.",
        )

    if isinstance(value, str):
        normalized_value = value.strip().replace(".", "").replace(",", "")
    else:
        normalized_value = value

    try:
        total_pendapatan = float(normalized_value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="TOTAL_PENDAPATAN harus berupa angka.",
        ) from exc

    if total_pendapatan < 0:
        raise HTTPException(
            status_code=422,
            detail="TOTAL_PENDAPATAN tidak boleh negatif.",
        )

    return total_pendapatan


def kategori_tanggungan(nilai: int) -> str:
    if nilai <= 1:
        return "Rendah"
    if nilai <= 4:
        return "Sedang"
    return "Tinggi"


def kategori_riwayat(nilai: int) -> str:
    return RIWAYAT_MAPPING[nilai]


def _parse_jumlah_tanggungan(value: Any) -> int:
    if isinstance(value, bool):
        raise HTTPException(
            status_code=422,
            detail="JUMLAH TANGGUNGAN harus berupa angka, bukan boolean.",
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="JUMLAH TANGGUNGAN harus berupa angka.",
        ) from exc

    if not number.is_integer():
        raise HTTPException(
            status_code=422,
            detail="JUMLAH TANGGUNGAN harus berupa bilangan bulat.",
        )

    tanggungan = int(number)
    if tanggungan < 0:
        raise HTTPException(
            status_code=422,
            detail="JUMLAH TANGGUNGAN tidak boleh negatif.",
        )

    return tanggungan


def _parse_riwayat(value: Any) -> int:
    if isinstance(value, bool):
        raise HTTPException(
            status_code=422,
            detail="RIWAYAT harus berupa angka, bukan boolean.",
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="RIWAYAT harus berupa angka.",
        ) from exc

    if not number.is_integer():
        raise HTTPException(
            status_code=422,
            detail="RIWAYAT harus berupa bilangan bulat.",
        )

    riwayat = int(number)
    if riwayat not in RIWAYAT_MAPPING:
        raise HTTPException(
            status_code=422,
            detail="RIWAYAT harus berada pada skala 0 sampai 3.",
        )

    return riwayat


def _prepare_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise RuntimeError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    required_columns = {
        "SUAMI",
        "ISTRI",
        "JUMLAH TANGGUNGAN",
        "RIWAYAT",
        TARGET,
        *FEATURES[:6],
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise RuntimeError(f"Kolom dataset tidak lengkap: {missing}")

    df["TOTAL_PENDAPATAN"] = df["SUAMI"] + df["ISTRI"]
    df["PENDAPATAN_PER_KAPITA"] = df["TOTAL_PENDAPATAN"] / (
        df["JUMLAH TANGGUNGAN"] + 1
    )
    df["KATEGORI_TOTAL_PENDAPATAN"] = df["TOTAL_PENDAPATAN"].apply(kategori_pendapatan)
    df["KATEGORI_PENDAPATAN_PER_KAPITA"] = df["PENDAPATAN_PER_KAPITA"].apply(
        kategori_pendapatan
    )
    df["KATEGORI_TANGGUNGAN"] = df["JUMLAH TANGGUNGAN"].apply(kategori_tanggungan)
    df["RIWAYAT"] = pd.to_numeric(df["RIWAYAT"], errors="coerce")
    if df["RIWAYAT"].isna().any():
        raise RuntimeError("Kolom RIWAYAT berisi nilai kosong atau bukan angka.")
    df["RIWAYAT"] = df["RIWAYAT"].clip(0, 3).astype(int)
    df["KATEGORI_RIWAYAT"] = df["RIWAYAT"].apply(kategori_riwayat)
    return df


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _build_options(df: pd.DataFrame) -> dict[str, list[str]]:
    preferred_order = {
        "KATEGORI_TOTAL_PENDAPATAN": ["Sangat Rendah", "Rendah", "Sedang", "Tinggi"],
        "KATEGORI_PENDAPATAN_PER_KAPITA": [
            "Sangat Rendah",
            "Rendah",
            "Sedang",
            "Tinggi",
        ],
        "KATEGORI_TANGGUNGAN": ["Rendah", "Sedang", "Tinggi"],
        "KATEGORI_RIWAYAT": [
            "Belum Pernah",
            "1 Kali",
            "2 Kali",
            "3 Kali atau Lebih",
        ],
    }

    options: dict[str, list[str]] = {}
    for column in FEATURES:
        values = df[column].fillna("Tidak Diketahui").astype(str).drop_duplicates().tolist()
        if column in preferred_order:
            order = preferred_order[column]
            options[column] = [value for value in order if value in values]
        else:
            options[column] = sorted(values)
    return options


def _encode(encoder: OrdinalEncoder, frame: pd.DataFrame):
    return encoder.transform(frame.fillna("Tidak Diketahui").astype(str)).astype(int) + 1


def _train_model(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURES].fillna("Tidak Diketahui").astype(str)
    y = df[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_encoded = _encode(encoder.fit(X_train), X_train)
    X_test_encoded = _encode(encoder, X_test)

    model = CategoricalNB(alpha=1.0)
    model.fit(X_train_encoded, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_encoded))
    return {
        "model": model,
        "encoder": encoder,
        "fitur": FEATURES,
        "target": TARGET,
        "accuracy_test": float(accuracy),
        "source": "trained_from_dataset",
    }


def _load_model(df: pd.DataFrame) -> dict[str, Any]:
    if ARTIFACT_PATH.exists():
        artifact = joblib.load(ARTIFACT_PATH)
        if isinstance(artifact, dict) and artifact.get("fitur") == FEATURES:
            artifact.setdefault("accuracy_test", None)
            artifact.setdefault("source", str(ARTIFACT_PATH.relative_to(BASE_DIR)))
            return artifact

    return _train_model(df)


dataset = _prepare_dataset()
options = _build_options(dataset)
option_lookup = {
    column: {_normalize_text(value): value for value in values}
    for column, values in options.items()
}
artifact = _load_model(dataset)
model: CategoricalNB = artifact["model"]
encoder: OrdinalEncoder = artifact["encoder"]


def _get_payload_value(payload: dict[str, Any], feature: str) -> Any:
    for alias in RAW_FIELD_ALIASES[feature]:
        if alias in payload:
            return payload[alias]
    raise HTTPException(
        status_code=422,
        detail=f"Field '{feature}' wajib diisi.",
    )


def _get_required_numeric_payload(payload: dict[str, Any], aliases: list[str], label: str) -> Any:
    for alias in aliases:
        if alias in payload:
            value = payload[alias]
            if value is None or _normalize_text(value) == "":
                raise HTTPException(
                    status_code=422,
                    detail=f"{label} tidak boleh kosong.",
                )
            return value

    raise HTTPException(status_code=422, detail=f"Field '{label}' wajib diisi.")


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=422, detail="Payload JSON tidak boleh kosong.")

    total_pendapatan = _parse_total_pendapatan(
        _get_required_numeric_payload(
            payload,
            TOTAL_PENDAPATAN_ALIASES,
            "TOTAL_PENDAPATAN",
        )
    )
    jumlah_tanggungan = _parse_jumlah_tanggungan(
        _get_required_numeric_payload(
            payload,
            TANGGUNGAN_NUMBER_ALIASES,
            "JUMLAH TANGGUNGAN",
        )
    )
    riwayat = _parse_riwayat(
        _get_required_numeric_payload(
            payload,
            RIWAYAT_ALIASES,
            "RIWAYAT",
        )
    )
    pendapatan_per_kapita = total_pendapatan / (jumlah_tanggungan + 1)

    cleaned: dict[str, str] = {}
    cleaned["KATEGORI_TOTAL_PENDAPATAN"] = kategori_pendapatan(total_pendapatan)
    cleaned["KATEGORI_PENDAPATAN_PER_KAPITA"] = kategori_pendapatan(
        pendapatan_per_kapita
    )
    cleaned["KATEGORI_TANGGUNGAN"] = kategori_tanggungan(jumlah_tanggungan)
    cleaned["KATEGORI_RIWAYAT"] = kategori_riwayat(riwayat)

    for feature in RAW_CATEGORICAL_FEATURES:
        value = _get_payload_value(payload, feature)
        if value is None or _normalize_text(value) == "":
            raise HTTPException(
                status_code=422,
                detail=f"Field '{feature}' tidak boleh kosong.",
            )

        normalized = option_lookup[feature].get(_normalize_text(value))
        if normalized is None:
            allowed = ", ".join(options[feature])
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Nilai '{value}' untuk '{feature}' tidak dikenal. "
                    f"Pilihan yang tersedia: {allowed}"
                ),
            )

        cleaned[feature] = normalized

    return {
        "features": cleaned,
        "derived_values": {
            "TOTAL_PENDAPATAN": total_pendapatan,
            "JUMLAH TANGGUNGAN": jumlah_tanggungan,
            "RIWAYAT": riwayat,
            "PENDAPATAN_PER_KAPITA": pendapatan_per_kapita,
        },
    }


def _predict_one(input_features: dict[str, str]) -> dict[str, Any]:
    frame = pd.DataFrame([{feature: input_features[feature] for feature in FEATURES}])
    encoded = _encode(encoder, frame)
    prediction = str(model.predict(encoded)[0])

    probabilities = model.predict_proba(encoded)[0]
    probability_by_class = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    return {
        "prediksi": prediction,
        "kelayakan": prediction,
        "confidence": float(max(probabilities)),
        "probabilitas": probability_by_class,
    }


@app.get("/")
def home():
    return {
        "status": "API aktif",
        "endpoint": "POST /predict",
        "fitur": FEATURES,
        "target": TARGET,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "fitur": FEATURES,
        "input_payload": INPUT_PAYLOAD_FIELDS,
        "target": TARGET,
        "pilihan": options,
        "feature_engineering": {
            "TOTAL_PENDAPATAN": "Dikirim sebagai angka dari frontend/API.",
            "PENDAPATAN_PER_KAPITA": (
                "Dihitung otomatis: TOTAL_PENDAPATAN / (JUMLAH TANGGUNGAN + 1)"
            ),
            "KATEGORI_TOTAL_PENDAPATAN": (
                "Dibuat otomatis dari TOTAL_PENDAPATAN: Sangat Rendah jika <= 850000, "
                "Rendah jika <= 1700000, Sedang jika <= 3400000, Tinggi jika > 3400000"
            ),
            "KATEGORI_PENDAPATAN_PER_KAPITA": (
                "Dibuat otomatis dari PENDAPATAN_PER_KAPITA dengan skala kategori pendapatan yang sama"
            ),
            "KATEGORI_TANGGUNGAN": (
                "Dibuat otomatis dari JUMLAH TANGGUNGAN: "
                "Rendah jika <= 1, Sedang jika <= 4, Tinggi jika > 4"
            ),
            "RIWAYAT": "Dikirim sebagai angka skala 0 sampai 3 dari frontend/API.",
            "KATEGORI_RIWAYAT": (
                "Dibuat otomatis dari RIWAYAT: 0 Belum Pernah, 1 1 Kali, "
                "2 2 Kali, 3 3 Kali atau Lebih"
            ),
        },
        "model": {
            "nama": "Categorical Naive Bayes",
            "source": artifact.get("source"),
            "accuracy_test": artifact.get("accuracy_test"),
        },
    }


@app.post("/predict")
def predict(payload: dict[str, Any] = Body(...)):
    try:
        validated = _validate_payload(payload)
        input_features = validated["features"]
        result = _predict_one(input_features)
        return {
            "status": "success",
            "target": TARGET,
            "fitur_digunakan": FEATURES,
            "input": input_features,
            "nilai_hasil_feature_engineering": validated["derived_values"],
            "feature_engineering": {
                "KATEGORI_TOTAL_PENDAPATAN": (
                    "Dihitung otomatis dari TOTAL_PENDAPATAN berdasarkan skala pendapatan"
                ),
                "KATEGORI_PENDAPATAN_PER_KAPITA": (
                    "Dihitung otomatis dari TOTAL_PENDAPATAN / (JUMLAH TANGGUNGAN + 1)"
                ),
                "KATEGORI_TANGGUNGAN": (
                    "Rendah jika JUMLAH TANGGUNGAN <= 1, "
                    "Sedang jika <= 4, Tinggi jika > 4"
                ),
                "KATEGORI_RIWAYAT": (
                    "0 Belum Pernah, 1 1 Kali, 2 2 Kali, 3 3 Kali atau Lebih"
                ),
            },
            "model": {
                "nama": "Categorical Naive Bayes",
                "source": artifact.get("source"),
                "accuracy_test": artifact.get("accuracy_test"),
            },
            **result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
