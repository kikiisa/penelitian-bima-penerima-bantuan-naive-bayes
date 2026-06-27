from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "model_bundle.pkl"


app = FastAPI(
    title="Mustahik Classification API",
    description="Naive Bayes & C4.5 Classification",
    version="2.0",
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


def _load_bundle() -> dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Model bundle not found: {ARTIFACT_PATH}. "
            "Jalankan `python main.py` terlebih dahulu untuk mengekspor model."
        )
    bundle = joblib.load(ARTIFACT_PATH)
    required_keys = {
        "model_nb",
        "model_dt",
        "scaler",
        "label_encoder_y",
        "feature_encoders",
        "feature_names",
    }
    missing = required_keys.difference(bundle.keys())
    if missing:
        raise RuntimeError(f"Model bundle tidak lengkap. Missing keys: {sorted(missing)}")
    return bundle


bundle = _load_bundle()
nb_model = bundle["model_nb"]
dt_model = bundle["model_dt"]
scaler = bundle["scaler"]
le_y = bundle["label_encoder_y"]
feature_encoders = bundle["feature_encoders"]
feature_names = bundle["feature_names"]
metrics = bundle.get("metrics", {})
comparison_summary = bundle.get("comparison_summary", {})
selection_metric = metrics.get("selection_metric", "cv_mean")


def _model_score(key: str) -> float:
    model_metrics = metrics.get(key, {})
    value = model_metrics.get(selection_metric)
    if value is None:
        value = model_metrics.get("accuracy_test", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


BEST_ALGORITHM_KEY = metrics.get("best_algorithm")
if BEST_ALGORITHM_KEY not in {"naive_bayes", "decision_tree"}:
    BEST_ALGORITHM_KEY = max(
        ("naive_bayes", "decision_tree"),
        key=_model_score,
    )

ALGORITHM_LABELS = {
    "naive_bayes": "Naive Bayes",
    "decision_tree": "Decision Tree C4.5",
}


def _build_comparison_summary() -> dict[str, Any]:
    summary = comparison_summary if isinstance(comparison_summary, dict) else {}
    if summary.get("rows"):
        return summary

    rows = []
    for key, label in [
        ("naive_bayes", "Naive Bayes"),
        ("decision_tree", "Decision Tree C4.5"),
    ]:
        model_metrics = metrics.get(key, {})
        selection_score = float(model_metrics.get(selection_metric, model_metrics.get("accuracy_test", 0.0)))
        rows.append(
            {
                "key": key,
                "nama_algoritma": label,
                "akurasi_test": float(model_metrics.get("accuracy_test", 0.0)),
                "cv_mean": float(model_metrics.get("cv_mean", 0.0)),
                "cv_std": float(model_metrics.get("cv_std", 0.0)),
                "selection_score": selection_score,
            }
        )

    rows.sort(key=lambda item: item["selection_score"], reverse=True)
    winner = rows[0] if rows else {}
    runner_up = rows[1] if len(rows) > 1 else {}
    score_gap = float(winner.get("selection_score", 0.0) - runner_up.get("selection_score", 0.0)) if runner_up else 0.0

    selection_label = "Cross-Validation Mean" if selection_metric == "cv_mean" else selection_metric.replace("_", " ").title()
    summary_text = (
        f"{winner['nama_algoritma']} unggul berdasarkan {selection_label} dengan selisih {score_gap:.4f}."
        if winner
        else "Ringkasan perbandingan belum tersedia."
    )

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


class MustahikRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nama: str | None = None
    no_kk: str | None = None

    total_pendapatan: float | None = Field(
        default=None,
        validation_alias=AliasChoices("total_pendapatan", "TOTAL_PENDAPATAN", "total_penghasilan"),
    )
    suami: float | None = None
    istri: float | None = None

    jumlah_tanggungan: int = 0
    jenis_usaha_pekerjaan: str
    kepemilikan_aset: Any = "tidak ada"
    luas_tempat_tinggal: str
    jenis_dinding: str
    jenis_lantai: str


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _normalize_to_encoder_value(column: str, value: Any) -> str:
    encoder = feature_encoders[column]
    lookup = {_normalize_text(label): label for label in encoder.classes_}
    normalized = lookup.get(_normalize_text(value))
    if normalized is not None:
        return normalized

    # Toleransi untuk input lama atau typo ringan.
    raw = _normalize_text(value)
    if column == "JENIS USAHA/PEKERJAAN":
        if any(token in raw for token in ["dagang", "wiraswasta", "usaha"]):
            return "perdagangan"
        if any(token in raw for token in ["petani", "nelayan", "tani"]):
            return "pertanian"
        if any(token in raw for token in ["tukang", "transportasi", "buruh", "bekerja"]):
            return "jasa"
        if "karyawan" in raw:
            return "karyawan"
        if any(token in raw for token in ["pns", "pegawai negeri"]):
            return "pns"
        if any(token in raw for token in ["irt", "urt", "ibu rumah"]):
            return "irt"
        return "lainnya"

    if column == "LUAS TEMPAT TINGGAL":
        if "sangat kecil" in raw:
            return "sangat kecil"
        if "kecil" in raw:
            return "kecil"
        if "besar" in raw:
            return "besar"
        return "sedang"

    if column == "JENIS DINDING":
        if any(token in raw for token in ["tembok", "beton"]):
            return "tembok"
        if "semi" in raw:
            return "semi"
        return "bilik"

    if column == "JENIS LANTAI":
        if any(token in raw for token in ["keramik", "kermaik"]):
            return "keramik"
        if "panggung" in raw:
            return "panggung"
        if "tanah" in raw:
            return "tanah"
        return "semen"

    allowed = ", ".join(map(str, encoder.classes_))
    raise HTTPException(
        status_code=422,
        detail=f"Nilai '{value}' untuk '{column}' tidak dikenali. Pilihan yang tersedia: {allowed}",
    )


def _as_asset_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 0 if int(value) == 0 else 1
    text = _normalize_text(value)
    if text in {"0", "tidak ada", "tidak", "false", "no"}:
        return 0
    return 1


def _build_input_frame(data: MustahikRequest) -> pd.DataFrame:
    total_pendapatan = data.total_pendapatan
    if total_pendapatan is None:
        if data.suami is None and data.istri is None:
            raise HTTPException(
                status_code=422,
                detail="total_pendapatan wajib diisi, atau isi suami dan istri untuk dihitung otomatis.",
            )
        total_pendapatan = float(data.suami or 0) + float(data.istri or 0)

    row = [
        float(total_pendapatan),
        int(data.jumlah_tanggungan),
        _normalize_to_encoder_value("JENIS USAHA/PEKERJAAN", data.jenis_usaha_pekerjaan),
        _as_asset_flag(data.kepemilikan_aset),
        _normalize_to_encoder_value("LUAS TEMPAT TINGGAL", data.luas_tempat_tinggal),
        _normalize_to_encoder_value("JENIS DINDING", data.jenis_dinding),
        _normalize_to_encoder_value("JENIS LANTAI", data.jenis_lantai),
    ]

    frame = pd.DataFrame([row], columns=feature_names)

    for column, encoder in feature_encoders.items():
        frame[column] = encoder.transform(frame[column].astype(str))

    return frame


def _kelayakan(status: str) -> str:
    return "Layak" if status in {"Fakir", "Miskin"} else "Tidak Layak"


def _predict_with_algorithm(algorithm_key: str, input_array):
    if algorithm_key == "naive_bayes":
        scaled_input = scaler.transform(input_array)
        raw_proba = nb_model.predict_proba(scaled_input)[0]
        pred = le_y.inverse_transform(nb_model.predict(scaled_input))[0]
    else:
        raw_proba = dt_model.predict_proba(input_array)[0]
        pred = le_y.inverse_transform(dt_model.predict(input_array))[0]

    class_probabilities = {
        str(label): float(prob)
        for label, prob in zip(le_y.classes_, raw_proba)
    }

    return {
        "nama_algoritma": ALGORITHM_LABELS[algorithm_key],
        "status": str(pred),
        "kelayakan": _kelayakan(str(pred)),
        "confidence": float(max(raw_proba)),
        "probabilitas": class_probabilities,
        "akurasi_model": _model_score(algorithm_key),
        "metode_seleksi": selection_metric,
    }


@app.get("/")
def home():
    return {
        "status": "API aktif",
        "service": "Klasifikasi Mustahik",
        "feature_names": feature_names,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/comparison")
def comparison():
    return {
        "status": "ok",
        "service": "Perbandingan Algoritma",
        "data": _build_comparison_summary(),
    }


@app.post("/predict")
def predict(data: MustahikRequest):
    try:
        input_data = _build_input_frame(data)
        input_array = input_data.to_numpy(dtype=float)

        algoritma_terbaik = _predict_with_algorithm(BEST_ALGORITHM_KEY, input_array)

        return {
            "nama": data.nama,
            "no_kk": data.no_kk,
            "total_pendapatan": float(input_data["TOTAL_PENDAPATAN"].iloc[0]),
            "algoritma_terbaik": algoritma_terbaik,
            "perbandingan_akurasi": {
                "naive_bayes": {
                    "akurasi_test": _model_score("naive_bayes"),
                    "cv_mean": float(metrics.get("naive_bayes", {}).get("cv_mean", 0.0)),
                    "cv_std": float(metrics.get("naive_bayes", {}).get("cv_std", 0.0)),
                },
                "decision_tree": {
                    "akurasi_test": _model_score("decision_tree"),
                    "cv_mean": float(metrics.get("decision_tree", {}).get("cv_mean", 0.0)),
                    "cv_std": float(metrics.get("decision_tree", {}).get("cv_std", 0.0)),
                },
                "dipilih": ALGORITHM_LABELS[BEST_ALGORITHM_KEY],
            },
            "analisis_perbandingan": _build_comparison_summary(),
            "kesimpulan": algoritma_terbaik["kelayakan"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn belum terpasang. Jalankan `pip install -r requirements.txt` "
            "atau `pip install uvicorn`, lalu coba lagi."
        ) from exc

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
