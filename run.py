from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "model_bundle.pkl"


def ensure_model_bundle() -> None:
    """
    Pastikan artifact model sudah tersedia.
    Jika belum ada, jalankan main.py untuk melatih model dan mengekspor bundle.
    """
    if ARTIFACT_PATH.exists():
        return


    print("Model bundle belum ditemukan, menjalankan `main.py` terlebih dahulu...")
    subprocess.run([sys.executable, "main.py"], cwd=BASE_DIR, check=True)


def main() -> None:
    ensure_model_bundle()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn belum terpasang. Jalankan `pip install -r requirements.txt` terlebih dahulu."
        ) from exc

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
