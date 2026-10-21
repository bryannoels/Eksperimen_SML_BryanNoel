"""
modelling.py
============
Baseline klasifikasi kelas kompleksitas waktu kode sumber (7 kelas) dengan MLflow autolog.

- Model    : RandomForestClassifier (scikit-learn), tanpa hyperparameter tuning
- Data     : codecomplex_preprocessing/train.csv dan test.csv (hasil preprocessing, bukan data mentah)
- Tracking : MLflow server lokal (http://127.0.0.1:5001), backend SQLite
- Logging  : mlflow.autolog() untuk parameter dan metrik pelatihan; model dan evaluasi data uji dicatat tambahan

Cara menjalankan (dari folder Membangun_model/, dua terminal):
    Terminal 1:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    Terminal 2:  python modelling.py
"""

import tempfile
import time
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "codecomplex_preprocessing"

TRACKING_URI = "http://127.0.0.1:5001/"
EXPERIMENT_NAME = "BigOracle_Baseline"
RUN_NAME = "random_forest_baseline"

N_ESTIMATORS = 200
MAX_DEPTH = None
SEED = 42

CLASS_NAMES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "NP-hard/eksponensial"]
CLASS_SHORT = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "NP/exp"]
TOTAL_STEPS = 7
T_START = time.time()


def step(n, text):
    print(f"\n[{n}/{TOTAL_STEPS}] {text}  (+{time.time() - T_START:.1f}s)")
    print("-" * 70)


def ok(text):
    print(f"  [OK] {text}")


# ---------------------------------------------------------------------------
print("=" * 70)
print(" BigOracle | Baseline RandomForest + MLflow autolog")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Cek koneksi ke MLflow server
# ---------------------------------------------------------------------------
step(1, "Memeriksa koneksi ke MLflow server")
try:
    with urllib.request.urlopen(TRACKING_URI.rstrip("/") + "/health", timeout=3) as resp:
        ok(f"Server aktif di {TRACKING_URI} (status: {resp.read().decode().strip()})")
except Exception:
    raise SystemExit(
        f"\n[GAGAL] MLflow server tidak ditemukan di {TRACKING_URI}\n"
        "Jalankan dulu di terminal lain (dari folder Membangun_model/):\n"
        "    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001\n"
    )

# ---------------------------------------------------------------------------
# 2. Memuat data
# ---------------------------------------------------------------------------
step(2, "Memuat dataset hasil preprocessing")
if not (DATA_DIR / "train.csv").exists() or not (DATA_DIR / "test.csv").exists():
    raise SystemExit(f"\n[GAGAL] train.csv / test.csv tidak ditemukan di {DATA_DIR}\n"
                     "Salin folder codecomplex_preprocessing/ ke dalam Membangun_model/.\n")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]
input_example = X_train[0:5]

ok(f"Data latih : {X_train.shape[0]:,} baris x {X_train.shape[1]} fitur")
ok(f"Data uji   : {X_test.shape[0]:,} baris x {X_test.shape[1]} fitur")
ok(f"Nilai kosong: latih={int(X_train.isna().sum().sum())}, uji={int(X_test.isna().sum().sum())}")
dist = pd.DataFrame({
    "kelas": CLASS_NAMES,
    "latih": y_train.value_counts().reindex(range(len(CLASS_NAMES))).fillna(0).astype(int).values,
    "uji": y_test.value_counts().reindex(range(len(CLASS_NAMES))).fillna(0).astype(int).values,
})
print("\n" + dist.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Menyiapkan MLflow
# ---------------------------------------------------------------------------
step(3, "Menyiapkan MLflow (tracking, experiment, autolog)")
mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(log_models=False)   # autolog scikit-learn; model disimpan manual (langkah 5) agar muncul di tab Artifacts
ok(f"Experiment: {EXPERIMENT_NAME} (id={experiment.experiment_id})")
ok("Autolog aktif (parameter + metrik pelatihan + confusion matrix pelatihan)")

with mlflow.start_run(run_name=RUN_NAME) as run:
    ok(f"Run dimulai: {run.info.run_id}")

    # -----------------------------------------------------------------------
    # 4. Pelatihan
    # -----------------------------------------------------------------------
    step(4, "Melatih RandomForestClassifier")
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=SEED, n_jobs=-1)
    print(f"  Parameter: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, random_state={SEED}")
    print("  Melatih model ... (autolog mencatat parameter dan metrik secara otomatis)")
    t0 = time.time()
    model.fit(X_train, y_train)
    ok(f"Pelatihan selesai dalam {time.time() - t0:.1f} detik")
    train_acc = accuracy_score(y_train, model.predict(X_train))
    ok(f"Akurasi data latih: {train_acc:.4f}")

    # -----------------------------------------------------------------------
    # 5. Menyimpan model sebagai artefak run (folder model/)
    # -----------------------------------------------------------------------
    step(5, "Menyimpan model sebagai artefak MLflow")
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        mlflow.sklearn.save_model(
            model, str(model_dir),
            signature=infer_signature(input_example, model.predict(input_example)),
            input_example=input_example,
            serialization_format="cloudpickle",   # MLflow 3.x: format skops bawaan menolak model pohon
        )
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
    ok("Model tersimpan di artefak: model/ (MLmodel, model.pkl, conda.yaml, requirements.txt)")

    # -----------------------------------------------------------------------
    # 6. Evaluasi pada data uji
    # -----------------------------------------------------------------------
    step(6, "Evaluasi pada data uji (soal yang belum pernah dilihat model)")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metrics({
        "test_accuracy_score": acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "train_test_accuracy_gap": train_acc - acc,
    })
    ok(f"Akurasi    : {acc:.4f}")
    ok(f"Macro-F1   : {macro_f1:.4f}")
    ok(f"Weighted-F1: {weighted_f1:.4f}")
    ok(f"Selisih akurasi latih - uji: {train_acc - acc:.4f} (indikasi overfitting)")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
    print("\n" + report)
    mlflow.log_text(report, "classification_report.txt")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=CLASS_SHORT, normalize="true",
                                            values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (data uji, dinormalisasi per kelas)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix_test.png")
    plt.close(fig)
    ok("Artefak tambahan tersimpan: classification_report.txt, confusion_matrix_test.png")

    run_id = run.info.run_id

# ---------------------------------------------------------------------------
# 7. Ringkasan
# ---------------------------------------------------------------------------
step(7, "Ringkasan")
run_url = f"{TRACKING_URI.rstrip('/')}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
print(f"  Run ID     : {run_id}")
print(f"  Eksperimen : {EXPERIMENT_NAME}")
print(f"  Akurasi uji: {acc:.4f} | Macro-F1 uji: {macro_f1:.4f}")
print(f"  Total waktu: {time.time() - T_START:.1f} detik")
print(f"\n  Buka run ini di MLflow UI:\n  {run_url}")
print("\n  Untuk screenshot: halaman run (parameter + metrik) dan tab Artifacts (folder model/).")
print("  Pastikan address bar browser (127.0.0.1:5001) ikut terlihat pada tangkapan layar.")
print("=" * 70)
