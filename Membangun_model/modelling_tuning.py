"""
modelling_tuning.py
===================
Hyperparameter tuning (grid search) RandomForest untuk klasifikasi kelas kompleksitas waktu (7 kelas),
dengan MANUAL LOGGING MLflow (tanpa autolog). Satu kali menjalankan skrip ini mengerjakan Skilled DAN Advance:

    1. Grid search dihitung SATU kali (cross-validation per kelompok soal, dipilih dengan macro-F1 CV).
    2. Hasil dicatat ke MLflow Tracking UI lokal  (http://127.0.0.1:5001)   -> Skilled
    3. Hasil yang sama dicatat ke DagsHub (online)                           -> Advance
       (jika DagsHub gagal, catatan lokal tetap tersimpan)

Struktur artefak pada run induk:
    - Seperti autolog : estimator.html, metric_info.json, training_confusion_matrix.png, model/
    - Tambahan        : confusion_matrix_test.png, feature_importance.png, feature_importance.csv,
                        classification_report.json, grid_results.csv, best_params.json

Metrik memakai nama yang sama dengan autolog: training_accuracy_score, training_precision_score,
training_recall_score, training_f1_score, training_log_loss, training_roc_auc, training_score.

Cara menjalankan (dari folder Membangun_model/):
    Terminal 1:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    Terminal 2:  python modelling_tuning.py                      # lokal + DagsHub
                 python modelling_tuning.py --targets local      # hanya lokal
                 python modelling_tuning.py --targets dagshub    # hanya DagsHub
"""

import argparse
import itertools
import json
import os
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # mencegah error GUI/Tkinter
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score

try:
    from sklearn.utils import estimator_html_repr
except ImportError:  # versi scikit-learn yang tidak lagi menyediakannya
    def estimator_html_repr(est):
        return est._repr_html_()

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "codecomplex_preprocessing"

LOCAL_TRACKING_URI = "http://127.0.0.1:5001/"
DAGSHUB_OWNER = "bryannoels"
DAGSHUB_REPO = "BigOracle-Eksperimen"
EXPERIMENT_NAME = "BigOracle_Tuning"

CLASS_NAMES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "NP-hard/eksponensial"]
CLASS_SHORT = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "NP/exp"]
TOTAL_STEPS = 7
T_START = time.time()


def step(n, text):
    print(f"\n[{n}/{TOTAL_STEPS}] {text}  (+{time.time() - T_START:.0f}s)")
    print("-" * 78)


def ok(text):
    print(f"  [OK] {text}")


def parse_grid(text):
    """'100,300' -> [100, 300]; '10,20,None' -> [10, 20, None]"""
    return [None if t.strip().lower() == "none" else int(t) for t in text.split(",")]


# ---------------------------------------------------------------------------
# Tracking: lokal (server) dan DagsHub
# ---------------------------------------------------------------------------
def check_local_server():
    try:
        with urllib.request.urlopen(LOCAL_TRACKING_URI.rstrip("/") + "/health", timeout=3) as resp:
            ok(f"Server MLflow lokal aktif di {LOCAL_TRACKING_URI} (status: {resp.read().decode().strip()})")
    except Exception:
        raise SystemExit(
            f"\n[GAGAL] MLflow server lokal tidak ditemukan di {LOCAL_TRACKING_URI}\n"
            "Jalankan dulu di terminal lain (dari folder Membangun_model/):\n"
            "    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001\n"
            "Atau jalankan hanya DagsHub dengan:  python modelling_tuning.py --targets dagshub\n"
        )


def setup_local():
    mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
    return LOCAL_TRACKING_URI


def setup_dagshub(owner, repo):
    """Arahkan MLflow ke DagsHub. Memakai DAGSHUB_USER_TOKEN bila ada, atau login lewat browser."""
    uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    token = os.getenv("DAGSHUB_USER_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        ok("Autentikasi DagsHub memakai DAGSHUB_USER_TOKEN")
    else:
        import dagshub
        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        ok("dagshub.init berhasil (login lewat browser bila diminta)")
    mlflow.set_tracking_uri(uri)   # pastikan URI menunjuk ke DagsHub
    return uri


# ---------------------------------------------------------------------------
# Metrik (nama sama dengan yang dicatat autolog untuk model scikit-learn)
# ---------------------------------------------------------------------------
def classification_metrics(model, X, y, prefix):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        f"{prefix}_accuracy_score": accuracy_score(y, pred),
        f"{prefix}_precision_score": precision_score(y, pred, average="weighted", zero_division=0),
        f"{prefix}_recall_score": recall_score(y, pred, average="weighted", zero_division=0),
        f"{prefix}_f1_score": f1_score(y, pred, average="weighted", zero_division=0),
        f"{prefix}_log_loss": log_loss(y, proba, labels=model.classes_),
        f"{prefix}_roc_auc": roc_auc_score(y, proba, multi_class="ovo", average="weighted", labels=model.classes_),
        f"{prefix}_score": model.score(X, y),
    }, pred


# ---------------------------------------------------------------------------
# Artefak (struktur autolog + tambahan)
# ---------------------------------------------------------------------------
def build_artifacts(best, X_train, y_train, X_test, y_test, candidates, art_dir):
    """Membuat semua berkas artefak SATU kali; dipakai ulang untuk setiap tujuan tracking."""
    art_dir = Path(art_dir)
    model, y_pred_test, cand = best["model"], best["y_pred"], best["cand"]

    # ---- struktur sama dengan autolog ----
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ConfusionMatrixDisplay.from_predictions(y_train, model.predict(X_train), display_labels=CLASS_SHORT,
                                            values_format="d", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Training Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(art_dir / "training_confusion_matrix.png", dpi=150)
    plt.close(fig)

    (art_dir / "estimator.html").write_text(estimator_html_repr(model), encoding="utf-8")

    metric_info = {k: f"{k.split('_', 1)[0]} set: {k.split('_', 1)[1]} dihitung dengan scikit-learn pada model terbaik"
                   for k in list(cand["train_m"]) + list(cand["test_m"])}
    (art_dir / "metric_info.json").write_text(json.dumps(metric_info, indent=2), encoding="utf-8")

    # ---- artefak tambahan di luar autolog ----
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, display_labels=CLASS_SHORT, normalize="true",
                                            values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Test Confusion Matrix (dinormalisasi per kelas)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(art_dir / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    imp = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(art_dir / "feature_importance.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#3b6ea5")
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("15 Fitur Terpenting (model terbaik)")
    fig.tight_layout()
    fig.savefig(art_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    report = classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    (art_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    grid_df = pd.DataFrame([{k: v for k, v in c.items() if k in
                             ("name", "n_estimators", "max_depth", "min_samples_leaf", "cv_mean", "cv_std",
                              "test_macro_f1")} for c in candidates])
    grid_df.sort_values("cv_mean", ascending=False).to_csv(art_dir / "grid_results.csv", index=False)
    (art_dir / "best_params.json").write_text(json.dumps(
        {k: cand[k] for k in ["n_estimators", "max_depth", "min_samples_leaf"]}, indent=2), encoding="utf-8")

    # ---- model: MLmodel, model.pkl, conda.yaml, python_env.yaml, requirements.txt ----
    sample = X_train[0:5]
    mlflow.sklearn.save_model(model, str(art_dir / "model"),
                              signature=infer_signature(sample, model.predict(sample)), input_example=sample,
                              serialization_format="cloudpickle")  # MLflow 3.x: skops bawaan menolak model pohon

    files = ["estimator.html", "metric_info.json", "training_confusion_matrix.png",
             "confusion_matrix_test.png", "feature_importance.png", "feature_importance.csv",
             "classification_report.json", "grid_results.csv", "best_params.json"]
    return files, imp


# ---------------------------------------------------------------------------
# Pencatatan ke tracking yang sedang aktif (lokal ATAU DagsHub)
# ---------------------------------------------------------------------------
def log_to_tracking(label, candidates, best, art_dir, files, cfg):
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    cand = best["cand"]
    with mlflow.start_run(run_name=f"rf_grid_search_{label}") as parent:
        mlflow.set_tags({"stage": "tuning", "dataset": "codecomplex", "split": "grouped_by_problem",
                         "tracking": label, "logging": "manual"})
        mlflow.log_params({"search_method": "GridSearch (manual loop)", "n_combinations": len(candidates),
                           "cv": cfg["cv_name"], "cv_scoring": "f1_macro", "seed": cfg["seed"],
                           "n_train": cfg["n_train"], "n_test": cfg["n_test"], "n_features": cfg["n_features"]})

        # run anak: satu per kombinasi hyperparameter
        for i, c in enumerate(candidates, start=1):
            with mlflow.start_run(run_name=c["name"], nested=True):
                mlflow.log_params(c["params"])
                mlflow.log_metrics({**c["train_m"], **c["test_m"],
                                    "cv_macro_f1_mean": c["cv_mean"], "cv_macro_f1_std": c["cv_std"],
                                    "test_macro_f1": c["test_macro_f1"], "fit_time_seconds": c["fit_time"]})
                mlflow.set_tag("selection_metric", "cv_macro_f1_mean")
            print(f"    [{i:>2}/{len(candidates)}] run anak tercatat: {c['name']}")

        # run induk: model terbaik
        mlflow.log_params(cand["params"])
        mlflow.log_params({"best_" + k: str(cand[k]) for k in ["n_estimators", "max_depth", "min_samples_leaf"]})
        mlflow.set_tag("best_child_run", cand["name"])
        per_class = f1_score(cfg["y_test"], best["y_pred"], average=None,
                             labels=list(range(len(CLASS_NAMES))), zero_division=0)
        mlflow.log_metrics({**cand["train_m"], **cand["test_m"],
                            "test_macro_f1": cand["test_macro_f1"],
                            "best_cv_macro_f1": cand["cv_mean"], "best_cv_macro_f1_std": cand["cv_std"],
                            "tuning_time_seconds": cfg["tuning_time"],
                            **{f"test_f1_class_{i}": float(v) for i, v in enumerate(per_class)}})
        for name in files:
            mlflow.log_artifact(str(Path(art_dir) / name))
        mlflow.log_artifacts(str(Path(art_dir) / "model"), artifact_path="model")
    return parent.info.run_id, experiment.experiment_id


# ---------------------------------------------------------------------------
# Utama
# ---------------------------------------------------------------------------
def main(args):
    targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    if not targets or not targets <= {"local", "dagshub"}:
        raise SystemExit("--targets harus berisi 'local', 'dagshub', atau 'local,dagshub'")

    print("=" * 78)
    print(f" BigOracle | Grid search RandomForest + MLflow manual logging  [{', '.join(sorted(targets))}]")
    print("=" * 78)
    mlflow.autolog(disable=True)            # manual logging: autolog dimatikan
    mlflow.sklearn.autolog(disable=True)

    # 1. Data
    step(1, "Memuat dataset hasil preprocessing")
    if "local" in targets:
        check_local_server()
    if not (DATA_DIR / "train.csv").exists() or not (DATA_DIR / "test.csv").exists():
        raise SystemExit(f"\n[GAGAL] train.csv / test.csv tidak ditemukan di {DATA_DIR}\n"
                         "Salin folder codecomplex_preprocessing/ ke dalam Membangun_model/.\n")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]
    ok(f"Data latih: {X_train.shape[0]:,} baris x {X_train.shape[1]} fitur | Data uji: {X_test.shape[0]:,} baris")

    groups_path = DATA_DIR / "train_groups.csv"
    groups = pd.read_csv(groups_path)["problem"].astype(str) if groups_path.exists() else None
    if groups is not None and len(groups) != len(X_train):
        raise SystemExit("\n[GAGAL] train_groups.csv tidak selaras dengan train.csv; jalankan ulang preprocessing.\n")
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_name = f"StratifiedGroupKFold({args.cv_folds}) per soal"
        ok(f"Cross-validation: {cv_name} ({groups.nunique()} soal pada data latih)")
    else:
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_name = f"StratifiedKFold({args.cv_folds})"
        print("  [PERINGATAN] train_groups.csv tidak ditemukan. CV tidak dikelompokkan per soal sehingga skor CV "
              "cenderung terlalu optimistis.\n  Jalankan ulang preprocessing terbaru, lalu salin train_groups.csv "
              "ke codecomplex_preprocessing/.")

    # 2. Grid
    step(2, "Mendefinisikan ruang pencarian hyperparameter")
    grid = list(itertools.product(args.n_estimators, args.max_depth, args.min_samples_leaf))
    ok(f"n_estimators     : {args.n_estimators}")
    ok(f"max_depth        : {args.max_depth}")
    ok(f"min_samples_leaf : {args.min_samples_leaf}")
    ok(f"Total kombinasi  : {len(grid)} (masing-masing {args.cv_folds}-fold CV + 1 pelatihan penuh)")

    # 3. Grid search (komputasi murni, dikerjakan satu kali)
    step(3, "Menjalankan grid search (dipilih berdasarkan CV macro-F1 pada data latih)")
    candidates, best = [], None
    t_grid = time.time()
    for i, (n_est, depth, leaf) in enumerate(grid, start=1):
        t0 = time.time()
        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                                       random_state=args.seed, n_jobs=-1)
        cv_scores = cross_val_score(model, X_train, y_train, groups=groups, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        train_m, _ = classification_metrics(model, X_train, y_train, "training")
        test_m, y_pred = classification_metrics(model, X_test, y_test, "test")

        cand = {"name": f"rf_n{n_est}_d{depth}_leaf{leaf}", "n_estimators": n_est, "max_depth": depth,
                "min_samples_leaf": leaf, "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std()),
                "train_m": train_m, "test_m": test_m, "fit_time": time.time() - t0,
                "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "params": {k: str(v) for k, v in model.get_params().items()}}
        candidates.append(cand)
        is_best = best is None or cand["cv_mean"] > best["cand"]["cv_mean"]
        if is_best:
            best = {"model": model, "y_pred": y_pred, "cand": cand}

        eta = (time.time() - t_grid) / i * (len(grid) - i)
        print(f"  [{i:>2}/{len(grid)}] n_est={n_est:<4} depth={str(depth):<4} leaf={leaf} | "
              f"CV macro-F1={cand['cv_mean']:.4f} +/- {cand['cv_std']:.4f} | {time.time() - t0:>4.0f}s | "
              f"ETA ~{eta:.0f}s{'  <- terbaik sementara' if is_best else ''}")
    tuning_time = time.time() - t_grid
    bc = best["cand"]
    ok(f"Terbaik: n_estimators={bc['n_estimators']}, max_depth={bc['max_depth']}, "
       f"min_samples_leaf={bc['min_samples_leaf']} | CV macro-F1={bc['cv_mean']:.4f}")

    cfg = {"cv_name": cv_name, "seed": args.seed, "n_train": len(X_train), "n_test": len(X_test),
           "n_features": X_train.shape[1], "tuning_time": tuning_time, "y_test": y_test}
    status, links = {}, {}

    with tempfile.TemporaryDirectory() as tmp:
        # 4. Artefak
        step(4, "Menyiapkan artefak (struktur autolog + artefak tambahan) dan model")
        files, imp = build_artifacts(best, X_train, y_train, X_test, y_test, candidates, tmp)
        for f in files:
            ok(f"Artefak: {f}")
        ok("Model terbaik: model/ (MLmodel, model.pkl, conda.yaml, python_env.yaml, requirements.txt)")

        # 5. Pencatatan lokal (Skilled)
        step(5, "Mencatat ke MLflow Tracking UI lokal (Skilled)")
        if "local" in targets:
            try:
                setup_local()
                run_id, exp_id = log_to_tracking("local", candidates, best, tmp, files, cfg)
                status["local"] = "BERHASIL"
                links["local"] = f"{LOCAL_TRACKING_URI.rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}"
                ok(f"Tersimpan di {LOCAL_TRACKING_URI} (run induk: {run_id})")
            except Exception as e:
                status["local"] = f"GAGAL: {e}"
                print(f"  [GAGAL] Pencatatan lokal: {e}")
        else:
            print("  (dilewati: --targets tidak memuat 'local')")

        # 6. Pencatatan DagsHub (Advance)
        step(6, "Mencatat ke DagsHub (Advance)")
        if "dagshub" in targets:
            try:
                setup_dagshub(args.dagshub_owner, args.dagshub_repo)
                run_id, exp_id = log_to_tracking("dagshub", candidates, best, tmp, files, cfg)
                status["dagshub"] = "BERHASIL"
                links["dagshub"] = f"https://dagshub.com/{args.dagshub_owner}/{args.dagshub_repo}/experiments"
                ok(f"Tersimpan di DagsHub (run induk: {run_id})")
            except Exception as e:
                status["dagshub"] = f"GAGAL: {type(e).__name__}: {e}"
                print(f"\n  [GAGAL] Pencatatan DagsHub: {type(e).__name__}: {e}")
                print("  Periksa: (1) repo DagsHub ada dan nama owner/repo benar, (2) login/token DagsHub valid,\n"
                      "  (3) koneksi internet. Catatan lokal tidak terpengaruh. Ulangi hanya DagsHub dengan:\n"
                      "      python modelling_tuning.py --targets dagshub")
        else:
            print("  (dilewati: --targets tidak memuat 'dagshub')")

    # 7. Ringkasan
    step(7, "Ringkasan")
    tm = bc["test_m"]
    print(f"  Kombinasi diuji : {len(grid)} | waktu grid search: {tuning_time:.0f} detik")
    print(f"  Model terbaik   : n_estimators={bc['n_estimators']}, max_depth={bc['max_depth']}, "
          f"min_samples_leaf={bc['min_samples_leaf']}")
    print(f"  CV macro-F1     : {bc['cv_mean']:.4f} +/- {bc['cv_std']:.4f} (data latih, per soal)")
    print(f"  Data uji        : akurasi={tm['test_accuracy_score']:.4f} | macro-F1={bc['test_macro_f1']:.4f}")
    print(f"  Akurasi latih   : {bc['train_m']['training_accuracy_score']:.4f}")
    print("\n  5 fitur terpenting:")
    print("  " + imp.head(5).to_string(index=False).replace("\n", "\n  "))
    print("\n  Status pencatatan:")
    for k in ("local", "dagshub"):
        if k in status:
            print(f"    {k:<8}: {status[k]}")
            if k in links:
                print(f"              {links[k]}")
    print("\n  Untuk screenshot: pastikan address bar 127.0.0.1:5001 terlihat pada tangkapan layar.")
    print("=" * 78)


def parse_args():
    p = argparse.ArgumentParser(description="Grid search RandomForest dengan manual logging MLflow.")
    p.add_argument("--targets", default="local,dagshub", help="local, dagshub, atau local,dagshub (bawaan)")
    p.add_argument("--dagshub-owner", default=os.getenv("DAGSHUB_REPO_OWNER", DAGSHUB_OWNER))
    p.add_argument("--dagshub-repo", default=os.getenv("DAGSHUB_REPO_NAME", DAGSHUB_REPO))
    p.add_argument("--n-estimators", type=parse_grid, default=parse_grid("100,300"))
    p.add_argument("--max-depth", type=parse_grid, default=parse_grid("10,20,None"))
    p.add_argument("--min-samples-leaf", type=parse_grid, default=parse_grid("1,3"))
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
