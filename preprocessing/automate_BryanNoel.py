"""
automate_BryanNoel.py
=====================
Preprocessing otomatis dataset CodeComplex (prediksi kelas kompleksitas waktu kode sumber).

Skrip ini merupakan konversi dari notebook `Eksperimen_BryanNoel.ipynb` (bagian 3 dan 5):
tahapan dan urutannya identik, hanya strukturnya dipecah menjadi fungsi.

    1. load_raw / standardize   : memuat dan menstandardisasi data mentah
    2. extract_all              : ekstraksi fitur struktural dari AST (tree-sitter)
    3. clean_data               : hapus data kosong, duplikat, dan konflik label; kodekan target
    4. split_data               : pembagian latih/uji per soal (StratifiedGroupKFold)
    5. impute_and_clip          : imputasi median dan winsorization P1-P99 (statistik dari data latih)
    6. scale_features           : StandardScaler (dipelajari dari data latih)
    7. save_outputs             : simpan train.csv, test.csv, train_groups.csv,
                                   scaler.joblib, preprocessing_meta.json

Penggunaan (CLI), dari akar proyek Membangun_model/:
    python preprocessing/automate_BryanNoel.py
    # Keluaran default: ./codecomplex_preprocessing/ (langsung untuk modelling_tuning.py)
    python preprocessing/automate_BryanNoel.py --raw codecomplex_raw --out codecomplex_preprocessing

Penggunaan (sebagai modul):
    from automate_BryanNoel import preprocess
    X_train, X_test, y_train, y_test = preprocess(raw_dir, out_dir)
"""

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_python as tspython

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = SCRIPT_DIR.parent / "codecomplex_raw"
# Simpan di akar Membangun_model/ agar langsung cocok dengan modelling_tuning.py.
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / "codecomplex_preprocessing"

CODE_COLS = ["src", "code", "source", "source_code", "content"]
LABEL_COLS = ["complexity", "label", "time_complexity", "target"]
LANG_COLS = ["language", "lang"]
PROBLEM_COLS = ["problem", "problem_id"]

CLASS_ORDER = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "NP-hard/eksponensial"]

LABEL_MAP = {
    "constant": "O(1)", "o(1)": "O(1)", "1": "O(1)",
    "logn": "O(log n)", "log": "O(log n)", "o(logn)": "O(log n)",
    "linear": "O(n)", "o(n)": "O(n)", "n": "O(n)",
    "nlogn": "O(n log n)", "o(nlogn)": "O(n log n)",
    "quadratic": "O(n^2)", "o(n^2)": "O(n^2)", "o(n2)": "O(n^2)", "n2": "O(n^2)", "nsquare": "O(n^2)",
    "cubic": "O(n^3)", "o(n^3)": "O(n^3)", "o(n3)": "O(n^3)", "n3": "O(n^3)", "ncube": "O(n^3)",
    "np": "NP-hard/eksponensial", "nphard": "NP-hard/eksponensial",
    "exponential": "NP-hard/eksponensial", "o(2^n)": "NP-hard/eksponensial", "2^n": "NP-hard/eksponensial",
}

COUNT_FEATURES = ["loc", "n_chars", "ast_nodes", "ast_depth", "loop_count", "for_count", "while_count",
                  "comprehension_count", "max_loop_depth", "branch_count", "max_nesting_depth",
                  "func_count", "call_count", "max_recursive_calls"]
BINARY_FEATURES = ["has_recursion", "has_sort", "has_halving", "has_doubling", "has_binary_search",
                   "has_heap", "has_hash_container", "has_memo", "is_java"]
FEATURES = COUNT_FEATURES + BINARY_FEATURES

# ---------------------------------------------------------------------------
# 1. Pemuatan dan standardisasi data mentah
# ---------------------------------------------------------------------------


def normalize_label(value):
    key = str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return LABEL_MAP.get(key, None)


def normalize_code(code):
    """Normalisasi spasi agar duplikat yang hanya berbeda format dapat terdeteksi."""
    return re.sub(r"\s+", " ", str(code)).strip()


def _first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def infer_language(code, hint=""):
    h = str(hint).lower()
    if "java" in h and "javascript" not in h:
        return "java"
    if "py" in h:
        return "python"
    return "java" if re.search(r"import java\.|public\s+(static\s+)?(class|void|int)|System\.out", str(code)) else "python"


def read_any(path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            with open(path, encoding="utf-8") as f:
                return pd.DataFrame(json.load(f))
    if suffix == ".csv":
        return pd.read_csv(path)
    return None


def load_raw(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Direktori data mentah tidak ditemukan: {raw_dir.resolve()}")
    frames = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        part = read_any(path)
        if part is None:
            continue
        part["source_file"] = str(path.relative_to(raw_dir))
        frames.append(part)
    if not frames:
        raise ValueError("Tidak ada berkas .jsonl/.json/.csv pada direktori data mentah.")
    return pd.concat(frames, ignore_index=True)


def standardize(df_raw):
    cols = list(df_raw.columns)
    code_col = _first_present(cols, CODE_COLS)
    label_col = _first_present(cols, LABEL_COLS)
    lang_col = _first_present(cols, LANG_COLS)
    if code_col is None or label_col is None:
        raise KeyError(f"Kolom kode/label tidak ditemukan. Kolom tersedia: {cols}")
    problem_col = _first_present(cols, PROBLEM_COLS)

    out = pd.DataFrame({"code": df_raw[code_col], "label_raw": df_raw[label_col]})
    if problem_col is not None:
        out["problem"] = df_raw[problem_col].astype(str)
    else:
        print("PERINGATAN: kolom 'problem' tidak ditemukan; setiap sampel diperlakukan sebagai kelompok tersendiri.")
        out["problem"] = [f"row_{i}" for i in range(len(out))]
    out["label"] = out["label_raw"].map(normalize_label)

    if lang_col is not None:
        out["language"] = df_raw[lang_col].astype(str).str.lower().map(lambda v: "java" if "java" in v else "python")
    else:
        out["language"] = [infer_language(c, h) for c, h in zip(df_raw[code_col], df_raw["source_file"])]

    unmapped = out.loc[out["label"].isna() & out["label_raw"].notna(), "label_raw"].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Label berikut belum terpetakan pada LABEL_MAP: {list(unmapped)}")
    return out[["code", "language", "label", "label_raw", "problem"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Ekstraksi fitur struktural dari AST
# ---------------------------------------------------------------------------
PARSERS = {
    "python": Parser(Language(tspython.language())),
    "java": Parser(Language(tsjava.language())),
}

NODE_CFG = {
    "python": {
        "for": {"for_statement"},
        "while": {"while_statement"},
        "comp": {"list_comprehension", "set_comprehension", "dictionary_comprehension", "generator_expression"},
        "branch": {"if_statement", "elif_clause", "conditional_expression", "match_statement"},
        "func": {"function_definition"},
        "call": {"call"},
    },
    "java": {
        "for": {"for_statement", "enhanced_for_statement"},
        "while": {"while_statement", "do_statement"},
        "comp": set(),
        "branch": {"if_statement", "switch_expression", "switch_statement", "ternary_expression"},
        "func": {"method_declaration", "constructor_declaration"},
        "call": {"method_invocation"},
    },
}

RE_SORT = re.compile(r"\bsorted?\s*\(|\.sort\s*\(|\bArrays\.sort\b|\bCollections\.sort\b")
RE_HALVING = re.compile(r"/=?\s*2\b|>>=?\s*1\b")
RE_DOUBLING = re.compile(r"\*=\s*2\b|<<=?\s*1\b|\*\s*2\b")
RE_BSEARCH = re.compile(r"bisect|binarySearch|\bmid\b")
RE_HEAP = re.compile(r"heapq|heappush|heappop|PriorityQueue")
RE_HASH = re.compile(r"\bdict\s*\(|\bset\s*\(|defaultdict|Counter\s*\(|HashMap|HashSet|LinkedHashMap")
RE_MEMO = re.compile(r"lru_cache|\bmemo\b|\bcache\b|\bdp\b")


def _text(node):
    return node.text.decode("utf-8", errors="ignore") if node is not None else ""


def _callee_name(call_node, language):
    if language == "python":
        fn = call_node.child_by_field_name("function")
        if fn is None:
            return ""
        if fn.type == "identifier":
            return _text(fn)
        if fn.type == "attribute":
            return _text(fn.child_by_field_name("attribute"))
        return ""
    return _text(call_node.child_by_field_name("name"))


def _iter_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def extract_features(code, language):
    """Ekstraksi vektor fitur struktural dari satu potongan kode (satu kali penelusuran AST)."""
    code = "" if code is None else str(code)
    cfg = NODE_CFG[language]
    tree = PARSERS[language].parse(code.encode("utf-8", errors="ignore"))
    root = tree.root_node

    ast_nodes = ast_depth = 0
    loop_count = for_count = while_count = comp_count = 0
    branch_count = call_count = 0
    max_loop_depth = max_nest = 0
    func_nodes = []

    stack = [(root, 1, 0, 0)]  # (node, kedalaman AST, kedalaman loop, kedalaman nesting)
    while stack:
        node, d, ld, nd = stack.pop()
        ast_nodes += 1
        ast_depth = max(ast_depth, d)
        t = node.type
        new_ld, new_nd = ld, nd

        if t in cfg["for"]:
            for_count += 1; loop_count += 1; new_ld += 1; new_nd += 1
        elif t in cfg["while"]:
            while_count += 1; loop_count += 1; new_ld += 1; new_nd += 1
        elif t in cfg["comp"]:
            k = sum(1 for c in node.children if c.type == "for_in_clause")
            comp_count += 1; loop_count += k; for_count += k; new_ld += k; new_nd += k
        elif t in cfg["branch"]:
            branch_count += 1
            is_else_if = (t == "if_statement" and node.parent is not None and node.parent.type == "if_statement")
            if t != "elif_clause" and not is_else_if:
                new_nd += 1
        elif t in cfg["func"]:
            func_nodes.append(node)

        if t in cfg["call"]:
            call_count += 1

        max_loop_depth = max(max_loop_depth, new_ld)
        max_nest = max(max_nest, new_nd)
        for c in node.children:
            stack.append((c, d + 1, new_ld, new_nd))

    # Rekursi: panggilan fungsi terhadap dirinya sendiri
    max_rec_calls = 0
    for fn in func_nodes:
        name = _text(fn.child_by_field_name("name"))
        if not name:
            continue
        n_self = sum(1 for n in _iter_nodes(fn)
                     if n.type in cfg["call"] and _callee_name(n, language) == name)
        max_rec_calls = max(max_rec_calls, n_self)

    lines = [ln for ln in code.splitlines() if ln.strip()]
    return {
        "loc": len(lines),
        "n_chars": len(code),
        "ast_nodes": ast_nodes,
        "ast_depth": ast_depth,
        "loop_count": loop_count,
        "for_count": for_count,
        "while_count": while_count,
        "comprehension_count": comp_count,
        "max_loop_depth": max_loop_depth,
        "branch_count": branch_count,
        "max_nesting_depth": max_nest,
        "func_count": len(func_nodes),
        "call_count": call_count,
        "max_recursive_calls": max_rec_calls,
        "has_recursion": int(max_rec_calls > 0),
        "has_sort": int(bool(RE_SORT.search(code))),
        "has_halving": int(bool(RE_HALVING.search(code))),
        "has_doubling": int(bool(RE_DOUBLING.search(code))),
        "has_binary_search": int(bool(RE_BSEARCH.search(code))),
        "has_heap": int(bool(RE_HEAP.search(code))),
        "has_hash_container": int(bool(RE_HASH.search(code))),
        "has_memo": int(bool(RE_MEMO.search(code))),
        "is_java": int(language == "java"),
        "parse_error": int(root.has_error),
    }


def extract_all(frame):
    rows = []
    for code_text, lang in zip(frame["code"], frame["language"]):
        try:
            rows.append(extract_features(code_text, lang))
        except Exception:
            failed = {k: np.nan for k in COUNT_FEATURES + BINARY_FEATURES}
            failed["is_java"] = int(lang == "java")
            failed["parse_error"] = 1
            rows.append(failed)
    return pd.DataFrame(rows, index=frame.index)


# ---------------------------------------------------------------------------
# 3. Pembersihan data dan pengodean target (notebook 5.1 - 5.3)
# ---------------------------------------------------------------------------
def clean_data(df_all):
    n0 = len(df_all)
    df_clean = df_all.copy()
    df_clean["code"] = df_clean["code"].fillna("")

    # 5.1 data kosong
    valid_mask = df_clean["code"].str.strip().ne("") & df_clean["label"].notna()
    df_clean = df_clean[valid_mask].copy()
    n_empty = n0 - len(df_clean)

    # 5.2 konflik label lalu duplikat murni
    df_clean["_norm"] = df_clean["code"].map(normalize_code)
    label_nunique = df_clean.groupby("_norm")["label"].nunique()
    conflict_keys = set(label_nunique[label_nunique > 1].index)
    df_clean = df_clean[~df_clean["_norm"].isin(conflict_keys)]
    n_after_conflict = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset="_norm", keep="first").drop(columns="_norm")

    # 5.3 pengodean target (urut menurut kompleksitas)
    label2id = {c: i for i, c in enumerate(CLASS_ORDER)}
    df_clean["target"] = df_clean["label"].map(label2id).astype(int)

    stats = {
        "baris_awal": n0,
        "dihapus_kosong": n_empty,
        "dihapus_konflik_label": (n0 - n_empty) - n_after_conflict,
        "dihapus_duplikat": n_after_conflict - len(df_clean),
        "baris_akhir": len(df_clean),
    }
    return df_clean, label2id, stats


# ---------------------------------------------------------------------------
# 4. Pembagian latih/uji per soal (notebook 5.4)
# ---------------------------------------------------------------------------
def split_data(df_clean, seed=SEED):
    X = df_clean[FEATURES].copy()
    y = df_clean["target"].copy()
    groups = df_clean["problem"].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

    overlap = set(g_train) & set(g_test)
    assert len(overlap) == 0, f"Kebocoran: {len(overlap)} soal muncul pada himpunan latih dan uji"
    return X_train, X_test, y_train, y_test, g_train, g_test


# ---------------------------------------------------------------------------
# 5. Imputasi dan penanganan pencilan (notebook 5.5)
# ---------------------------------------------------------------------------
def impute_and_clip(X_train, X_test):
    medians = X_train[COUNT_FEATURES].median()

    X_train[COUNT_FEATURES] = X_train[COUNT_FEATURES].fillna(medians)
    X_test[COUNT_FEATURES] = X_test[COUNT_FEATURES].fillna(medians)
    X_train[BINARY_FEATURES] = X_train[BINARY_FEATURES].fillna(0)
    X_test[BINARY_FEATURES] = X_test[BINARY_FEATURES].fillna(0)

    clip_lower = X_train[COUNT_FEATURES].quantile(0.01)
    clip_upper = X_train[COUNT_FEATURES].quantile(0.99)

    X_train[COUNT_FEATURES] = X_train[COUNT_FEATURES].clip(clip_lower, clip_upper, axis=1)
    X_test[COUNT_FEATURES] = X_test[COUNT_FEATURES].clip(clip_lower, clip_upper, axis=1)
    return X_train, X_test, medians, clip_lower, clip_upper


# ---------------------------------------------------------------------------
# 6. Standardisasi fitur (notebook 5.6)
# ---------------------------------------------------------------------------
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    X_train_s[COUNT_FEATURES] = scaler.fit_transform(X_train[COUNT_FEATURES])
    X_test_s[COUNT_FEATURES] = scaler.transform(X_test[COUNT_FEATURES])
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# 7. Penyimpanan dan verifikasi (notebook 5.7 - 5.8)
# ---------------------------------------------------------------------------
def save_outputs(out_dir, X_train_s, X_test_s, y_train, y_test, g_train, g_test, scaler,
                 label2id, medians, clip_lower, clip_upper, seed=SEED):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = X_train_s.copy()
    train_out["target"] = y_train.values
    test_out = X_test_s.copy()
    test_out["target"] = y_test.values

    # IDs kelompok harus berurutan persis sama dengan baris pada train.csv.
    if len(g_train) != len(train_out) or len(g_test) != len(test_out):
        raise ValueError("Jumlah group ID tidak sesuai dengan jumlah baris data latih/uji.")
    overlap = set(g_train.astype(str)) & set(g_test.astype(str))
    if overlap:
        raise ValueError(f"Kebocoran kelompok: {len(overlap)} soal muncul di latih dan uji.")

    groups_out = pd.DataFrame({"problem": g_train.astype(str).to_numpy()})
    train_out.to_csv(out_dir / "train.csv", index=False)
    test_out.to_csv(out_dir / "test.csv", index=False)
    groups_out.to_csv(out_dir / "train_groups.csv", index=False)
    joblib.dump(scaler, out_dir / "scaler.joblib")

    meta = {
        "seed": seed,
        "split_scheme": "StratifiedGroupKFold(n_splits=5, shuffle=True), group=problem, test=fold pertama",
        "features": FEATURES,
        "count_features": COUNT_FEATURES,
        "binary_features": BINARY_FEATURES,
        "class_order": CLASS_ORDER,
        "label2id": label2id,
        "impute_median": medians.to_dict(),
        "clip_lower": clip_lower.to_dict(),
        "clip_upper": clip_upper.to_dict(),
        "n_train": int(len(train_out)),
        "n_test": int(len(test_out)),
        "n_train_groups": int(g_train.nunique()),
        "n_test_groups": int(g_test.nunique()),
        "train_groups_file": "train_groups.csv",
    }
    with open(out_dir / "preprocessing_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # verifikasi integritas
    train_chk = pd.read_csv(out_dir / "train.csv")
    test_chk = pd.read_csv(out_dir / "test.csv")
    groups_chk = pd.read_csv(out_dir / "train_groups.csv")
    assert train_chk.isna().sum().sum() == 0 and test_chk.isna().sum().sum() == 0, "Masih ada nilai kosong"
    assert list(train_chk.columns) == FEATURES + ["target"], "Urutan kolom latih tidak sesuai"
    assert list(test_chk.columns) == FEATURES + ["target"], "Urutan kolom uji tidak sesuai"
    assert list(groups_chk.columns) == ["problem"], "Kolom train_groups.csv harus bernama 'problem'"
    assert len(groups_chk) == len(train_chk), "Jumlah group ID tidak sama dengan baris train.csv"
    assert groups_chk["problem"].notna().all(), "train_groups.csv berisi group ID kosong"
    assert groups_chk["problem"].tolist() == g_train.astype(str).tolist(), "Urutan group ID tidak selaras dengan train.csv"
    return train_out, test_out


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------
def preprocess(raw_dir=DEFAULT_RAW_DIR, out_dir=DEFAULT_OUT_DIR, seed=SEED, verbose=True):
    """Menjalankan seluruh pra-pemrosesan dan mengembalikan data siap latih.

    Returns
    -------
    X_train, X_test : DataFrame fitur terstandardisasi
    y_train, y_test : Series target (indeks kelas 0-6 sesuai CLASS_ORDER)
    """
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    log(f"[1/7] Memuat data mentah dari {Path(raw_dir).resolve()}")
    df = standardize(load_raw(raw_dir))
    log(f"      {len(df):,} sampel, {df['problem'].nunique():,} soal unik")

    log("[2/7] Ekstraksi fitur AST (tree-sitter) ...")
    df_feat = extract_all(df)
    df_all = pd.concat([df, df_feat], axis=1)

    log("[3/7] Pembersihan data dan pengodean target")
    df_clean, label2id, stats = clean_data(df_all)
    log("      " + ", ".join(f"{k}={v:,}" for k, v in stats.items()))

    log("[4/7] Pembagian latih/uji per soal (StratifiedGroupKFold)")
    X_train, X_test, y_train, y_test, g_train, g_test = split_data(df_clean, seed)
    log(f"      latih={len(X_train):,} ({g_train.nunique()} soal), uji={len(X_test):,} ({g_test.nunique()} soal)")

    log("[5/7] Imputasi median dan winsorization P1-P99 (dari data latih)")
    X_train, X_test, medians, clip_lower, clip_upper = impute_and_clip(X_train, X_test)

    log("[6/7] Standardisasi fitur hitungan")
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    log(f"[7/7] Menyimpan keluaran ke {Path(out_dir).resolve()}")
    save_outputs(out_dir, X_train_s, X_test_s, y_train, y_test, g_train, g_test, scaler,
                 label2id, medians, clip_lower, clip_upper, seed)
    log(f"      train_groups.csv: {len(g_train):,} ID untuk {g_train.nunique():,} soal latih")

    log(f"Selesai dalam {time.time() - t0:.1f} detik.")
    return X_train_s, X_test_s, y_train, y_test


def parse_args():
    p = argparse.ArgumentParser(description="Preprocessing otomatis dataset CodeComplex.")
    p.add_argument("--raw", default=str(DEFAULT_RAW_DIR), help="Folder data mentah (default: ../codecomplex_raw)")
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                   help="Folder keluaran (default: ../codecomplex_preprocessing dari folder preprocessing/)")
    p.add_argument("--seed", type=int, default=SEED, help="Seed pembagian data (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.raw, args.out, args.seed)
