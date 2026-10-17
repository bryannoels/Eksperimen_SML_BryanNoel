# BigOracle: Worst-Case Time Complexity Prediction

![Preprocess Dataset](https://github.com/<username>/Eksperimen_SML_BryanNoel/actions/workflows/preprocess.yml/badge.svg)

Static prediction of a code snippet's worst-case time complexity (7 classes) from structural features extracted from its AST, without executing the code. This repository contains the **experimentation and data-preprocessing stage**: exploratory analysis, feature extraction, an automated preprocessing script, and a CI workflow that regenerates the training dataset on every trigger.

```
codecomplex_raw/*.jsonl
   -> standardize -> AST feature extraction (tree-sitter) -> clean -> grouped split
   -> impute + winsorize -> scale -> preprocessing/codecomplex_preprocessing/{train,test}.csv
```

## Dataset

[CodeComplex](https://github.com/sybaik1/CodeComplex) ([paper](https://arxiv.org/abs/2401.08719)): Codeforces solutions annotated with worst-case time complexity.

| Property | Value |
|---|---|
| Languages | Java (4,900), Python (4,900) |
| Samples | 9,800 |
| Unique problems | 300 (258 Java, 277 Python, 235 shared) |
| Raw fields | `src`, `complexity`, `problem`, `from`, `tags` |
| Raw labels | `constant`, `logn`, `linear`, `nlogn`, `quadratic`, `cubic`, `np` |

Labels are mapped to an ordered 7-class target:

| id | class | id | class |
|---|---|---|---|
| 0 | O(1) | 4 | O(n^2) |
| 1 | O(log n) | 5 | O(n^3) |
| 2 | O(n) | 6 | NP-hard / exponential |
| 3 | O(n log n) | | |

Raw files are stored unmodified in `codecomplex_raw/`. Refer to the upstream repository for license and citation terms.

## Repository Structure

```
Eksperimen_SML_BryanNoel/
├── .github/workflows/preprocess.yml     # CI workflow (executed by GitHub Actions)
├── .workflow/preprocess.yml             # copy of the workflow, kept for the required layout
├── codecomplex_raw/                     # raw dataset (java_data.jsonl, python_data.jsonl)
└── preprocessing/
    ├── Eksperimen_BryanNoel.ipynb       # manual experiment: loading, EDA, preprocessing
    ├── automate_BryanNoel.py            # automated preprocessing (same steps as the notebook)
    ├── requirements.txt
    └── codecomplex_preprocessing/       # output: train.csv, test.csv, scaler.joblib, preprocessing_meta.json
```

## Feature Set

Each snippet is parsed with [tree-sitter](https://tree-sitter.github.io/) (Java and Python grammars) and reduced to 23 model features in a single AST traversal, plus regex-based algorithmic-pattern flags.

| Group | Features |
|---|---|
| Size | `loc`, `n_chars`, `ast_nodes`, `ast_depth` |
| Loops | `loop_count`, `for_count`, `while_count`, `comprehension_count`, `max_loop_depth` |
| Branching | `branch_count`, `max_nesting_depth` |
| Functions / recursion | `func_count`, `call_count`, `max_recursive_calls`, `has_recursion` |
| Algorithmic patterns (lexical) | `has_sort`, `has_halving`, `has_doubling`, `has_binary_search`, `has_heap`, `has_hash_container`, `has_memo` |
| Meta | `is_java` |

`parse_error` is computed for audit purposes only and is not a model feature. Python comprehensions count as loops, and `else if` chains do not inflate nesting depth.

## Preprocessing Pipeline

The notebook (`Eksperimen_BryanNoel.ipynb`) is the exploratory reference; `automate_BryanNoel.py` is its function-level conversion with an identical step order. Both produce byte-identical `train.csv`, `test.csv`, `scaler.joblib` and `preprocessing_meta.json`.

| Step | Operation | Detail |
|---|---|---|
| 1 | Load and standardize | Auto-detects code, label, language and problem columns; maps labels to the 7 classes |
| 2 | Feature extraction | 24 columns per sample (23 features + `parse_error`) |
| 3 | Missing values | Drops empty code or missing labels |
| 4 | Duplicates and conflicts | Drops groups of whitespace-normalized identical code that have conflicting labels, then deduplicates |
| 5 | Target encoding | Ordinal by complexity, not alphabetical |
| 6 | Split | `StratifiedGroupKFold(5, shuffle=True, seed=42)`, first fold as test, grouped by `problem` |
| 7 | Impute + winsorize | Median imputation and P1-P99 clipping, fitted on train only |
| 8 | Scale | `StandardScaler` on the 14 count features, fitted on train only |

**Design decisions**

- **Grouped split.** The 9,800 samples come from only 300 problems (about 33 solutions per problem). A random row-level split places solutions to the same problem in both train and test, which inflates test scores. The split is therefore grouped by `problem`, and the pipeline asserts that no problem appears on both sides.
- **No leakage from statistics.** Medians, clipping bounds and scaler parameters are learned from the training set only and saved to `preprocessing_meta.json` and `scaler.joblib` so they can be re-applied at inference time.
- **Percentile clipping instead of IQR.** Count features are dominated by small values and zeros, which makes the IQR frequently zero and flags most positive values as outliers.

**Reference run**

| Stage | Samples |
|---|---|
| Raw | 9,800 |
| Removed: conflicting labels | 133 |
| Removed: exact duplicates | 283 |
| After cleaning | 9,384 |
| Train | 7,462 (246 problems) |
| Test | 1,922 (54 problems) |

Exact split sizes may vary slightly across scikit-learn versions.

## Quick Start

Requires Python 3.10+ (CI uses 3.12).

```bash
# from the repository root
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r preprocessing/requirements.txt
```

**Run the automated preprocessing**

```bash
python preprocessing/automate_BryanNoel.py
# or with explicit paths
python preprocessing/automate_BryanNoel.py --raw codecomplex_raw --out preprocessing/codecomplex_preprocessing --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--raw` | `../codecomplex_raw` (relative to the script) | Raw data directory |
| `--out` | `./codecomplex_preprocessing` (relative to the script) | Output directory |
| `--seed` | `42` | Seed for the grouped split |

**Use as a module**

```python
from automate_BryanNoel import preprocess

X_train, X_test, y_train, y_test = preprocess(raw_dir, out_dir)   # scaled features, class ids 0-6
```

**Run the notebook.** Open `preprocessing/Eksperimen_BryanNoel.ipynb`, select the `.venv` kernel, and run all cells from the `preprocessing/` directory.

**Outputs** (`preprocessing/codecomplex_preprocessing/`)

| File | Content |
|---|---|
| `train.csv`, `test.csv` | 23 scaled features + `target` |
| `scaler.joblib` | Fitted `StandardScaler` for the count features |
| `preprocessing_meta.json` | Feature order, class mapping, imputation medians, clipping bounds, split scheme |

## CI: GitHub Actions

`.github/workflows/preprocess.yml` regenerates the dataset automatically.

| Item | Value |
|---|---|
| Triggers | `push` to `main` touching `codecomplex_raw/`, `automate_BryanNoel.py`, `requirements.txt` or the workflow; manual `workflow_dispatch` |
| Runner | `ubuntu-latest`, Python 3.12, pip cache |
| Steps | checkout, install dependencies, run `automate_BryanNoel.py`, upload artifact, commit changed dataset |
| Output | artifact `codecomplex_preprocessing` and, when the data changed, a bot commit to `preprocessing/codecomplex_preprocessing/` |

The workflow needs `contents: write`. If the commit step fails with a 403, enable **Settings → Actions → General → Workflow permissions → Read and write**.

## Limitations

- Algorithmic-pattern flags are lexical (regular expressions) and can produce false positives, for example `/ 2` unrelated to binary search.
- Structural features do not capture data dependencies between variables or the semantics of loop bounds, so classes that depend on input constraints can remain hard to separate.
- CodeComplex labels follow an annotation convention based on the problem's input variables, which is not fully recoverable from code structure alone.
- The grouped test set contains 54 problems, so single-split scores have high variance. Grouped cross-validation is preferable for final evaluation.

## Roadmap

- [x] Experiment notebook: loading, EDA, preprocessing
- [x] Automated preprocessing script
- [x] GitHub Actions workflow producing the processed dataset
- [ ] Model training with MLflow tracking (local, then DagsHub), hyperparameter tuning
- [ ] CI retraining with MLflow Project and Docker image publishing
- [ ] Serving with Prometheus and Grafana monitoring and alerting
- [ ] C++ feature extractor with parity tests and benchmark against the Python reference

## Citation

```bibtex
@misc{baik2024codecomplex,
  title         = {CodeComplex: Dataset for Worst-Case Time Complexity Prediction},
  author        = {Seung-Yeop Baik and Joonghyuk Hahn and Jungin Kim and Mingi Jeon and Aditi and Yo-Sub Han and Sang-Ki Ko},
  year          = {2024},
  eprint        = {2401.08719},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2401.08719}
}
```
