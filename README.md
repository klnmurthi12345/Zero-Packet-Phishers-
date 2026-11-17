# Zero-Packet Phishers: URL-only phishing detection and a feature-availability audit

ES 335 · Machine Learning · Undergraduate course project

**Live site:** https://klnmurthi12345.github.io/Zero-Packet-Phishers-/ (GitHub Pages; the site is the single file `index.html`)
**Repository:** https://github.com/klnmurthi12345/Zero-Packet-Phishers-

The website is the primary write-up (question, data, preprocessing, models, evaluation, results, errors, limitations, reproduction). This repository holds the code, data and results needed to check it.

## Question, unit and target

* **Question.** Can a click-time filter that sees only the raw URL string ("zero-packet": no page fetch, no WHOIS, no redirects) detect phishing URLs, and how much does using columns that are not available at click-time change a benchmark score?
* **One row** = one URL string from the UCI PhiUSIIL benchmark.
* **Target** `is_phishing` (1 = phishing, 0 = legitimate). The raw `label` column is the opposite (1 = legitimate); it is flipped in `audit.ipynb`.
* **Prediction time.** When the user clicks or hovers over a link. Only the URL string is available.

## Headline results (held-out test split, 47,074 URLs)

| Model (default settings) | ROC-AUC | 95% bootstrap CI | ROC-AUC after validation tuning | Recall at τ=0.5 | FP | FN |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9961 | [0.9955, 0.9968] | 0.9963 | 98.64% | 17 | 273 |
| Decision Tree | 0.9948 | [0.9939, 0.9957] | 0.9977 | 99.13% | 67 | 175 |
| Random Forest | 0.9969 | [0.9962, 0.9975] | 0.9976 | 99.14% | 43 | 172 |

* Random Forest at the validation-tuned τ = 0.0809: recall 99.33%, FPR 0.92% (95% CI up to 1.03%).
* Paired AUC differences: RF − DT +0.0020 (CI +0.0015 to +0.0026); RF − LR +0.0007 (CI +0.0000 to +0.0014).
* `URLSimilarityIndex` alone reaches validation AUC 0.9959. Adding it and `TLDLegitimateProb` lifts validation AUC to 0.9999 for both Logistic Regression and Decision Tree. Both columns need an offline reference list that a browser does not have at click-time, so they are excluded from all main models.

### Caveats that matter more than the headline

* **The benchmark is structurally easy.** All 134,850 legitimate URLs use HTTPS and have no path, query or special symbol. A parameter-free rule ("flag if HTTP or has a path") scores 99.07% accuracy and 97.82% recall with 0 false positives on the test split. (The rule was written after looking at class-conditional rates on the full corpus, so it is a diagnostic baseline.)
* **On the hard subset** (bare HTTPS root URLs, 27,408 test URLs of which 438 are phishing) the Random Forest's ROC-AUC is 0.857. Every false negative lies in this subset.
* **Hostname memorisation is not the explanation.** 7.7% of test URLs share a hostname with training URLs in the random split. A hostname-grouped split gives Random Forest AUC 0.9968 (random split: 0.9969).
* **Operating point.** At 1% phishing prevalence the tuned τ gives about 52% precision; τ = 0.90 gives about 97%. These use the formula on the site with measured TPR/FPR; the benchmark itself is about 43% phishing.
* One dataset, one split seed (42). Bootstrap intervals capture test-sample noise, not split or seed variation. Tuned-model and extra-feature results are single runs.

## Repository contents

| File | Purpose |
|---|---|
| `index.html` | The project website (single static file; HTML, CSS, JavaScript) |
| `audit.ipynb` | Main notebook: dataset checks, 55-column audit, feature extraction, split, Experiments A, B and C, error samples, slider export |
| `final_analysis.py` | Re-derives the notebook's numbers (asserts they match), then adds baselines, bootstrap CIs, tuning, grouped split, ablations, error analysis, calibration |
| `results/final_results.json` | Every number produced by `final_analysis.py` and shown in the website's extra-evidence blocks |
| `results/final_analysis_log.txt` | Console log of the last run |
| `phiusiil_phishing.csv` | Raw UCI PhiUSIIL data (235,795 rows, 55 columns) |
| `phiusiil_zero_packet_features.csv` | 13 extracted features + raw label (written by the notebook) |
| `phiusiil_working_dataset.csv` | Features + `is_phishing` + the two audited columns (written by the notebook) |
| `dataset_column_audit.csv` | Treatment of all 55 columns |
| `simulator_data.json` | Random Forest test-set confusion counts at 100 thresholds, plus Experiment A values (feeds the slider) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. The numbers on the site were re-derived with Python 3.12, pandas 3.0, NumPy 2.5 and scikit-learn 1.9. Other recent versions should reproduce them closely, but exact values can shift in the last digit across scikit-learn releases.

## Data access

The data is the public UCI **PhiUSIIL Phishing URL Dataset** (ID 967). `phiusiil_phishing.csv` is included in the repository so nothing needs downloading. To fetch it again:

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd
d = fetch_ucirepo(id=967)
pd.concat([d.data.features, d.data.targets], axis=1).to_csv("phiusiil_phishing.csv", index=False)
```

Check the dataset's page on the UCI Machine Learning Repository for its licence and cite the paper below. The file contains URLs of real phishing sites; do not open them.

## How to reproduce

1. Run `audit.ipynb` top to bottom (from the repository root). It writes `phiusiil_zero_packet_features.csv`, `phiusiil_working_dataset.csv`, `dataset_column_audit.csv` and `simulator_data.json`.
2. Run `python final_analysis.py` from the repository root (about 10–15 minutes on one CPU core; the bootstrap and grid searches dominate). It first checks that
   * features recomputed from the raw URLs equal the stored features,
   * the split sizes are 141,222 / 47,074 / 47,074,
   * test AUCs, τ = 0.0809, Experiment A and all 100 slider thresholds equal what the notebook exported,

   and stops with an assertion error if any of these differ.
3. Open `index.html` in a browser (it loads Tailwind, Chart.js and KaTeX from CDNs, so it needs internet access).

### Exact split and preprocessing

* Duplicates: 425 exact duplicate URLs (all phishing) dropped before splitting, leaving 235,370 rows.
* Split: `train_test_split(test_size=0.20, stratify=y, random_state=42)` then `test_size=0.25, stratify=y_temp, random_state=42` on the remainder, giving 60/20/20 = 141,222 / 47,074 / 47,074.
* Scaling: `StandardScaler` fitted on the training split only, used for Logistic Regression; tree models use raw features.
* Model selection: the validation split is used only for τ and for the tuning grids. Each test number comes from a model fitted on the training split and scored once on the test split.
* Features (13, from the URL string only): `url_length, hostname_length, dot_count, hyphen_count, digit_count, digit_ratio, special_symbols, subdomain_count, uses_https, host_is_ip, suspicious_words, path_len_ratio, tld_length`.

## Which result comes from where

| Website part | Source |
|---|---|
| Sections 2–3: counts, duplicates, feature summary, 55-column audit | `audit.ipynb`, `dataset_column_audit.csv`; class-conditional rates in `final_analysis.py` |
| Experiment A | `audit.ipynb` (re-checked by `final_analysis.py`) |
| Experiment B table, confusion matrices, Experiment C, slider, first-five error examples | `audit.ipynb`, `simulator_data.json` |
| Baselines, tuning, bootstrap, grouped split, ablation, hard subset, extra features, error breakdown, calibration, prevalence-adjusted precision, robustness explorer | `final_analysis.py` → `results/final_results.json` |
| URL extractor panel | Logistic Regression coefficients and scaler statistics from the notebook split, embedded as `LR_PARAMS` in `index.html` |

## Known issues

* `audit.ipynb` prose says "12 features" in places; the code extracts and uses 13.
* The notebook's first-five error examples are `.head(5)` of the test-ordered errors, not a random sample.
* The ablation that drops `uses_https`, `path_len_ratio` and `special_symbols` is not a clean removal of path information, because `url_length` minus `hostname_length` still encodes it.
* Hostname-grouped split groups by hostname without `www.`, not by registered domain.
* Feature extraction time (1.76 s in the notebook run) depends on the machine.

## Team Contributions

The project was carried out collaboratively by **Golla Gowtham Surya** and **K. L. N. Murthi**, with each member taking primary responsibility for different technical components.

### 1. Golla Gowtham Surya

**Dataset Engineering, Feature Extraction and Baseline Modelling**

- Acquired and studied the UCI PhiUSIIL Phishing URL Dataset.
- Performed dataset cleaning, duplicate removal and data-quality checks.
- Conducted an audit of all 55 dataset columns.
- Designed and implemented 13 URL-based lexical features.
- Prepared the working and feature-engineered datasets.
- Designed the training, validation and testing split.
- Implemented and evaluated Logistic Regression and Decision Tree baseline models.
- Contributed to baseline comparisons and evaluation.

### 2. K. L. N. Murthi

**Advanced Model Analysis, Evaluation and Interactive Website**

- Implemented and evaluated the Random Forest model.
- Performed hyperparameter tuning and threshold optimization.
- Conducted bootstrap confidence-interval analysis.
- Investigated data leakage and hostname-grouped evaluation.
- Performed ablation studies and hard-subset evaluation.
- Conducted error analysis and model calibration.
- Developed the interactive project website using HTML, CSS and JavaScript.
- Implemented visualizations, the interactive URL inspector and threshold simulator.
- Integrated experimental results into the website and handled GitHub Pages deployment.

### 3. Joint Contributions

Both team members collaborated on:

- Defining the research problem and project objectives.
- Selecting machine-learning methods and evaluation metrics.
- Discussing experimental methodology and interpreting results.
- Identifying limitations and potential improvements.
- Testing the complete project and checking reproducibility.
- Preparing project documentation and the final presentation.

### 4. Collaboration Statement

Both members contributed to the development and integration of the project. Each member is responsible for understanding the complete project, including its methodology, experimental results, evaluation and limitations.

## AI-use disclosure

Generative AI assistants (Google Gemini and Anthropic Claude) were used for brainstorming, scaffolding the HTML/Tailwind layout and Chart.js code, debugging, a critical review of the site against the course rubric, drafting `final_analysis.py` and this README. The analysis code was run on the repository data; the notebook numbers were re-derived and are asserted by `final_analysis.py`; the website's numbers for the extra analyses are read from `results/final_results.json`. Both team members are responsible for understanding the experimental results and being able to explain the methodology, implementation and conclusions of the project.

## Citations

* Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. *Computers & Security*, 136, 103545. Dataset: UCI Machine Learning Repository, ID 967.
* Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
* McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*.
* Chart.js, Tailwind CSS and KaTeX (loaded from CDNs on the website); `tldextract` (public-suffix lookups in the error analysis).

## Authors

- Kangeti Laxmi Narayana Murthi (GitHub: klnmurthi12345). Roll number: 25110156.
- Golla Gowtham Surya (GitHub: Gowthamsuryagolla). Roll number: 25110119.