# Fairness-Aware XAI Audit Dashboard

Machine-learning models that make decisions about people can be **black boxes** and can be **biased** against
groups such as women or Black defendants. Explanation methods (**SHAP**, **LIME**) explain individual predictions,
and **fairness metrics** measure bias - but the two are usually used separately. This project combines them and asks
a further question: **are the explanations themselves equally reliable and similar across demographic groups?**
A second module asks whether explanations are **stable** when the input changes slightly.

The result is a Python package (`src/`), Jupyter notebooks that also run in Google Colab, and a Streamlit dashboard.

## Research questions and hypotheses

**Fairness module**

| | Question | Method |
|---|---|---|
| RQ1 | How unfair are Logistic Regression, Random Forest and XGBoost on Adult Income and COMPAS, by demographic parity and equal opportunity? | fairlearn `MetricFrame`, DP difference/ratio, EO difference, equalized odds |
| RQ2 | Do SHAP and LIME explanations differ between demographic groups (feature reliance, feature ranking, explanation fidelity)? | sensitive/proxy reliance, Spearman/Kendall rank agreement with bootstrap CI, top-5 Jaccard, LIME fidelity gap |
| RQ3 | Are those differences statistically significant? | Mann-Whitney U, rank-biserial effect size, Holm correction |
| RQ4 | Which model + XAI combination best balances accuracy, fairness and explanation consistency? | Pareto analysis (no weighted score) |

Hypotheses for RQ3, tested for each model x method x sensitive attribute x setting and each measure
(sensitive reliance, proxy reliance, LIME fidelity):

* **H0:** explanation reliance / fidelity does **not** differ between the privileged and unprivileged group.
* **H1:** it does differ (two-sided).

**Stability module** (noise of 1%, 5%, 10% of each numeric feature's standard deviation; 0% = baseline)

* **S-H1:** H0 - SHAP and LIME are equally stable (Wilcoxon signed-rank, paired by instance, Holm).
* **S-H2:** H0 - LR, RF and XGBoost are equally stable (Friedman test with Kendall's W, pairwise Wilcoxon post-hoc, Holm).
* **S-H3:** H0 - stability is the same for the privileged and unprivileged group (Mann-Whitney U, Holm).

## Folder structure

```
fairness-xai/
├── README.md, requirements.txt, pytest.ini, .gitignore
├── data/raw/            downloaded datasets (Adult via OpenML cache, COMPAS CSV)
├── data/processed/      train/test splits (CSV)
├── models/              fitted pipelines (joblib), 12 = 2 datasets x 2 settings x 3 models
├── results/tables/      all result tables (CSV) - read by the app
├── results/figures/     all figures (PNG)
├── results/cache/       cached SHAP/LIME explanations
├── results/quick/       output of `run_all --quick` (kept separate from the full results)
├── notebooks/
│   ├── 01_eda.ipynb                     groups, base rates, proxies
│   ├── 02_preprocess_train.ipynb        split, preprocessing, training, performance
│   ├── 03_fairness.ipynb                RQ1
│   ├── 04_explanations_compare.ipynb    RQ2, RQ3, RQ4
│   ├── 05_mitigation.ipynb              bias mitigation, before vs after
│   └── 06_stability.ipynb               Stability module 
├── src/
│   ├── config.py        every path, seed, sample size, model parameter and flag
│   ├── data.py          loaders for Adult and COMPAS
│   ├── preprocess.py    split, ColumnTransformer, one-hot -> original feature mapping
│   ├── models.py        LR / RF / XGBoost pipelines, save/load
│   ├── evaluate.py      performance metrics, ROC curves, confusion matrices
│   ├── fairness.py      fairness metrics and plain-English sentences
│   ├── explain.py       SHAP + LIME per group, cached
│   ├── compare.py       group comparison of explanations, trade-off + Pareto
│   ├── stats.py         Mann-Whitney U, rank-biserial, Holm
│   ├── mitigation.py    ExponentiatedGradient (demographic parity) before vs after
│   ├── stability.py     Stability module
│   └── run_all.py       runs the whole pipeline
├── app/                 Streamlit dashboard: Home.py, pages/1_Data, 2_Models, 3_Fairness, 4_Stability, utils.py
└── tests/               pytest tests
```

## Setup (local, Windows)

Requires Python 3.11-3.13. From the `fairness-xai` folder in PowerShell:

```
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

On macOS/Linux use `.venv/bin/python` instead of `.\.venv\Scripts\python`.
The first run needs internet: Adult is downloaded from OpenML (via fairlearn) and COMPAS from ProPublica's GitHub;
both are cached in `data/raw/` for offline use afterwards.

## Setup (Google Colab)

Every notebook starts with the same setup cell. Choose **one** option in it:

* `USE_DRIVE = True` - upload the `fairness-xai` folder to the top level of *My Drive*; or
* `REPO_URL = "https://github.com/<user>/fairness-xai.git"` - clone from GitHub (a private repository needs a token).

The cell installs `requirements.txt` once. If Colab then asks to **Restart session**, click it and run the cell
again (the install is skipped the second time). Run the notebooks in order 01 -> 06, or run the pipeline from a cell
with `!python -m src.run_all`.

## Running

| Task | Command (from `fairness-xai`) | Expected result |
|---|---|---|
| Full pipeline | `.\.venv\Scripts\python -m src.run_all` | ends with `All done in ... min`; all tables in `results/tables/` |
| Quick test run | `.\.venv\Scripts\python -m src.run_all --quick` | tiny samples, about 3 minutes, results in `results/quick/` |
| Tests | `.\.venv\Scripts\python -m pytest -q` | all tests pass |
| Dashboard | `.\.venv\Scripts\python -m streamlit run app/Home.py` | opens http://localhost:8501 |

`run_all` options: `--quick`, `--datasets adult compas`, `--skip-training` (load saved models),
`--force-explain` (ignore cached explanations), `--no-mitigation`, `--no-stability`, `--no-figures`.
Explanations are cached, so a second run is much faster. Run times are given under *Key results*.

## Configuration (`src/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `SEED` | 42 | one seed for splits, models, sampling, LIME and bootstrap |
| `TEST_SIZE` | 0.30 | stratified test share |
| `INCLUDE_SENSITIVE` / `SETTINGS` | True / both | with and without sensitive features ("fairness through unawareness"); `run_all` runs both |
| `LOGREG_PARAMS`, `RF_PARAMS`, `XGB_PARAMS` | see file | model parameters (RF uses `min_samples_leaf=5`, see *Design decisions*) |
| `SHAP_N_PER_GROUP`, `LIME_N_PER_GROUP` | 200, 100 | explained test instances per group (LIME instances are a subset of the SHAP ones) |
| `LIME_NUM_SAMPLES` | 1000 | perturbed samples per LIME explanation |
| `N_PROXIES`, `PROXY_OVERRIDES` | 3, {} | proxies = top features by mutual information; can be fixed manually |
| `N_BOOTSTRAP`, `ALPHA` | 1000, 0.05 | bootstrap resamples; significance level after Holm |
| `RUN_MITIGATION`, `MITIGATION_EPS`, `MITIGATION_MAX_ITER` | True, 0.01, 50 | bias-mitigation step |
| `RUN_STABILITY`, `STABILITY_NOISE_LEVELS`, `STABILITY_N_INSTANCES`, `STABILITY_SEEDS` | True, (0.01, 0.05, 0.10), 100, (42, 43, 44) | stability step |
| `QUICK_*` | small values | sizes used by `--quick` |

## Method in brief

1. **Data.** Adult: target `>50K`; `fnlwgt` and `education` dropped; rows with `?` removed; race binarised
   (White / Non-White). COMPAS: ProPublica's filters, African-American and Caucasian defendants only, features
   age, sex, race, priors and juvenile counts, charge degree. `decile_score`, `score_text` and `v_decile_score`
   are **excluded** - they are the COMPAS tool's own output (target leakage).
2. **Models.** Each model is a scikit-learn `Pipeline` (StandardScaler + OneHotEncoder, then the classifier),
   trained on a stratified 70/30 split, in two settings: with and without the sensitive features.
3. **Fairness (RQ1).** Per-group selection rate, TPR, FPR and accuracy; demographic parity difference/ratio,
   equal opportunity difference, equalized odds difference; one plain-English sentence per result.
4. **Explanations (RQ2).** SHAP (TreeExplainer for RF/XGBoost, LinearExplainer for LR) and LIME on the same sampled
   test instances per group. One-hot attributions are summed back to the original feature (absolute values).
   Measures: sensitive reliance, proxy reliance, per-group global importance, Spearman/Kendall/Jaccard between
   groups with a bootstrap CI, LIME fidelity gap.
5. **Statistics (RQ3).** Mann-Whitney U per measure, rank-biserial effect size, Holm across all tests of a run.
6. **Trade-off (RQ4).** Accuracy, F1, DP and EO difference, Spearman rho, |fidelity gap|; Pareto-optimal rows marked.
7. **Mitigation.** Logistic Regression + fairlearn `ExponentiatedGradient` with a demographic-parity constraint on the
   primary attribute; performance, fairness and LIME comparison before vs after.
8. **Stability.** Noise on numeric features, original vs perturbed explanation (rho, top-3/top-5 overlap,
   prediction flips), three seeds, tests S-H1 to S-H3.

## Key results

All numbers come from the full run (`python -m src.run_all`, seed 42) and can be checked in `results/tables/`.
On the development laptop the full run took 7.7 minutes with cached explanations; computing the SHAP/LIME
explanations from scratch adds about 4 minutes (Colab's 2-core CPU is slower).

**Performance** (`performance.csv`, with sensitive features): Adult accuracy 0.843 (LR), 0.854 (RF), 0.865 (XGBoost);
COMPAS 0.669, 0.667, 0.664. Removing the sensitive features changes accuracy by at most 0.007.

**RQ1 - fairness** (`fairness.csv`)
* All 24 model x attribute x setting combinations have a demographic-parity ratio below 0.8 (four-fifths rule).
* Example sentences: "XGBoost predicts >50K for 27% of men but 9% of women (demographic parity difference 0.18)";
  "Logistic Regression predicts re-offending within two years for 50% of African-American defendants but 23% of
  Caucasian defendants (demographic parity difference 0.27)".
* On COMPAS the models roughly double the gap present in the data (real race gap 0.13, model gap 0.23-0.28); on
  Adult they slightly narrow it (sex: 0.19 in the data, 0.18 in the models).
* Fairness through unawareness barely helps on Adult: removing the sensitive features removes 2% of the sex gap.

**RQ2/RQ3 - explanations across groups** (`explanation_comparison.csv`, `stat_tests.csv`)
* Proxies chosen automatically: Adult/sex - relationship, marital-status, occupation; COMPAS/race - age,
  priors_count, juv_fel_count.
* 17 of 96 tests are significant after Holm correction - all for **SHAP** (12 sensitive reliance, 5 proxy
  reliance), none for LIME. Example: SHAP on XGBoost gives `sex` 1.7% of the attribution for men and 6.3% for
  women. The direction follows group size (on COMPAS the larger group is African-American and the gap reverses),
  consistent with SHAP measuring features against an average person dominated by the majority group.
* Feature rankings of the two groups are similar: median Spearman rho 0.98, 90% of the 48 comparisons above 0.89,
  lowest 0.71.
* When the sensitive features are removed, the unprivileged group's proxy reliance rises in 23 of 24
  model/method/attribute combinations.

**RQ4 - trade-off** (`tradeoff.csv`): 37 of 48 rows are Pareto-optimal; with six criteria and six rows per table
most rows win on at least one criterion, so the table narrows the choice but does not name one winner.

**Mitigation** (`mitigation.csv`, Logistic Regression, constrained attribute)
* COMPAS/race: DP difference 0.272 -> 0.023, EO difference 0.328 -> 0.058, accuracy 0.669 -> 0.647.
* Adult/sex: DP difference 0.180 -> 0.003, accuracy 0.843 -> 0.829, but EO difference rises 0.124 -> 0.334 -
  equal selection rates are not equal true positive rates when base rates differ.
* LIME group differences: no significant difference before or after (`mitigation_tests.csv`, 0 of 40 tests).

**Stability** (`stability_*.csv`, 100 instances x 3 seeds)
* S-H1: SHAP is more stable than LIME in 30 of 36 tests (H0 rejected in all 30). On COMPAS at 1% noise:
  SHAP rho 0.93-1.00 vs LIME 0.63-0.77.
* LIME's 0%-noise baseline is below 1 (Adult 0.88-0.97, COMPAS 0.73-0.86): much of LIME's "instability" is its own
  sampling randomness. SHAP's baseline is exactly 1.
* S-H2: stability differs between models in all 24 Friedman tests; XGBoost is the least stable.
* S-H3: stability is the same for both groups in 23 of 24 tests; the one significant case (COMPAS, LR, SHAP, race:
  0.992 vs 0.985) is a negligible difference.

## Manual test cases for the dashboard

| # | Page / input | Expected output |
|---|---|---|
| 1 | Home | Title, four research questions, navigation table; "Results status" lists every table as ✅ available after `run_all` |
| 2 | Data, dataset = Adult Income | Rows 45,222; positive rate 24.8%; group sizes and base rates for sex and race |
| 3 | Data, dataset = COMPAS | Rows 5,278; groups African-American / Caucasian and Male / Female |
| 4 | Models, Adult, with sensitive features | Performance table (6 rows), three ROC curves with AUC in the legend, three confusion matrices |
| 5 | Models, switch setting to "without sensitive" | ROC curves and confusion matrices change to the unaware models |
| 6 | Fairness, defaults (Adult, Logistic Regression, sex, SHAP, with) | Four metric tiles (DP difference 0.180), two plain-English sentences, four-fifths warning, per-group bars |
| 7 | Fairness → Explanations | Two top-10 bar charts (Male blue, Female orange), "(sensitive)" / "(proxy)" labels, rank table |
| 8 | Fairness → Comparison | Reliance tiles, Spearman rho with 95% CI, box plots, test table, interpretation paragraph |
| 9 | Fairness → Comparison, XAI method = LIME | A third box plot (fidelity) and a fidelity-gap tile appear |
| 10 | Fairness → Comparison, setting = without sensitive | Sensitive reliance shows "n/a"; proxy reliance still shown |
| 11 | Fairness → Comparison → "Recompute live", 20 per group | Spinner, then a paragraph and a proxy-reliance box plot within seconds |
| 12 | Fairness → Trade-off | Table with Pareto rows highlighted and marked "✓ yes"; accuracy-vs-DP scatter |
| 13 | Fairness → Mitigation | Before/after tiles and bars; the DP difference of the constrained attribute is lower after |
| 14 | Any table → "Download CSV" | A CSV file with the table's rows downloads |
| 15 | Stability → Stability vs noise | Two line charts (rho with CI, top-5 overlap); SHAP at 0% noise = 1.0 |
| 16 | Stability → Run live (COMPAS, XGBoost, SHAP, 10 instances) | Spinner, then a table with one row per noise level (0% row: rho = 1) |
| 17 | Delete `results/tables/fairness.csv`, open Fairness | Warning telling the user to run `python -m src.run_all` instead of an error |

## Design decisions and deviations from the original specification

* **Random Forest uses `min_samples_leaf=5`.** With the default (1) the forest memorised the COMPAS training data
  (train accuracy 0.84, test 0.62). With 5 leaves test accuracy is 0.667 on COMPAS and 0.854 on Adult.
* **shap 0.48.0** is pinned because it has prebuilt wheels for Python 3.11-3.13 (older versions need a C++ compiler
  on Windows with Python 3.13).
* **`--quick` writes to `results/quick/`**, so a test run never overwrites the full results shown by the app.
* **Pareto check, SHAP fidelity.** SHAP has no surrogate model, so its fidelity gap is shown as n/a; in the Pareto
  check it counts as 0 because Tree/Linear SHAP is exact (attributions sum to the model output).
* **Mitigated model.** `ExponentiatedGradient` is a randomised mixture; accuracy, F1 and fairness use its seeded random
  predictions (the guarantee holds for these), ROC-AUC and LIME use its expected probability via fairlearn's
  private `_pmf_predict` (pinned fairlearn version).
* **Stability module integration.** The method was kept (shared noise draw scaled per level, clipping,
  0% baseline, rho/top-k/prediction flip, bootstrap CI, Wilcoxon, Friedman + Kendall's W, Holm) and connected to the
  shared data, models and explainers. Changes: COMPAS added; German Credit not included (the project uses two
  datasets); seeds derived from `SEED` (42, 43, 44); LIME configured like the Fairness module (one-hot columns
  declared categorical); the weighted "balance score" replaced by a Pareto view; group columns added (S-H3).

## Datasets and licences

* **Adult Income** - UCI Machine Learning Repository (Becker & Kohavi, 1996), licensed CC BY 4.0; loaded through
  `fairlearn.datasets.fetch_adult` (OpenML copy, both original files combined).
* **COMPAS** - `compas-scores-two-years.csv` from ProPublica's `compas-analysis` repository (Angwin et al., 2016,
  "Machine Bias"), obtained by ProPublica through a public-records request in Broward County, Florida. Used here
  for non-commercial academic research; see the repository for its terms.

## Ethics note

COMPAS is criminal-justice data about real people. This project **audits** bias in models trained on it; it does
**not** endorse using such models (or the COMPAS tool) for decisions about people. Group labels follow the datasets'
own categories (binary sex; race reduced to two groups), which simplify real identities. The COMPAS file contains
defendants' names; they are never used as features or shown in the dashboard.

## Limitations

* Two datasets and binary group comparisons only; intersectional groups (e.g. Black women) are not analysed.
* Explanations are computed on samples (200 SHAP / 100 LIME instances per group), so small groups give wider
  confidence intervals.
* LIME fidelity is measured on LIME's own synthetic neighbourhood, not on real data, and LIME perturbs one-hot
  columns independently.
* SHAP values are relative to an average person (the background/expected value); because the majority group
  dominates that average, SHAP reliance on the sensitive feature tends to be larger for the minority group. Part of
  a SHAP group gap therefore reflects group size, not unequal treatment.
* Proxies are chosen by mutual information, which captures association, not causation.
* Mitigation covers one method (ExponentiatedGradient, demographic parity) on one model (Logistic Regression).
* Stability noise is applied to numeric features only; tree models split at observed integer values, so for them the
  direction of the noise matters more than its size.
