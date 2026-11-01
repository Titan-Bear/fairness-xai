# Note for the Stability module

Hi! I reviewed `Project.ipynb`. The stability idea is right, but four bugs make most of the current numbers unreliable, and the notebook rebuilds the shared data and models instead of using our shared `src/` package. I checked the bugs on our real models:

- **LIME:** explaining the *same unchanged row* twice gives a Spearman of only 0.29, because no seed is set. Your LIME scores (0.04 to 0.11) mostly measure LIME's own randomness, not the noise.
- **SHAP:** `model.predict` gives hard 0/1 labels. 75% of the attributions come out exactly zero, and two runs on unchanged rows agree only 0.81.
- **Noise:** it is also added to one-hot columns (`sex_Male` becomes 1.047 or −0.035, an impossible person).
- **Spearman:** it is computed over 104 mostly-zero one-hot columns, so ties dominate.

Please paste the prompt below into your AI assistant together with our `fairness-xai` folder. It will write `src/stability.py` so your module plugs into the shared code and the dashboard.

---

## Prompt to paste

```text
You are helping me build the Stability module of a two-person Python project, "Fairness-Aware XAI Audit Dashboard". My teammate built the shared package in src/ (data, preprocessing, models, explainers). I must write ONE new file, src/stability.py, plus tests/test_stability.py, that REUSES the shared package. Do not modify any existing file in src/.

GOAL
Measure how stable SHAP and LIME explanations are when the input is slightly perturbed with Gaussian noise (1%, 5%, 10%), for 3 models x 2 datasets.

ENVIRONMENT
Python 3.11-3.13. Pinned versions (requirements.txt): pandas 2.2.3, numpy 2.1.3, scikit-learn 1.5.2, xgboost 2.1.4, shap 0.48.0, lime 0.2.0.1, scipy 1.14.1. Random seed 42 everywhere (src/config.py: SEED). Use logging, not print. Type hints and docstrings on every function. Use pathlib; no absolute paths.

SHARED API YOU MUST USE (already exists, do not rewrite):
- src.config: SEED=42, DATASETS=("adult","compas"), MODEL_NAMES=("logistic_regression","random_forest","xgboost"), SETTINGS={"with_sensitive": True, "without_sensitive": False}, LIME_NUM_SAMPLES=1000
- src.preprocess.prepare_dataset(name, save=False) -> (loaded, split)
    loaded.metadata has .numeric (list of numeric column names), .categorical, .sensitive_attributes, .privileged
    split has X_train, X_test (raw DataFrames, original row ids as index), y_train, y_test, s_train, s_test (sensitive labels)
- src.models.load_model(dataset, setting, model_name) -> fitted sklearn Pipeline with steps "preprocessor" (ColumnTransformer) and "model"; models are saved in models/ by `python -m src.run_all`
- src.explain.shap_values(pipeline, X_train, X_explain) -> np.ndarray (n_rows, n_transformed_columns), positive class. Uses TreeExplainer for RF/XGBoost and LinearExplainer for Logistic Regression, so it is deterministic. Takes RAW DataFrames and transforms internally.
- src.explain.lime_values(pipeline, X_train, X_explain, num_samples, seed) -> (weights (n_rows, n_transformed), fidelity (n_rows,), local_pred (n_rows,)). It builds a new LimeTabularExplainer with random_state=seed on every call, so the SAME seed gives the SAME random neighbourhood. Takes RAW DataFrames.
- src.preprocess.map_transformed_to_original(pipeline.named_steps["preprocessor"]) -> dict {"sex_Male": "sex", "age": "age", ...}
- src.preprocess.aggregate_by_original(attributions, transformed_names, mapping) -> DataFrame (n_rows, n_original_features) of ABSOLUTE attributions summed per original feature
- transformed names: list(pipeline.named_steps["preprocessor"].get_feature_names_out())

BUGS IN MY OLD NOTEBOOK THAT THE NEW CODE MUST AVOID
1. LIME without a fixed seed: the original and perturbed explanations used different random neighbourhoods, so even zero noise gave Spearman ~0.3. FIX: call lime_values for the original and the perturbed rows with the SAME seed (config.SEED).
2. SHAP on model.predict (hard 0/1 labels) with a permutation explainer: 75% of values were exactly 0 and results were random. FIX: use src.explain.shap_values (probabilities, exact Tree/Linear SHAP).
3. Noise added to every transformed column, including one-hot columns, which created impossible people (sex_Male = 1.047). FIX: perturb ONLY the numeric columns (metadata.numeric) of the RAW DataFrame: x_num + N(0, (level * std_train)^2), where std_train is that column's standard deviation on X_train. Leave categorical columns unchanged. Use np.random.default_rng(SEED).
4. Spearman over ~100 mostly-zero one-hot columns, with abs() for SHAP but signed values for LIME. FIX: for BOTH methods, aggregate to original features with aggregate_by_original (absolute values) and compare those vectors.
Also: do not rebuild the explainer inside a per-row loop; explain all sampled rows in one call.

REQUIRED INTERFACE (exact name and first four parameters; extra keyword arguments must have defaults):

def run_stability(
    dataset_name: str,
    model_name: str,
    xai_method: str,                       # "shap" or "lime"
    perturbation_levels: tuple[float, ...] = (0.01, 0.05, 0.10),
    setting: str = "with_sensitive",
    n_instances: int = 100,
    num_samples: int = config.LIME_NUM_SAMPLES,
    seed: int = config.SEED,
) -> pd.DataFrame:

Behaviour:
- Load data with prepare_dataset(dataset_name, save=False) and the model with load_model(dataset_name, setting, model_name). Raise ValueError for an unknown xai_method.
- Sample n_instances rows from split.X_test with np.random.default_rng(seed) (all rows if fewer).
- Explain the original rows once. For each level, perturb the numeric columns as in bug fix 3 and explain the perturbed rows (same seed for LIME).
- Per instance and level compute:
    spearman_rho  = scipy.stats.spearmanr(original_vector, perturbed_vector).statistic over the original-feature vectors (NaN if a vector is constant)
    jaccard_top5  = |top5(original) & top5(perturbed)| / |top5(original) | top5(perturbed)|
    pred_change   = |p_perturbed - p_original| of pipeline.predict_proba(...)[:, 1]
- Return a tidy DataFrame, one row per (instance, level), with columns:
  dataset, setting, model, method, noise_level, instance_id, spearman_rho, jaccard_top5, pred_change,
  plus one column per sensitive attribute, named group_<attribute>, taken from split.s_test, so stability can later be compared between demographic groups.
- Add summarise_stability(df) -> DataFrame: mean, std and count of spearman_rho and jaccard_top5 grouped by dataset, setting, model, method, noise_level.
- Add a main() for `python -m src.stability --quick` that runs every dataset x model x method on the default levels, logs the summary table, and writes results/tables/stability.csv (per instance) and results/tables/stability_summary.csv. --quick uses n_instances=10 and num_samples=200 and writes to results/quick/tables/ by calling config.set_results_dir(config.PROJECT_ROOT / "results" / "quick") first. Use config.table_path("stability.csv") for paths.

TESTS (tests/test_stability.py, pytest; models must exist in models/):
1. Zero noise is perfectly stable: run_stability("compas", "random_forest", "shap", perturbation_levels=(0.0,), n_instances=5) gives spearman_rho == 1 and jaccard_top5 == 1 for every row. Same for "lime" with num_samples=200. (This test catches bugs 1 and 2.)
2. Output has n_instances x len(levels) rows, the required columns, spearman_rho in [-1, 1], jaccard_top5 in [0, 1], pred_change >= 0.
3. Categorical columns are never changed by the perturbation function, and numeric columns are changed when level > 0.
4. Same seed gives identical DataFrames on two calls.
5. Unknown xai_method raises ValueError.

DELIVERABLE
Give the complete code of src/stability.py and tests/test_stability.py (no "...", no TODOs), explain each function in two or three sentences so I can defend it in a viva, and tell me the commands to run:
  .\.venv\Scripts\python -m pytest tests/test_stability.py -q
  .\.venv\Scripts\python -m src.stability --quick
```
