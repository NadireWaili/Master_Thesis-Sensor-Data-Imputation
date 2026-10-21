This repository consists of my and my project parter's work. For my work, please check "nadire" folder. 

Data retrival and exploratory analysis is done on data insights library using SQL, which can be seen in the SQL folder. 

Most mature models: 
1. For VAE model, please check "VAE_All_final_Mcar.py" file;
2. For GP-VAE model, please check "GP_VAE_Pypots.py" file;
3. For SAITS model, please check "SAITS.py";
4. For Random Forest, Gradient Boosting, and Linear Regression Models, please check "Simplemodelsfortargetcol" file.

In the VAEs file folder, I experimented different version of VAE model, such as redesigning loss function and input data for the model training to make it specialized for one target column. 

For GP-VAE model, though the final applied one was from Pypot library, I also tried to architect it on top of the VAE model. 


Other files contained in my folder are the model attempts and model revision records that consist of different architecture of models. 



# Sensor data imputation for engine time series

Master thesis code for missing-value imputation on two-stroke marine engine sensor data. The repository trains single-sensor regressors and multivariate imputers on engine time series, scores them on artificially masked positions, and carries the filled series into specific fuel oil consumption (SFOC) and the performance index.

The Python package name is `sensor-imputation-thesis`. The license is MIT, copyright 2025 Nadire Waili.

This tree is research code. Scripts and notebooks train as soon as they run, paths point at an Amazon SageMaker home directory, and the sensor data and model weights stay outside git.

## Who wrote what

`nadire/` is Nadire Waili's work. `han/` is the project partner's work. `pyproject.toml` lists Christoffer Sommerlund as the package author. Shared loaders, the Babayaga reference-curve wrapper, and the MLflow notes live under `shared/`.

## What to open first

The short list of mature experiments, using the filenames as they exist in the tree:

| Question | File |
| --- | --- |
| How is the comparison table built? | [`nadire/Dataload&SQL/SQL_load_data_for_models.py`](src/sensor_imputation_thesis/nadire/Dataload&SQL/SQL_load_data_for_models.py) |
| Which sensors are missing, and on which engines? | [`han/data_availablility_of_top9_columns.py`](src/sensor_imputation_thesis/han/data_availablility_of_top9_columns.py) and [`han/EDA_data_availability.ipynb`](src/sensor_imputation_thesis/han/EDA_data_availability.ipynb) |
| Linear regression, random forest, and gradient boosting for one target column | [`nadire/BasicModels/SimpleModelsfortargetcol.ipynb`](src/sensor_imputation_thesis/nadire/BasicModels/SimpleModelsfortargetcol.ipynb) |
| Multivariate VAE, evaluated under MCAR | [`nadire/FinalImputationModels/VAE_ALL_final_MCAR.py`](src/sensor_imputation_thesis/nadire/FinalImputationModels/VAE_ALL_final_MCAR.py) |
| GP-VAE from PyPOTS | [`nadire/FinalImputationModels/GP_VAE_Pypots.py`](src/sensor_imputation_thesis/nadire/FinalImputationModels/GP_VAE_Pypots.py) |
| SAITS from PyPOTS | [`nadire/FinalImputationModels/SAITS.py`](src/sensor_imputation_thesis/nadire/FinalImputationModels/SAITS.py) |
| Plots, metric tables, SFOC before and after imputation | [`nadire/Visualizations.ipynb`](src/sensor_imputation_thesis/nadire/Visualizations.ipynb) |

Older VAE revisions, TabPFN, and the first scikit-learn notebooks are catalogued below. They are the record of how the files in the table were reached.

## Question the models answer

Engine time series in this project come from the internal `timeseries` table, joined to ship and engine metadata. Many sensors that the performance calculations need are missing on some engines, some cylinders, or some hours. Two imputation problems are set up in the code.

**One column at a time.** Choose a target such as barometric pressure (`pr_baro`), scavenge-receiver air temperature (`te_air_scav_rec`), or turbine outlet pressure (`pr_exh_turb_out`). Fit a regressor on the other columns. The chronological hold-out scores MAE, MSE, RMSE, and R² on rows where the target was observed.

**Many columns at once.** Reconstruct a window of sensors together. A variational autoencoder, a Gaussian-process VAE, and SAITS are trained for this. Because the real missing entries have no ground truth, evaluation adds a missing-completely-at-random (MCAR) mask on top of values that were observed, then scores only those held-out positions. The usual rate in the final scripts is 30 percent.

Filled frames are then available to the SFOC calculation (`uc286core`) and the performance index (`uc90core`). Those calls need the internal reference-curve service (Babayaga) and the analytics clients. The example scripts are `shared/uc90_example.py`, `shared/uc220_example.py`, and `nadire/uc220_example.py`.

## Layout

```text
src/sensor_imputation_thesis/
├── shared/                  loaders, Babayaga wrapper, MLflow notes, use-case examples
├── han/                     partner: availability, sklearn pipelines, XGBoost, TabPFN, a TensorFlow VAE
└── nadire/
    ├── Dataload&SQL/        DuckDB queries that build the modeling frames
    ├── BasicModels/         scikit-learn experiments, including the single-column comparison
    ├── VAEs/                single-target VAE revisions
    ├── FinalImputationModels/
    └── Visualizations.ipynb
```

Import the package as `sensor_imputation_thesis`. Several scripts do `import sensor_imputation_thesis.shared.load_data as load`. Put `src` on `PYTHONPATH` (the package directory is `src/sensor_imputation_thesis`). `pyproject.toml` does not declare a package include, so a plain install may not put that import on the path by itself.

## Data

Sensor tables are not in the repository. `.gitignore` ignores `*.parquet`, `*.pq`, `*.csv`, `*.json`, `*.txt`, `*.pth`, `*.pypots`, `**/mlruns/**`, and `**/mlartifacts/**`. Caches, metric tables, imputed frames, and checkpoints stay on the machine that ran the training.

Queries go through `data_insight.setup_duckdb()` and read `timeseries` and `shipinfo`. A typical query keeps rows for one product id between two timestamps. Some queries also require `in_stable` and `in_reference_mode` to be non-null.

`SQL_load_data_for_models.py` writes two frames used later. Both are filtered to rows with `fr_eng` greater than `10/60`, which the comments call the engine-running cut. Both are cached under `~/SageMaker/data_cache` and keyed by an MD5 of the tag list, the product id, and the date window.

| Output (under `nadire/`, on SageMaker) | Window | What it is |
| --- | --- | --- |
| `df_for_presentation.parquet` | 2023-11-01 to 2024-10-31 | Shorter tag list for the presentation runs |
| `dffinal_for_comparison.parquet` | 2019-10-01 to 2024-10-01 | Wider tag list: SFOC inputs, high-importance sensors, and a one-hot column for engine type |

The product id in that loader is `ea52565f40ed312a2f3a18071998ce0a`. The comment on that id says the extract is for an ME-LGIM engine. Other scripts hardcode other product ids. `sql.py` lists ME engine types from `shipinfo` and keeps the product ids that also appear in `timeseries`, then writes `MEpidOverlapped_df.parquet`.

`han/data_availablility_of_top9_columns.py` walks engines and records the missing-value percentage of the sensors that the availability study treats as the worst-covered group. `han/EDA_data_availability.ipynb` plots mean and standard deviation of availability for those families.

The three final model files do not all read the same parquet in the committed source:

- `VAE_ALL_final_MCAR.py` and `SAITS.py` load `dffinal_for_comparison.parquet`.
- `GP_VAE_Pypots.py` loads `df_for_presentation.parquet`. The line that would load `dffinal_for_comparison.parquet` is commented, next to a note that the comparison frame has 119,439 rows and was the file shared with SAITS and the VAE. Turn that line back on when the three models should see the same extract. The GP-VAE imputation block at the bottom of the file also reads `df_for_presentation.parquet`.

`SAITS.py` builds a one-in-three subsample called `df1` (the comment describes it as roughly 3-minute sampling of 1-minute data). Scaling and training use `df`, the full frame. The subsample is not what enters SAITS.

## Sensor names used in the code

Indexed tags end in `__0`, `__1`, and so on. That index is a cylinder or a turbocharger. `_iso` tags are the series passed into the reference-curve and performance calculations. `_mv` tags are the mean-value versions. Names below are the ones written next to the tags in the EDA notebook and the loaders.

| Tag | Name used in the notebooks |
| --- | --- |
| `fr_eng`, `fr_eng_ecs`, `fr_eng_setpoint` | Engine speed, the ECS speed, and the speed setpoint. Running rows are `fr_eng > 10/60`. |
| `re_eng_load`, `re_eng_load_estimate_ecs` | Engine load, and the ECS load estimate |
| `pr_baro` | Barometric pressure |
| `te_seawater` | Seawater temperature |
| `pd_air_ic__*` | Pressure drop of air across the scavenge air cooler |
| `te_air_ic_out__*` | Scavenge air cooler, air outlet temperature |
| `te_air_comp_in_a__*`, `te_air_comp_in_b__*` | Turbocharger air intake temperature, two sensors |
| `fr_tc__*` | Turbocharger speed |
| `pr_exh_turb_out__*` | Turbine back pressure |
| `te_exh_turb_in__*`, `te_exh_turb_out__*` | Turbine inlet and outlet temperature |
| `te_exh_cyl_out__*` | Exhaust gas temperature after the exhaust valve |
| `pr_exh_rec` | Exhaust receiver pressure |
| `pr_air_scav`, `pr_air_scav_ecs`, `pr_air_scav_iso` | Scavenge air pressure |
| `te_air_scav_rec`, `te_air_scav_rec_iso` | Scavenge receiver air temperature |
| `pr_cyl_max__*`, `pr_cyl_comp__*` | Maximum cylinder pressure, compression pressure |
| `se_mip__*` | Mean indicated pressure |
| `in_stable`, `in_reference_mode` | Operating-mode flags used as SQL filters |

`shared/baba_wrapper.py` maps reference-mode integers onto fuel and tier names (`HFO_TII`, `DF_TIII`, and the others) and maps ISO operating-point tags onto the names `uc90core` expects (`reference_curve_pmax_control` for `pr_cyl_max_mv_iso`, and the matching curves for turbine inlet temperature, turbocharger speed, scavenge pressure, and compression pressure).

## Single-column models

[`SimpleModelsfortargetcol.ipynb`](src/sensor_imputation_thesis/nadire/BasicModels/SimpleModelsfortargetcol.ipynb) is the comparison for one target. It reads `df_for_simplemodels.parquet`. The notebook states that the first two targets use product id `cdb73a020c59f7745d85e04d85bb8521`.

For each target the split is chronological, 80 percent train and 20 percent test. Features are standardized with `StandardScaler`. Three estimators are fit, each twice: once on every other column, and once on a reduced set taken from feature importance.

1. `pr_baro`, with linear regression, random forest, gradient boosting, and a grid search on the reduced gradient-boosting model.
2. `te_air_scav_rec`, with the same three estimators.
3. `pr_exh_turb_out`.

Runs are logged to the MLflow experiment `SimpleModelsforTargetCols` at `http://localhost:5000`. Metrics are MAE, MSE, RMSE, and R².

The gradient-boosting fits in that notebook use 100 trees, learning rate 0.1, and max depth 3 unless the grid search overwrites them. The grid on the important-feature model for `pr_baro` searches `n_estimators` in `{100, 200}`, `learning_rate` in `{0.05, 0.1, 0.2}`, and `max_depth` in `{3, 4, 5}`, on a 20 percent sample of the training rows.

[`han/feature_importance.ipynb`](src/sensor_imputation_thesis/han/feature_importance.ipynb) ranks predictors with `XGBRegressor` for nine targets (`te_exh_cyl_out__0`, `pd_air_ic__0`, `pr_exh_turb_out__0`, `te_air_ic_out__0`, `te_seawater`, `te_air_comp_in_b__0`, `te_air_comp_in_a__0`, `fr_tc__0`, `pr_baro`). It keeps rows where both `fr_eng` and `fr_eng_ecs` exceed `10/60`, splits 80/20 in time order, and marks features up to a cumulative importance of 0.95. XGBoost is left to handle missing values. The frame is `han/dataframe_forxgboost`, which is not in git.

### Single-target VAE revisions

`nadire/VAEs/` specializes the autoencoder for one column. The target column is replaced with the placeholder `-2` for every row. Other missing entries are filled with the same placeholder, which sits outside the scaled range of observed data. The reconstruction term of the loss is the mean squared error on that target column, plus the KL term. Scalers in the later revisions are min-max. Batch size in the revision notes is 64.

| File | Target written in the script |
| --- | --- |
| `vaeexample.py`, `vaeexample copy.py`, `vae3.py` | `te_exh_cyl_out__0`, from `9missingcolsdata.parquet` |
| `vaereviseddiffeimputation copy.py`, `pr_cyl_max__vaerevised.py` | `pr_cyl_max__0` |
| `FInal_te_air_scav_recvae.py`, `multilinear(VAE)forte_air_scav_rec.py`, `VAErescue.py` | `te_air_scav_rec` |
| `vae2.0.py` | Earlier preprocessing on `9missingcolsdata.parquet`. The helper fills one chosen target column with `-2`. |
| `vae.ipynb` | Long notebook: tag selection for the use-case score and SFOC, the placeholder scheme, training, density plots, and the performance-score and SFOC cells |

`nadire/FinalImputationModels/GP-VAE-revisedfromvae.py` and `Unmasked-GP-VAE.py` keep that single-target VAE and replace the KL term with a Gaussian-process KL. The kernel in those files is RBF. The target set in both is `te_air_scav_rec`. The model that the final comparison uses is the PyPOTS GP-VAE, described next.

## Multivariate models

All three final scripts are linear programs: imports and data loading run immediately, then an Optuna study, then a retrain, and in SAITS and GP-VAE a further imputation block that expects a weight file on disk. Point the MLflow tracking URI at a live server before starting them. Each script sets it to `http://localhost:5000`. Device selection is CUDA, then MPS, then CPU.

Windows are 20 steps long. Comments in the PyPOTS scripts say the source spacing they had in mind is one minute, and that 1,440 steps would be a day. The committed window is 20. Each feature is min-max scaled to `[-1, 1]` with a scaler fit on the training windows only. The split of windows is chronological: 60 percent train, 20 percent validation, 20 percent test.

### VAE (`VAE_ALL_final_MCAR.py`)

This is a fully connected VAE, not a sequence model.

- Encoder and decoder widths: input → 400 → latent 20 → 400 → input. The decoder ends in `tanh`.
- Rows that still contain a NaN are dropped, and columns that are entirely missing are dropped, before any artificial mask. Metrics are therefore computed on rows that were complete in the parquet.
- `mask_f` then replaces about 30 percent of entries with the placeholder `-2` (`missing_rate=0.3`, seed 42). Observed values are scaled to `[-1, 1]`. Placeholder cells stay at `-2`.
- The loss is the mean squared error on positions equal to `-2`, plus the closed-form KL to a standard normal.
- Optuna minimizes validation loss for 20 trials or 3,600 seconds. It searches batch size in `{32, 64, 96, 128}` and learning rate log-uniform from `1e-6` to `1e-2`. Epochs are fixed at 10. Gradients are clipped to global norm 1.
- The best trial is retrained on the concatenated train and validation sets. The MLflow experiment is `VAE_final_test2`, parent run `VAE_final_test_official`.
- After training, the script expects a registered model named `VAE_All_Final_with_mcar`, version `1`, stage Production. `impute_with_vae` walks rows that still contain NaN, fills those inputs with 0, decodes, and writes the reconstructed values back. Columns that were entirely missing are left out, because the network never saw them. The imputed frame is written to `DFImputedwizVAE.parquet`. The pre-imputation numeric frame is written to `df_wiz_NaNs.parquet`.

### GP-VAE (`GP_VAE_Pypots.py`)

PyPOTS `GPVAE` with a Cauchy kernel over the latent trajectory.

Training-time search space, 30 Optuna trials, objective equal to the average per-feature MAE after inverse scaling:

| Argument | Value or search range |
| --- | --- |
| `latent_size` | 37 |
| `encoder_sizes` | `(128, 128)` |
| `decoder_sizes` | `(256, 256)` |
| `kernel` | `cauchy` |
| `sigma` | 1.005 |
| `window_size` | 24 (CNN window inside the GP-VAE, separate from the 20-step data window) |
| `M`, `K` | 1 Monte Carlo sample, 1 importance weight |
| `patience` | 3 |
| `lr` | log-uniform from `1e-6` to `1e-2` |
| `epochs` | 10 to 50 |
| `batch_size` | 32, 64, 96, or 128 |
| `length_scale` | 0.5 to 5 |
| `beta` | 0.1 to 1 |

MCAR is applied with `pygrinder` at rate 0.3, and only on positions that were already observed. Prediction draws `n_sampling_times=2` and averages them. MAE, MSE, and RMSE are computed per feature on the masked positions, in the original engineering units, and MAE is also logged as a percentage of that feature's standard deviation on the masked cells. A feature is skipped when the imputation or the ground truth still contains a NaN. Experiments used in the file are `GP_VAE_Final` and `GP-VAE-Presentation`. Checkpoints are written under `nadire/gp_vae`, and the script deletes that directory at the start of a study.

The imputation block at the bottom does not rerun the search. It loads `GPVAE_weights2.pth` and writes `imputed_wizGPVAE2.parquet`. The hyperparameters stored next to that load are beta `0.6568903882476151`, 34 epochs, length scale `4.881546647934297`, batch size 128, and learning rate `0.0005404144686687628`, with the architecture in the table above.

### SAITS (`SAITS.py`)

PyPOTS `SAITS`. The self-attention stack used for the retrain and for the weight-loading imputation is:

| Argument | Value |
| --- | --- |
| `n_layers` | 4 in the retrain and in the imputation block. Optuna searches 2, 3, or 4. |
| `d_model`, `d_ffn` | 512 |
| `n_heads` | 8 |
| `d_k`, `d_v` | 64 |
| `dropout`, `attn_dropout` | 0.1 |
| `diagonal_attention_mask` | true |
| `ORT_weight`, `MIT_weight` | 1 |
| learning rate | `0.0039` in the retrain. Optuna searches log-uniform from `1e-6` to `1e-2`. |
| `epochs` | Optuna searches 10 to 20. The retrain uses the best trial's epoch count. The final imputation constructor sets epochs and patience to 1 because it only loads weights. |
| `batch_size` | Optuna searches 4 to 16. The retrain uses the best trial's batch size. |
| `patience` | 6 during search and retrain |

The study is 30 trials under the MLflow experiment `SAITS-Final`, minimizing average per-feature MAE. The same masked-position MAE, MSE, RMSE, and MAE-over-standard-deviation metrics are used as in the GP-VAE. After the study, train and validation windows are concatenated and split 90/10, the validation slice is masked again, and the model is retrained into `nadire/SAITS_Final`. The file removes that directory before training.

The last block loads `saits_weights.pth`, pads the comparison frame so the row count divides by 20, imputes, trims the padding, puts `time` back, and writes `saitsimputed.parquet`.

`nadire/Pypots_model.py` is an earlier SAITS script on a hand-listed set of cylinder and turbocharger tags from 2023-10-01 to 2024-02-01. `nadire/SAITS_synthetic.py` runs SAITS on the first 20,000 rows of `Newdataframeforpypots.parquet`, with window length 30 and an MCAR rate of 0.2.

## Metrics, plots, and SFOC

Per-feature numbers from the thesis runs are not committed. They were logged to MLflow and written to local parquet files. [`Visualizations.ipynb`](src/sensor_imputation_thesis/nadire/Visualizations.ipynb) is the place those tables are read back. It expects, all under `nadire/` on the SageMaker path:

- `dffinal_for_comparison.parquet` and `df_for_presentation.parquet`
- `DFImputedwizVAE.parquet`, `df_SAITSimputed.parquet`, `saitsimputed.parquet`, `imputed_wizGPVAE2.parquet`
- `VAE_metrics.parquet`, `GPVAE_metrics.parquet`, `GPVAE_metrics2.parquet`, `SAITS_metrics.parquet`, `final_SAITS_metrics2.parquet`
- `sfoc_beforeimpute.parquet`, `sfoc_afterimpute.parquet`, and it writes `df_combined.parquet`

The notebook draws a missingness heatmap, bars of missing rate, and MAE curves for VAE, GP-VAE, and SAITS. One cell restricts the overlay to features with partial missingness and names `pr_cyl_max_mv_iso`, `pr_cyl_comp_mv_iso`, `te_seawater`, and `pr_exh_rec`. Later cells plot SFOC before and after imputation.

`nadire/uc220_example.py` builds the inputs `ExcessSFOC` wants. Reference data it asks Babayaga for are `te_air_scav_iso`, `pr_cyl_max_mv_iso`, `mr_fuel_iso`, `tc_eff`, and `pr_exh_turb_out_iso`. `shared/uc90_example.py` builds a dummy frame and calls `PerformanceIndex` so the column contract can be checked without a full extract.

## Partner experiments in `han/`

| File | What it does |
| --- | --- |
| `EDA_data_availability.ipynb` | Availability plots by sensor family |
| `data_availablility_of_top9_columns.py` | Missing-value percentage of the priority sensors, per engine |
| `dataframe_Feb.py` | DuckDB extract for 2023-11-01 to 2024-10-31, cached under `/tmp` |
| `linear_regression_model.py` | Standardized linear regression of `fr_eng` on the other columns, 5-fold R² and MSE, after dropping rows with NaN. Product id in the file is `ced05645997a2804bde7aff0b6703b90`, September 2023 to September 2024. |
| `mlflow.ipynb` | Pipelines logged to MLflow: linear regression, polynomial regression of degree 3, random forest, gradient boosting, and a support vector machine. A later section forward-selects features for `pr_baro` and refits a forest on `fr_eng`, `pr_cyl_max__7`, `re_egb_pos`, and `te_air_ic_out__0`. The notebook also fits a SHAP explainer on the forest. Random-forest and gradient-boosting grids are in the same notebook. The linear and polynomial pipelines impute with `KNNImputer(n_neighbors=2)` before scaling. |
| `feature_importance.ipynb` | XGBoost importances, described above |
| `tabPFN_model.py`, `tabPFN_model_baseline.py` | `TabPFNRegressor` for `pr_baro` on the XGBoost frame. Comments say TabPFN's supported sample count is 10,000, so each side of an 80/20 time split is subsampled to 10 percent. Columns with a single unique value, or with more than 95 percent missing, are dropped. Remaining NaNs are removed. `tabpfn` is imported here and is not declared in `pyproject.toml`. |
| `vae_model.py` | TensorFlow VAE with a small regression head that predicts `pr_baro`. Mean imputation fills the features. TensorFlow is not declared in `pyproject.toml`. The CSV path in the file is a placeholder (`dataframe (not uploaded)`). |
| `vae_finetuned.py` | PyTorch VAE revision with MLflow logging, parallel to the early Nadire VAE scripts |

## Environment

`requires-python` is `>=3.12,<3.13`. Dependencies are declared in `pyproject.toml` for [uv](https://docs.astral.sh/uv/). The build backend is `poetry-core`. There is no `poetry.lock` or `uv.lock` in the tree. On Linux and Windows, `torch` and `torchvision` are taken from the CUDA 12.4 wheel index `https://download.pytorch.org/whl/cu124`.

Public packages the model scripts import include `numpy`, `pandas`, `pyarrow`, `fastparquet`, `duckdb` (pinned to 1.1.0), `scikit-learn`, `xgboost`, `torch`, `pypots` (at least 0.19), `optuna`, `mlflow`, `matplotlib`, and `seaborn`.

These distributions resolve from the team index, not from PyPI: `babayaga`, `calc2s`, `data-insight` (pinned to `0.1.3a2`), `uc90core`, `uc286core`, and `util2s` (kept below `0.4.98`). The pre-commit config records the index URL `https://artifactory.ceon-dev.io/artifactory/api/pypi/pypi-team-analytics/simple/`. `pyproject.toml` does not. Point uv at that index before syncing, or resolution stops on those names. Training then needs `data_insight.setup_duckdb()` to be able to read `timeseries`.

Scripts also import `pygrinder`, and some import `shap`. `han/tabPFN_model.py` imports `tabpfn`. `han/vae_model.py` imports `tensorflow`. Add those locally if you run those files.

```bash
uv python pin 3.12
uv sync --all-groups
```

`--all-groups` installs the `dev` group (`ipykernel`) and the `test` group (`pytest`, `ruff`, and the other test extras). `pyproject.toml` sets `testpaths = ["tests"]`. There is no `tests/` directory in the tree.

MLflow, from [`shared/mlflow.example/README.md`](src/sensor_imputation_thesis/shared/mlflow.example/README.md). That note was written with `poetry`; the same server command runs under uv. Training scripts expect the server on port 5000. The VAE files record an `HTTPConnectionPool(host='0.0.0.0', port=5000)` failure when nothing is listening, and they mention renaming the experiment if a stale run blocks logging.

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

`shared/mlflow.example/example_mlflow_model_logging.py` is the small logging example. Port 5000 is also what a local forward to the SageMaker notebook host used.

## Running a final script

1. Install the environment above, including the internal packages, and confirm `data_insight.setup_duckdb()` can see `timeseries`.
2. Start MLflow on port 5000.
3. From the repository root, with `src` on `PYTHONPATH`:

   ```bash
   PYTHONPATH=src uv run python "src/sensor_imputation_thesis/nadire/Dataload&SQL/SQL_load_data_for_models.py"
   ```

   Rewrite the output paths in that file if your home directory is not `/home/ec2-user/SageMaker`. Every final model script embeds that prefix.
4. Run one model file the same way, for example `VAE_ALL_final_MCAR.py`. Optuna starts immediately. SAITS and GP-VAE delete their checkpoint directories at the start of a study.
5. The imputation cells at the bottom of `SAITS.py` and `GP_VAE_Pypots.py` load `saits_weights.pth` and `GPVAE_weights2.pth`. Those files are gitignored. They have to be produced by the training section, or copied onto the path the script prints, before that block can succeed. Running the whole file top to bottom runs the study first and only then reaches the weight load.
6. Open `Visualizations.ipynb` once the metric parquet files exist next to the imputed frames.

`shared/load_data.py` is a smaller query example (a few tags, a different product id, January to December 2024) and is safe to read as the shape of a query. It still needs the internal database.

## Development hooks

`.pre-commit-config.yaml` runs `check-added-large-files` at 500 KB, TOML and JSON checks, line-ending normalization for shell scripts, Ruff lint and format (line length 120, from `pyproject.toml`), and `nbstripout` so notebook outputs are removed on commit. The config comment says to install the hook with `python -m pre-commit install`. `pre-commit` is not listed in `pyproject.toml`. Install the tool, then run its console script:

```bash
pip install pre-commit
pre-commit install
```

Some notebooks already in history still contain embedded figures, including `nadire/VAEs/vae.ipynb`, `nadire/Visualizations.ipynb`, and `han/mlflow.ipynb`. A new commit that touches them will have to pass the 500 KB hook, which those files exceed today. Strip outputs before staging them.

## File index

### `shared/`

| File | Role |
| --- | --- |
| `load_data.py` | Minimal `timeseries` query |
| `baba_wrapper.py` | Reference curves and engine rating from Babayaga |
| `uc90_example.py` | Performance index on a dummy frame |
| `uc220_example.py` | SFOC column contract and reference-curve lookup |
| `vaeexample.py` | Early shared copy of the single-target VAE |
| `tags.py`, `print_tags.py` | Tag helper. `tags.py` imports `util2s.moodel.cph_ceon_tags` (the module name is spelled that way in the file). |
| `mlflow.example/` | Server commands and a logging snippet |

### `nadire/`

| File | Role |
| --- | --- |
| `Dataload&SQL/sql.py` | ME engines that exist in both `shipinfo` and `timeseries` |
| `Dataload&SQL/SQL_load_data_for_models.py` | Presentation frame and comparison frame |
| `Dataload&SQL/hello.py` | Early look at the `engines` table |
| `BasicModels/Basicmodels.ipynb` | First scikit-learn passes: linear, random forest, and gradient boosting for `pr_hydr_out`, `re_eng_load_estimate_ecs`, `pr_air_control`, and `re_total_fuel_quality_offset`, with KNN imputation and grids |
| `BasicModels/new_models.ipynb` | Distributions, the engine-running filter, the three regressors, a VIF check, PCA into linear regression and gradient boosting for `te_exh_cyl_out__0`, then an early VAE |
| `BasicModels/SimpleModelsfortargetcol.ipynb` | Final single-column comparison |
| `VAEs/` | Single-target VAE revisions listed above |
| `FinalImputationModels/VAE_ALL_final_MCAR.py` | Multivariate VAE under MCAR |
| `FinalImputationModels/GP_VAE_Pypots.py` | PyPOTS GP-VAE |
| `FinalImputationModels/GP-VAE-revisedfromvae.py` | RBF GP-KL on the single-target VAE, target `te_air_scav_rec` |
| `FinalImputationModels/Unmasked-GP-VAE.py` | Same architecture, unmasked variant, same target |
| `FinalImputationModels/SAITS.py` | PyPOTS SAITS |
| `Pypots_model.py`, `SAITS_synthetic.py` | Earlier SAITS runs |
| `Visualizations.ipynb` | Missingness, model overlay, SFOC |
| `uc220_example.py` | SFOC on a real product id, reading `df_combined.parquet` |

### `han/`

Listed in the partner section above.

## License

MIT. See [LICENSE](LICENSE). Copyright 2025 Nadire Waili.
