# Neural ageing surrogates (AMS-CeNN and GRU)

This branch adds neural surrogates to the formation recommendation system. The MCO, the PyBaMM
simulation code, the featurizer and `run_recommendation.py` are unchanged. The XGBoost models in
`surrogate/models` remain the default, so the repository behaves as before until you change one line.

## Delivered models

The two delivered models are AMS-CeNN models, one per task:

| Task | Folder | Model |
|---|---|---|
| Formation surrogate, called by the MCO | `surrogate/models_ams_cenn_tuned` | AMS-CeNN, tuned, certified contraction |
| Ageing forecaster, used after cycling has started | `ageing_forecast/models_ams_cenn_forecast_tuned` | AMS-CeNN, tuned, certified contraction |

Why AMS-CeNN: it is the model named in the project proposal, its cellular layers have a certified
contraction property, and the cellular architecture is intended for hardware implementation. On the
formation task it reaches the accuracy of the GRU with about one sixth of the parameters.

The GRU models and the original XGBoost models are included as reference models, so that the
comparison can be repeated. Where a GRU is better, the tables below say so.

## Use a neural surrogate in the MCO

In `inputs/surrogate.toml`, point `models_dir` at another model folder:

```toml
[paths]
models_dir = "surrogate/models_ams_cenn_tuned"   # AMS-CeNN, certified
# models_dir = "surrogate/models_gru"            # GRU
# models_dir = "surrogate/models"                # XGBoost (default)
```

Then run the recommendation as usual:

```bash
python run_recommendation.py
```

`SurrogatePredictor` reads `metadata.json` in the model folder and loads the matching backend. The
interface the MCO uses is the same for all models:

```python
predictor = SurrogatePredictor(config.surrogate, config)
predictor.available()
predictor.predict(features, [1, 500])   # DataFrame: ageing_cycle, capacity, plating, sei, resistance
```

The neural models need `torch`, which is already in `requirements.txt`. They run on CPU; no GPU is
needed for prediction. The shipped weights were checked with torch 2.2.0 (the pinned version).

## Shipped models

| Folder | Model | Certified contraction | Members |
|---|---|---|---|
| `surrogate/models` | XGBoost (original) | no | 4 regressors |
| `surrogate/models_ams_cenn_tuned` | AMS-CeNN, tuned | yes | 3 seeds |
| `surrogate/models_gru` | GRU | no | 3 seeds |

A neural model folder holds three networks trained with different seeds; the prediction is their mean.
"Certified contraction" means that the state update of every cellular layer is a contraction by
construction (norm cap on the feedback template and a bounded retention gate). It is a property of the
cell's state update. It is not a bound on the prediction error.

## Cross-validated accuracy

Protocol-grouped cross-validation on `surrogate/dataset.csv` (96 protocols, 15 ageing cycles each):
5 folds, 3 seeds, a held-out protocol is never seen in training or early stopping. The error is the
RMSE in percent of each target's range, averaged over the four targets. "Worst protocol" is the largest
per-protocol error. "Rank" is the Spearman correlation between predicted and true capacity at cycle 500.

| Model | Certified | nRMSE (%) | Worst protocol (%) | Rank | Parameters |
|---|---|---|---|---|---|
| AMS-CeNN, tuned | yes | 1.09 | 5.1 | 0.95 | 42,377 |
| GRU | no | 1.09 | 6.4 | 0.97 | 250,116 |
| GRU, tuned | no | 1.18 | 7.3 | 0.97 | 38,660 |
| AMS-CeNN, patch embedding and MLP read-out | yes | 2.27 | 7.7 | 0.92 | 15,976 |
| AMS-CeNN, single scale, no linear shortcut | yes | 2.91 | 7.7 | 0.89 | 8,201 |
| MLP | no | 4.27 | 28.7 | 0.93 | 295,812 |
| XGBoost (original settings) | no | 5.57 | 34.5 | 0.96 | |
| AMS-CeNN, paper configuration (with linear shortcut) | yes | 9.53 | 50.9 | 0.80 | 11,917 |
| Linear | no | 9.95 | 50.5 | 0.84 | 4,100 |

The full table with all variants is in `surrogate/neural_cv/summary.csv`.

How to read it:

- The tuned AMS-CeNN and the GRU have the same error. The difference between them is smaller than the
  spread between seeds, so they should be treated as equally accurate.
- Both have about one fifth of the XGBoost error, and their worst protocol is about six times better.
- XGBoost and the GRU order the candidates slightly better than AMS-CeNN (rank 0.96 to 0.97 against 0.95).
- The paper configuration of AMS-CeNN does not work on this task: with the linear shortcut it stays at
  the error of a linear model. The tuned model has no linear shortcut.
Limits of this comparison:

- XGBoost uses the original settings and was not tuned, and it was cross-validated with one seed. The
  neural models use three seeds, and two of them were tuned. A tuned XGBoost may be closer.
- The tuned models come from a random search (30 trials for AMS-CeNN, 20 for the GRU). A trial was
  selected by its error on validation protocols, and the selected configuration was then
  cross-validated like every other model. The search was not nested: it used the same 96 protocols as
  the cross-validation. The "tuned" rows are therefore somewhat optimistic compared with the others.
- All numbers are for the 96 protocols of this dataset. Protocols outside its range were not tested.

The tuned configurations are in `surrogate/neural/tuned_arms.json`, and each model folder stores its
own architecture in `metadata.json`.

## End-to-end check through the MCO

`run_recommendation.py` was run once per surrogate on the grid in `inputs/mco.toml` (72 candidates,
PyBaMM 26.6 formation simulation, then the surrogate). The predictions at cycle 1 and cycle 500 were
compared with PyBaMM ageing simulations of the same 72 candidates (`ageing_forecast/data`). 18 of the
72 candidates are in `surrogate/dataset.csv`; the other 54 were not seen in training.

| Surrogate | Capacity nRMSE (%) | Plating nRMSE (%) | Rank, final capacity | Rank, capacity fade | Rank, plating growth |
|---|---|---|---|---|---|
| GRU | 1.86 | 1.67 | 0.99 | 0.96 | 0.98 |
| AMS-CeNN, tuned | 1.99 | 1.65 | 0.96 | 0.87 | 0.80 |
| XGBoost | 5.85 | 5.62 | 0.93 | 0.88 | 0.06 |

"Rank" is the Spearman correlation with the PyBaMM value over the 72 candidates. Capacity fade and
plating growth are the differences between cycle 500 and cycle 1 that the MCO scores.

- The neural surrogates have about one third of the XGBoost error on these candidates.
- XGBoost does not order the candidates by plating growth (rank 0.06). The GRU does (0.98), and
  AMS-CeNN is in between (0.80).
- The MCO ranking changes with the surrogate. The scores from the two neural surrogates agree with each
  other (rank correlation 0.82) and disagree with the XGBoost scores (-0.49 and -0.40). The top four
  candidates are different for each surrogate.
- The GRU orders the candidates best, mainly by plating growth. AMS-CeNN has the same error and is the
  delivered model; the GRU is available as a cross-check of a recommendation.

The reference ageing simulations come from a separate run. On the 18 shared candidates they agree with
`dataset.csv` within 0.002 Ah.

## Sensitivity to the PyBaMM version

`pybamm` is not pinned in `requirements.txt`. A formation run with PyBaMM 26.6 gives a trace that
differs from the stored dataset trace at the first and the last sample (first current sample 0 A
instead of -1.25 A, last voltage sample 2.70 V instead of 2.755 V). For the protocol
`x1=0.5|x2=0.5|x3=0.1|x4=1`, which is in the training data:

| Cycle 1 | Capacity (Ah) | Plating |
|---|---|---|
| True value | 3.736 | 0.028 |
| XGBoost, stored trace | 3.736 | 0.028 |
| XGBoost, fresh PyBaMM 26.6 trace | 3.711 | 0.053 |
| AMS-CeNN, stored trace | 3.750 | 0.027 |
| AMS-CeNN, fresh PyBaMM 26.6 trace | 3.750 | 0.027 |
| GRU, stored trace | 3.745 | 0.029 |
| GRU, fresh PyBaMM 26.6 trace | 3.745 | 0.029 |

On this protocol the XGBoost prediction moves when two samples of the trace change, and the neural
predictions do not. This was checked on one protocol only. If the MCO shows deviations between the
surrogate and PyBaMM, the PyBaMM version used to create the dataset is worth checking first.

## Compare the models on your own points

```bash
python -m surrogate.compare_models                               # list models, run the reload check
python -m surrogate.compare_models --dataset points.csv --out table.csv
```

`points.csv` has the format of `surrogate/dataset.csv` (feature columns, `ageing_cycle`, and the four
targets). The script scores every model folder on these points. The reload check recomputes a stored
reference prediction and reports the difference, which confirms that the weights load correctly on
your machine.

## Gradients

The neural models give the exact derivative of every output with respect to every input sample:

```python
jacobian, names = predictor.gradient(features, [1, 500])
# jacobian.shape == (cycles, targets, 768); names are the feature columns
```

This is not available for XGBoost.

## Ageing forecaster (after cycling has started)

A second model forecasts the rest of an ageing trajectory from the first 64 measured ageing cycles of
the four indicators. It is not called by the MCO, because it needs measured cycles. It can be used to
check a surrogate prediction once a cell is cycling.

```bash
python run_ageing_forecast.py
python run_ageing_forecast.py --models ageing_forecast/models_ams_cenn_forecast_tuned
```

```python
from ageing_forecast.forecaster import AgeingForecaster
forecaster = AgeingForecaster("ageing_forecast/models_gru_forecast_tuned")
forecast = forecaster.forecast(measured, until_cycle=500)   # measured: first 64 cycles or more
```

Cross-validation grouped over 72 formation candidates (8 folds, 3 seeds), forecast of cycles 65 to 500.
The error is in percent of each candidate's indicator range. No rollout diverged for any model.

| Folder in `ageing_forecast/` | Model | Certified | Median (%) | Worst candidate (%) | Parameters |
|---|---|---|---|---|---|
| `models_gru_forecast_tuned` | GRU, tuned | no | 0.11 | 3.12 | 30,080 |
| `models_ams_cenn_forecast_tuned` | AMS-CeNN, tuned | yes | 0.44 | 3.78 | 79,333 |
| not shipped | GRU, default | no | 0.53 | 10.05 | 282,624 |
| not shipped | Linear | no | 0.81 | 6.54 | 65,792 |
| `models_ams_cenn` | AMS-CeNN, paper configuration | yes | 1.06 | 2.76 | 89,797 |

The delivered forecaster is the tuned AMS-CeNN (`run_ageing_forecast.py` uses it by default). The tuned
GRU is more accurate on the median and is smaller on this task; it is included as a reference. The three
shipped models are close on the worst candidate. The two tuned models come from a random search on the same 72 candidates (not nested), so
their rows are somewhat optimistic. The full table is in `ageing_forecast/cv/summary.csv`.

## Retrain or extend

```bash
python -m surrogate.train_neural cv                       # cross-validation, resumable
python -m surrogate.train_neural summarize                # summary.csv and selection.json
python -m surrogate.train_neural tune --family cenn-certified --trials 0 30
python -m surrogate.train_neural tune-summarize --family cenn-certified
python -m surrogate.train_neural final --arms ams-cenn-tuned gru

python -m ageing_forecast.train cv
python -m ageing_forecast.train summarize
python -m ageing_forecast.train final --arms gru-forecast-tuned
```

Settings are in the `[neural]` section of `inputs/surrogate.toml` and in `inputs/ageing_forecast.toml`.
Training used a GPU; prediction does not need one.

## Files added or changed

| Path | Content |
|---|---|
| `surrogate/predictor.py` | changed: selects the backend from `metadata.json`; the XGBoost path is as before |
| `surrogate/neural/` | models, training, cross-validation, random search |
| `surrogate/train_neural.py`, `surrogate/compare_models.py` | command-line entry points |
| `surrogate/models_ams_cenn_tuned/`, `surrogate/models_gru/` | shipped weights |
| `surrogate/neural_cv/` | cross-validation tables |
| `ageing_forecast/`, `run_ageing_forecast.py`, `inputs/ageing_forecast.toml` | ageing forecaster |
| `inputs/surrogate.toml` | added the `[neural]` section; `models_dir` is unchanged |
