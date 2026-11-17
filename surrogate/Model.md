# PatchAttention: the custom surrogate model

`PatchAttention` is the project's custom surrogate. It predicts four ageing
metrics (capacity, plating, SEI and resistance) at the training ageing cycles
from the formation voltage and current trajectories. It is a small
PatchTST-style Transformer. Unlike PatchTST, voltage and current share each
patch token, so there is one token stream rather than one per channel.

| | |
|---|---|
| Code | `surrogate/estimators/custom_estimator.py` (`PatchAttention`, `CustomEstimator`) |
| Configuration | `inputs/models/custom.toml` |
| Training loop | `surrogate/estimators/torch_estimator.py` (shared with the GRU) |
| Data pipeline | `surrogate/estimators/torch_data.py` |
| Train | `python -m surrogate.train_models --model custom` |
| Use in the MCO | `surrogate_model = "custom"` in `inputs/mco.toml` |
| Artifacts | `surrogate/models/custom/` (`custom.pt`, `metadata.json`, `cv_predictions.csv`) |

## Inputs and outputs

- **Input:** one formation protocol, given as 256 voltage samples and 256
  current samples, evenly spaced in time over the formation
  (`TrajectoryFeaturizer`). The tensor shape is `(batch, 256, 2)`.
- **Output:** all 4 targets at every cycle of `training_ageing_mesh`
  (`inputs/surrogate.toml`). The tensor shape is `(batch, n_cycles, 4)`.

`ScaledModel` (in `torch_estimator.py`) wraps the network. It standardises
each input channel and each target using statistics from the training
protocols only, so the network always works in standardised units. Predictions
are converted back to physical units afterwards.

## Architecture

The shapes below are for the tuned configuration: `sequence_stride = 1`,
`patch_len = 8`, `d_model = 32`, `n_heads = 2`, `ffn_size = 64`,
`n_blocks = 2`, and the current mesh of 2 cycles, `[1, 500]`.

```text
formation trajectory          (B, 256, 2)      voltage, current; standardised
│
├─ 1. input reduction          (B, 256, 2)      average pool by sequence_stride (1 = off)
├─ 2. patching                 (B, 32, 16)      32 patches of 8 steps × 2 channels
├─ 3. patch embedding          (B, 32, 32)      Linear(16 → d_model) + learnable position
├─ 4. encoder block × 2        (B, 32, 32)      pre-norm: x + MHA(LN x); x + FFN(LN x)
├─    final LayerNorm          (B, 32, 32)
├─ 5. flatten head             (B, 1024)        dropout
└─    Linear(1024 → n_cycles·4) → reshape       (B, n_cycles, 4)
```

### 1. Input reduction (optional Haar approximation)

`sequence_stride` average-pools the sequence before patching; the pooling
happens in `formation_sequences`. A one-level Haar decomposition splits a
signal into approximation and detail coefficients:

$$a_k = \frac{x_{2k} + x_{2k+1}}{\sqrt{2}}, \qquad d_k = \frac{x_{2k} - x_{2k+1}}{\sqrt{2}}$$

Keeping only $a_k$ is the same as stride-2 average pooling scaled by
$\sqrt{2}$. Standardisation absorbs that constant factor, so
`sequence_stride = 2` gives exactly a one-level Haar approximation with the
detail coefficients dropped. Stride 4 is two levels, and so on.

**At present it works best without reduction (`sequence_stride = 1`).** At 256
steps the sequence is still short enough for attention. A likely reason is
that the fine detail pooling smooths away carries information about plating
and SEI, but this has not been tested in isolation.

### 2. Patching

The sequence is cut into non-overlapping patches of `patch_len` steps.
**Nothing is padded.** `256 / sequence_stride` must be divisible by
`patch_len`; otherwise building the model fails with a clear error.

| `sequence_stride` | steps | `patch_len = 8` | `patch_len = 16` |
|---|---|---|---|
| 1 | 256 | 32 patches | 16 patches |
| 2 | 128 | 16 patches | 8 patches |
| 4 | 64 | 8 patches | 4 patches |

### 3. Channel-mixed patch embedding

Each token is one patch of both channels, flattened (`patch_len × 2` values,
alternating voltage and current), and one linear layer maps it to `d_model`.
A learnable positional embedding (one vector per patch position, initialised
with a truncated normal, σ = 0.02) is added, followed by dropout.

Putting both channels in one token lets attention relate voltage and current
of the same time window directly. PatchTST's channel-independent design
cannot do that.

### 4. Pre-norm Transformer encoder

There are `n_blocks` standard `nn.TransformerEncoderLayer` blocks with
`norm_first=True`:

$$x \leftarrow x + \mathrm{MHA}(\mathrm{LN}(x)), \qquad x \leftarrow x + \mathrm{FFN}(\mathrm{LN}(x))$$

The feed-forward network is `Linear(d_model → ffn_size) → GELU → Linear(ffn_size → d_model)`.
Dropout is applied inside both sub-layers. Pre-norm leaves the residual
stream un-normalised, so a final LayerNorm follows the last block.
`n_heads` must divide `d_model`.

### 5. Flatten head: all cycles in one step

All patch tokens are flattened (`n_patches × d_model`), with no pooling over
tokens, so each patch position keeps its own weights. One linear layer then
predicts every target at every training cycle at once. Its size is
`n_patches · d_model · n_cycles · 4` weights plus biases, which makes it most
of the model. A denser `training_ageing_mesh` grows the head proportionally.

## Cycle handling

The head predicts a **fixed cycle grid**: the sorted unique `ageing_cycle`
values of the dataset, which are the `training_ageing_mesh`.

- During training, `CustomEstimator.train()` stores this grid in
  `params["cycle_grid"]`. It is saved in `custom.pt` and restored by
  `load_final_model()`, so the head size always matches the saved weights.
- `forward()` ignores the cycle values it receives. It only checks that their
  count equals the grid length.
- `predict()`, used by the MCO and `validate_protocol`, predicts the whole
  grid and then, for each requested cycle:
  - **on the grid:** returns the grid value exactly;
  - **between grid cycles:** interpolates linearly in `log(1 + cycle)`;
  - **outside the grid:** clamps to the nearest end and logs a warning.

With the current mesh `[1, 500]` and `prediction_cycles = [1, 500]`, the MCO
uses the two predictions directly, with no interpolation.

## Training

Training uses the `TorchEstimator` loop shared with the GRU:

- **Protocol-level 7-fold cross-validation** (`[split]` in
  `surrogate.toml`). All cycles of a protocol stay in one split. Within the
  non-test protocols, 18 % are held out for early stopping.
- **Standardisation** is fitted per fold on training protocols only.
- **Loss:** MSE on standardised targets. Missing (NaN) targets are masked out.
- **Optimisation:** AdamW, gradient clipping, optional input noise, and early
  stopping on validation loss. The weights from the best epoch are restored.
- **Final model:** after cross-validation, one model is refitted on all
  protocols, still holding out a validation share for early stopping. This is
  the model the MCO uses. No fold models are ensembled or averaged.

Fold metrics and their mean go to `formation_system.log`. Out-of-fold test
predictions go to `surrogate/models/custom/cv_predictions.csv`.

## Configuration (`inputs/models/custom.toml`)

| Key | Tuned | Meaning |
|---|---|---|
| `sequence_stride` | 1 | Average-pool factor; 2 = one-level Haar approximation |
| `patch_len` | 8 | Steps per patch; `256 / sequence_stride` must divide by it |
| `d_model` | 32 | Token width |
| `n_heads` | 2 | Attention heads; must divide `d_model` |
| `ffn_size` | 64 | Feed-forward width per block |
| `n_blocks` | 2 | Encoder blocks |
| `dropout` | 0.05 | Embedding, attention, FFN and head |
| `input_noise_std` | 0.0 | Gaussian input noise during training |
| `learning_rate` | 1e-3 | AdamW |
| `weight_decay` | 1e-4 | AdamW |
| `batch_size` | 16 | Protocols per batch |
| `max_epochs` / `patience` / `min_delta` | 300 / 50 / 1e-4 | Early stopping |
| `grad_clip_norm` | 1.0 | Max gradient norm |
| `seed` | 1 | Weight initialisation and batch order (folds use `[split] random_state`) |
| `device` | `"cpu"` | `"cpu"`, `"mps"` or `"cuda"` |

Any key left out falls back to `CustomEstimator.DEFAULTS`, then to
`DEFAULT_TRAINING`. Changing these keys only needs retraining, not a new
dataset.

## Tuning history

The table shows the mean over 7 folds of test MAE. The GRU is shown for
reference.

| Run | capacity | plating | sei | resistance | Notes |
|---|---|---|---|---|---|
| GRU (`gru.toml`) | 0.0091 | 0.0075 | 3.8e-4 | 3.1e-4 | baseline |
| stride 2, patch 16, d 32, dropout 0.2, noise 0.02, wd 1e-3 | 0.0095 | 0.0093 | 5.5e-4 | 2.7e-4 | train ≈ test: underfitting |
| stride 2, patch 8, d 64, dropout 0.05, noise 0, wd 1e-4, patience 50 | 0.0066 | 0.0060 | 3.3e-4 | 1.8e-4 | regularisation loosened |
| same, d 32, seed 1 | 0.0068 | 0.0059 | 3.0e-4 | 1.6e-4 | same accuracy, half-size head |
| same, 3 blocks, dropout 0.1 | 0.0073 | 0.0071 | 3.2e-4 | 1.9e-4 | worse on every target |
| **stride 1**, patch 8, d 32, 2 blocks, dropout 0.05 | — | — | — | — | current; best observed |

What these runs showed:

- When train and test error are almost equal, the model is underfitting.
  Reduce regularisation (dropout, weight decay, input noise) before adding
  capacity.
- Doubling `d_model` from 32 to 64 brought no gain. It mostly enlarges the
  flatten head.
- Adding depth (3 blocks) hurt on about 128 protocols.
- Several runs changed more than one setting at once, and seeds vary. To
  confirm a gain, change one key at a time and repeat with 2–3 seeds.

## Limitations

- **Fixed cycle grid.** The model predicts only the training cycles. Any
  other cycle is interpolated in log-cycle, or clamped if it lies outside the
  grid. If the mesh changes, the model must be retrained; its `cycle_grid` is
  read from the dataset.
- **Small dataset** (about 128 protocols). Results vary noticeably with the
  seed, so compare configurations over several seeds.
- **The sequence length is fixed** by `trajectory_feature_count`. Changing it
  requires a new dataset and retraining. `SurrogatePredictor` refuses a model
  trained on different feature columns.
