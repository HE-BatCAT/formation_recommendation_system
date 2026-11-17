"""CeNN surrogate-model adapter for the SEI-rate prediction framework.

Wraps the vendored, self-contained CeNN backbone (``models/cenn_core.py``, ported verbatim
from ``neuralforecast-cenn/neuralforecast/cenn/model.py`` v0.2.0) into this framework's
``class Model(nn.Module)`` contract:  ``forward((B, window, in_features)) -> (B, out_features)``.

The 19 variant presets are ported from ``neuralforecast-cenn/experiments/runner.py``
(``_VARIANT_BASE`` + ``VARIANT_SPECS`` + ``_VARIANT_K`` + the architecture-relevant items of
``_CENN_FIXED``). Pick a variant via ``configs.cenn_variant`` (default ``"CeNN-Full"``).

Adapter design (faithful + bottleneck-free):

    x (B, window, in)  --backbone._embed-->  u (B, hidden, L')   # patch / cross-var per variant
                       --backbone.stack--->  z (B, hidden, L')   # multiscale/integrator/tau/cap/...
                       --mean over time--->  z (B, hidden)        # global temporal pooling
                       --head------------->  (B, out_features)

We reuse cenn_core's input embedding + the FULL CeNN stack (every architecturally-meaningful
variant knob) and replace ONLY cenn_core's reconstruction head (hidden -> n_features) with a
clean pool+project head (hidden -> out_features), so there is no n_features bottleneck. Global
mean-pooling over time matches the CNN baseline's ``AdaptiveAvgPool1d`` (fair comparison). The
variant's ``head_type`` ("linear"/"mlp") is honoured on THIS head, so APP-Head-MLP stays
architecturally distinct from ABL-Patch.
"""
import torch
import torch.nn as nn

from .cenn_core import CeNNModel1D


# --- Architecture settings shared by every variant (from runner._CENN_FIXED). The training-only
#     items there (windows_batch_size, gradient_clip_val) are intentionally omitted: they belong
#     to the framework's own training loop (run_exp.py / exp.py), not to the nn.Module. ---
_CENN_FIXED = dict(
    hidden_dim=64, N=1, num_layers=4, dropout=0.25, neighborhood=3,
    alpha_init=0.9, enforce_bistability=False, cross_channel=False, spectral_rho=0.9,
)
_DEFAULT_K = 8

# --- Reference point: C1-BoundedTau. Every variant is base + ONLY the kwargs it changes,
#     so each row's delta (the single factor it ablates) is self-documenting. ---
_VARIANT_BASE = dict(
    adaptive_tau=True, dilation_schedule="none", multiscale_mode="none",
    integrator="euler", var_mix=False, cross_var="none", pointwise_mix=False,
    channel_groups=1, patch_len=None, stride=None, head_type="linear",
    alpha_min=0.5, alpha_max=0.99, spectral_cap=True, linear_skip=False,
)

# --- The 19 presets (ported verbatim from runner.VARIANT_SPECS). ---
VARIANT_SPECS = {
    # ---- MAIN ----
    "S0-StableBase":              {"adaptive_tau": False},                                          # fixed bounded alpha (substrate)
    "C1-BoundedTau":              {},                                                               # = base (adaptive tau, reference)
    "C2-MultiScaleEnsemble":      {"adaptive_tau": False, "multiscale_mode": "parallel_ensemble"},  # S0 + MA-A3 ensemble
    "CeNN-Full":                  {"multiscale_mode": "parallel_ensemble", "patch_len": 16, "stride": 8},  # headline
    # ---- ABLATION (leave-one-out off C1-BoundedTau unless noted) ----
    "ABL-SpectralCapOff":         {"spectral_cap": False},                                          # -MA-A2
    "ABL-Patch":                  {"patch_len": 16, "stride": 8},                                   # +patch
    "ABL-GateParam-Unbounded":    {"alpha_min": 0.0, "alpha_max": 1.0},                             # -MA-A1
    "ABL-CrossVar-Pointwise":     {"pointwise_mix": True},                                          # latent 1x1 channel mix
    "ABL-CrossVar-VarMix":        {"cross_var": "varmix"},                                          # dense O(V^2) V x V mix
    "ABL-CrossVar-STAR":          {"cross_var": "star"},                                            # Inc-4: O(V) STAR core
    "ABL-ChannelGroups-G4":       {"channel_groups": 4},                                            # grouped conv over hidden
    # ---- K x Integrator sweep (base = C1; K via _VARIANT_K) ----
    "K4-Euler":                   {},                                                               # K=4
    "K4-Heun":                    {"integrator": "heun"},                                           # K=4
    "K4-ExpEuler":                {"integrator": "exp_euler"},                                       # K=4
    "K2-Heun":                    {"integrator": "heun"},                                           # K=2
    # ---- APPENDIX ----
    "CeNN-RawBase":               {"adaptive_tau": False, "alpha_min": 0.0, "alpha_max": 1.0, "spectral_cap": False},
    "APP-C2Form-DilatedTemplate": {"dilation_schedule": "exponential"},                             # dilated TEMPLATES (vs ensemble)
    "APP-Head-MLP":               {"patch_len": 16, "stride": 8, "head_type": "mlp"},               # MLP head (paired w/ patch)
    "APP-RK4":                    {"integrator": "rk4"},                                            # RK4 accuracy ceiling
    # ---- ADDITIVITY (post-hoc): does multi-scale add anything ON TOP of the winning patch+MLP-head recipe? ----
    "APP-MultiScale-Patch-MLP":   {"multiscale_mode": "parallel_ensemble", "patch_len": 16, "stride": 8, "head_type": "mlp"},  # CeNN-Full + MLP head
    "APP-MultiScale-MLP":         {"multiscale_mode": "parallel_ensemble", "head_type": "mlp"},     # multi-scale + MLP head, no patch
    # ---- HEADLINE (AMS-CeNN, ported 2026-06-20 from runner.VARIANT_SPECS): C1 (adaptive tau) + C2
    #      (parallel-ensemble multi-scale) + a ZERO-INIT linear skip. Same spec as the SHIPPED
    #      Paper-A model. The skip here is a Linear(window*in_features -> out_features) added to the
    #      head output (faithful analog of the LTSF raw-input Linear(L->H) path, generalized to the
    #      regression task where in_features != out_features). Linear head (head_type stays base). ----
    "C1C2-Skip-K2":               {"multiscale_mode": "parallel_ensemble", "linear_skip": True},    # THE HEADLINE (K=2)
    # Anchored headline (2026-09-03): the LTSF AMS-Anc protocol ported to the surrogate — subtract
    # the window's last value per feature before the model, add it back after (NLinear form, with
    # a per-feature affine). adapter-level knob, see forward(). In the LTSF benchmark this moved
    # the mean rank from 5.32 (4th) to 3.18 (1st of 12) and confined dead-sensor damage to the
    # dead channel. The surrogate already standardizes globally, so no pipeline scaler is involved.
    "AMS-Anc":                    {"multiscale_mode": "parallel_ensemble", "linear_skip": True, "anchor": True},
    "C1C2-Skip":                  {"multiscale_mode": "parallel_ensemble", "linear_skip": True},    # K=8 reference (K-neutrality check)
    # ---- Added for the formation-recommendation surrogate (2026-11): one factor off the headline each. ----
    "AMS-NoSkip-K2":              {"multiscale_mode": "parallel_ensemble"},                          # headline without the linear skip
    "AMS-MLPHead-K2":             {"multiscale_mode": "parallel_ensemble", "linear_skip": True, "head_type": "mlp"},
    "AMS-Relaxed-K2":             {"multiscale_mode": "parallel_ensemble", "linear_skip": True,
                                   "alpha_min": 0.0, "alpha_max": 1.0},                            # headline, gate bound relaxed
    "AMS-NoSkip-Relaxed-K2":      {"multiscale_mode": "parallel_ensemble", "alpha_min": 0.0, "alpha_max": 1.0},
}

# K values for the K-sweep variants (override the default K=8).
_VARIANT_K = {"K4-Euler": 4, "K4-Heun": 4, "K4-ExpEuler": 4, "K2-Heun": 2,
              "C1C2-Skip-K2": 2,  # headline runs at K=2 (4x cheaper, accuracy-neutral on LTSF)
              "AMS-NoSkip-K2": 2, "AMS-MLPHead-K2": 2, "AMS-Relaxed-K2": 2, "AMS-NoSkip-Relaxed-K2": 2}

CENN_VARIANTS = list(VARIANT_SPECS)
CENN_MAIN_VARIANT = "CeNN-Full"


def resolve_variant(variant):
    """Return (full_backbone_kwargs_spec, K) for a named variant, or raise on unknown."""
    if variant not in VARIANT_SPECS:
        raise ValueError(
            f"Unknown CeNN variant {variant!r}. Known: {sorted(VARIANT_SPECS)}."
        )
    spec = {**_VARIANT_BASE, **VARIANT_SPECS[variant]}
    K = _VARIANT_K.get(variant, _DEFAULT_K)
    return spec, K


class Model(nn.Module):
    """CeNN surrogate adapter: ``(B, window, in_features) -> (B, out_features)``."""

    def __init__(self, configs):
        super().__init__()
        variant = getattr(configs, "cenn_variant", CENN_MAIN_VARIANT)
        spec, K = resolve_variant(variant)
        self.variant = variant
        # Optional hyperparameter overrides (tuned arms): "K", any shared architecture setting
        # (hidden_dim, num_layers, dropout, neighborhood, spectral_rho, ...) or any variant knob.
        fixed = dict(_CENN_FIXED)
        for key, value in (getattr(configs, "overrides", None) or {}).items():
            if key == "K":
                K = int(value)
            elif key in fixed:
                fixed[key] = value
            elif key in spec or key in ("anchor",):
                spec[key] = value
            else:
                raise ValueError(f"Unknown CeNN hyperparameter {key!r}")
        self.spec, self.K, self.fixed = dict(spec), K, fixed

        in_features = configs.in_features
        out_features = configs.out_features
        window = configs.window_size

        hidden = fixed["hidden_dim"]
        dropout = fixed["dropout"]
        head_type = spec.get("head_type", "linear")
        self.use_skip = bool(spec.get("linear_skip", False))
        # The backbone's OWN reconstruction head is discarded (we tap the latent before it), so
        # drop head_type before constructing it — otherwise APP-Head-MLP would build an unused MLP
        # pred-head. head_type is realized below on OUR head instead (keeps APP-Head-MLP distinct).
        # linear_skip is ALSO popped: it is an adapter-level path (the vendored cenn_core v0.2.0
        # predates the skip knob and would reject it). It is realized on OUR output below.
        # NOTE: any NEW adapter-level key added to a spec/_VARIANT_BASE must be added to this
        # exclusion set too — otherwise it flows into cenn_core's **cell_kwargs chain and crashes
        # at CeNNCell1D (which has no **kwargs). Keep this list in sync with adapter-only knobs.
        self.use_anchor = bool(spec.get("anchor", False))
        backbone_spec = {k: v for k, v in spec.items() if k not in ("head_type", "linear_skip", "anchor")}

        # Full, tested CeNN backbone. pred_length / output_multiplier only size the (discarded)
        # backbone head, so set =1.
        self.backbone = CeNNModel1D(
            n_features=in_features,
            seq_length=window,
            pred_length=1,
            output_multiplier=1,
            K=K,
            **fixed,
            **backbone_spec,
        )

        # Clean pool+project head: hidden -> out_features (no n_features bottleneck). head_type is
        # honoured here so APP-Head-MLP (mlp) is architecturally distinct from ABL-Patch (linear).
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, out_features),
            )
        else:
            self.head = nn.Linear(hidden, out_features)

        # Zero-init raw-input linear skip (AMS-CeNN headline). Faithful analog of the LTSF skip
        # (nn.Linear(seq_length, pred_length), zero-init, added to the output): here a direct linear
        # map from the FLATTENED input window to the targets, added to the head output. Zero-init =>
        # starts as a no-op, then learns a linear baseline the CeNN dynamics correct.
        if self.use_skip:
            self.skip = nn.Linear(window * in_features, out_features)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

        # Last-value anchor (AMS-Anc): the targets are future values of the SAME features, so the
        # inverse is applied per feature to the (B, H, in_features) view of the output.
        if self.use_anchor:
            if out_features % in_features != 0:
                raise ValueError("anchor requires out_features to be a multiple of in_features")
            self.in_features = in_features
            # NO learnable affine here (2026-09-03 fix). The first rerun carried a RevIN-style
            # per-feature affine; on the near-constant aging trajectories (x - last ~ 1e-3) the
            # learned scale drifted and its INVERSE (division) amplified every rollout step:
            # H32 MSE 0.22 vs 0.0004, NaN by H256. Autoregressive use needs the pure NLinear form,
            # subtract last / add last, which is exactly identity-preserving under rollout.

    def forward(self, x):
        """x: (B, window, in_features)  ->  (B, out_features) [SEI Rate, Temperature]."""
        if self.use_anchor:
            last = x[:, -1:, :]                                          # (B, 1, in)
            x = x - last
        u = self.backbone._embed(x)     # (B, hidden, L')  -- patching + cross-var per variant
        z = self.backbone.stack(u)      # (B, hidden, L')  -- full CeNN dynamics per variant
        z = z.mean(dim=-1)              # (B, hidden)       -- global temporal pooling
        out = self.head(z)              # (B, out_features)
        if self.use_skip:
            out = out + self.skip(x.reshape(x.size(0), -1))  # + zero-init Linear(window*in -> out)
        if self.use_anchor:
            B = out.size(0)
            o = out.view(B, -1, self.in_features)
            o = o + last
            out = o.reshape(B, -1)
        return out
