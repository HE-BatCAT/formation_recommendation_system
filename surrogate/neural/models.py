"""Model registry for the neural surrogates.

Every model maps ``(batch, window, in_features)`` to ``(batch, out_features)``.

An arm is "certified" when the cellular state recurrence is a contraction by construction: the retention
gate is bounded below one and the feedback template is norm-capped. The relaxed arms drop one of the two
conditions and are otherwise the same network.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from .cenn_adapter import Model as CeNNAdapter
from .cenn_adapter import resolve_variant


class GRU(nn.Module):
    def __init__(self, in_features, out_features, window, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, out_features)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class MLP(nn.Module):
    def __init__(self, in_features, out_features, window, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers, width = [nn.Flatten()], window * in_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, window):
        super().__init__()
        self.lin = nn.Linear(window * in_features, out_features)

    def forward(self, x):
        return self.lin(x.reshape(x.size(0), -1))


# arm name -> (kind, CeNN variant or None, one-line description)
ARMS = {
    # certified AMS-CeNN family
    "ams-cenn":              ("cenn", "C1C2-Skip-K2", "AMS-CeNN headline: bounded gate, multi-scale, linear skip, K=2"),
    "ams-cenn-k8":           ("cenn", "C1C2-Skip", "headline with K=8 integration steps"),
    "ams-cenn-noskip":       ("cenn", "AMS-NoSkip-K2", "headline without the linear skip"),
    "ams-cenn-mlphead":      ("cenn", "AMS-MLPHead-K2", "headline with an MLP read-out"),
    "cenn-single-scale":     ("cenn", "C1-BoundedTau", "bounded gate only: single scale, no skip, K=8"),
    "cenn-patch-mlphead":    ("cenn", "APP-Head-MLP", "single scale with patching and an MLP read-out, K=8"),
    # relaxed AMS-CeNN family (no contraction guarantee)
    "ams-cenn-relaxed":      ("cenn", "AMS-Relaxed-K2", "headline with the gate bound relaxed to (0, 1)"),
    "ams-cenn-noskip-relaxed": ("cenn", "AMS-NoSkip-Relaxed-K2", "headline without skip, gate bound relaxed"),
    "cenn-single-scale-relaxed": ("cenn", "ABL-GateParam-Unbounded", "single scale, gate bound relaxed, K=8"),
    "cenn-single-scale-capoff":  ("cenn", "ABL-SpectralCapOff", "single scale, template norm cap removed, K=8"),
    # references
    "gru":                   ("gru", None, "3-layer GRU, hidden 128"),
    "mlp":                   ("mlp", None, "MLP on the flattened traces, 256-128"),
    "linear":                ("linear", None, "linear map on the flattened traces"),
}
# variants that subtract the last input value (LVN); only valid when outputs continue the inputs
ARMS_LVN = {"ams-cenn-lvn": ("cenn", "AMS-Anc", "headline with last-value normalization, K=8")}


# Arms found by the hyperparameter search (python -m surrogate.train_neural tune ...). Each entry:
# {"kind", "variant", "overrides": architecture settings, "training": optimizer settings, "description"}.
TUNED_PATH = Path(__file__).with_name("tuned_arms.json")


def tuned_arms():
    if not TUNED_PATH.exists():
        return {}
    return json.loads(TUNED_PATH.read_text(encoding="utf-8"))


def register_tuned(name, entry):
    table = tuned_arms()
    table[name] = entry
    TUNED_PATH.write_text(json.dumps(table, indent=2), encoding="utf-8")


def _lookup(arm):
    """(kind, variant, description, architecture overrides, training overrides)."""
    table = {**ARMS, **ARMS_LVN}
    if arm in table:
        return (*table[arm], {}, {})
    tuned = tuned_arms()
    if arm in tuned:
        entry = tuned[arm]
        return (entry["kind"], entry.get("variant"), entry.get("description", "tuned"),
                entry.get("overrides", {}), entry.get("training", {}))
    raise ValueError(f"Unknown arm {arm!r}. Known: {sorted(table) + sorted(tuned)}")


def is_cellular(arm):
    return _lookup(arm)[0] == "cenn"


def certified_spec(variant, overrides):
    """True when the gate is bounded below one and the feedback template is norm-capped below one."""
    spec, _ = resolve_variant(variant)
    spec = {**spec, **{k: v for k, v in overrides.items() if k in spec}}
    rho = overrides.get("spectral_rho", 0.9)
    return bool(spec["spectral_cap"]) and spec["alpha_max"] < 1.0 and rho < 1.0


def is_certified(arm):
    kind, variant, _, overrides, _ = _lookup(arm)
    return kind == "cenn" and certified_spec(variant, overrides)


def describe(arm):
    return _lookup(arm)[2]


def training_overrides(arm):
    return dict(_lookup(arm)[4])


def build_custom(kind, variant, overrides, in_features, out_features, window):
    if kind == "cenn":
        return CeNNAdapter(SimpleNamespace(cenn_variant=variant, in_features=in_features,
                                           out_features=out_features, window_size=window,
                                           overrides=overrides))
    cls = {"gru": GRU, "mlp": MLP, "linear": Linear}[kind]
    return cls(in_features, out_features, window, **overrides)


def architecture(arm):
    """The arm's architecture as plain data. It is stored in metadata.json, so that saved weights can be
    loaded without the arm registry."""
    kind, variant, _, overrides, _ = _lookup(arm)
    return {"kind": kind, "variant": variant, "overrides": dict(overrides)}


def build(arm, in_features, out_features, window):
    kind, variant, _, overrides, _ = _lookup(arm)
    return build_custom(kind, variant, overrides, in_features, out_features, window)


def build_saved(metadata, in_features, out_features, window):
    """Rebuild a saved model from its metadata; metadata without "architecture" uses the registry."""
    saved = metadata.get("architecture")
    if saved is None:
        return build(metadata["arm"], in_features, out_features, window)
    return build_custom(saved["kind"], saved.get("variant"), saved.get("overrides", {}),
                        in_features, out_features, window)


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def fit(model, x_train, y_train, x_val, y_val, *, epochs, patience, batch_size, learning_rate,
        seed, device, weight_decay=0.0):
    """Adam on MSE. With a validation set: early stopping, best weights restored.

    Returns (model in eval mode, epochs run, best epoch).
    """
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    validate = x_val is not None and len(x_val) > 0
    if validate:
        xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    best, best_state, best_epoch, bad, epoch = float("inf"), None, 0, 0, 0
    n = len(xt)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n).to(device)
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad()
            nn.functional.mse_loss(model(xt[index]), yt[index]).backward()
            optimizer.step()
        if not validate:
            continue
        model.eval()
        with torch.no_grad():
            loss = sum(float(nn.functional.mse_loss(model(xv[s:s + 2048]), yv[s:s + 2048], reduction="sum"))
                       for s in range(0, len(xv), 2048)) / yv.numel()
        if loss < best - 1e-6:
            best, bad, best_epoch = loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience and bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    return model.eval(), epoch, best_epoch


def predict(model, x, device, batch_size=1024):
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).cpu())
    return torch.cat(out).numpy()
