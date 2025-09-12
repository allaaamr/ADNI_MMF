"""
Utilities to compute and visualize SHAP values for a clinical (tabular) PyTorch MLP.

This module produces *global* explanations on the **validation** split using the
model's **current weights** (no checkpoint reload). It saves two PNG figures:

1) Beeswarm plot  .......... "who drives the model *and in what direction*"
2) Importance bar chart .... "who matters on average" (mean |SHAP|)

Design choices:
- Background set is drawn from TRAIN (small, 64–256 rows) – this anchors attributions.
- Explanations are computed on VALIDATION (generalization, not memorization).
- For multiclass, we select the SHAP vector for each sample’s **predicted class** so
  a single (N,F) matrix feeds global plots (cleaner than stacking all classes).

Typical use:
    from utils.clinical_shap import shap_clinical_global_plots
    shap_clinical_global_plots(model, train_loader, val_loader, device,
                               feature_names=train_loader.dataset.feature_cols)
"""

from typing import Optional, Sequence, Tuple
import numpy as np
import torch
import shap
import matplotlib
matplotlib.use("Agg")  # safe for servers/CI
import matplotlib.pyplot as plt


@torch.no_grad()
def _collect_X(loader, limit: Optional[int] = None) -> torch.Tensor:
    """
    Concatenate clinical feature batches from a DataLoader.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Yields tuples like (dummy_mri, clinical_tensor, y).
    limit : int or None
        Max number of rows to collect (None = all rows).

    Returns
    -------
    X : torch.Tensor, shape (N, F)
        Clinical features concatenated across batches.
    """
    xs, seen = [], 0
    for _, x_cli, _ in loader:
        xs.append(x_cli)
        seen += x_cli.size(0)
        if limit is not None and seen >= limit:
            break
    X = torch.cat(xs, dim=0)
    return X[:limit] if limit is not None else X


def shap_clinical_global_plots(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    feature_names: Optional[Sequence[str]] = None,
    background_n: int = 128,
    chunk_size: int = 1024,
    out_beeswarm: str = "shap_beeswarm_val.png",
    out_bar: str = "shap_importance_bar_val.png",
) -> Tuple[str, str]:
    """
    Compute SHAP values for a clinical MLP and save *global* plots on the validation set.

    Parameters
    ----------
    model : torch.nn.Module
        Trained clinical model. Its forward should accept either:
          - forward(clinical_tensor) -> logits
          - forward(data_mri, data_clinical) -> logits
        (The wrapper below adapts either signature.)
    train_loader : DataLoader
        Used only to draw a small *background* set (distributions anchor).
    val_loader : DataLoader
        All rows are explained (chunked if needed) for global plots.
    device : torch.device
        CPU or CUDA device where the model runs.

    feature_names : list[str], optional
        Column names to label axes in the plots. If None, SHAP uses generic names.
    background_n : int
        Number of TRAIN rows to use as SHAP background (64–256 is typical).
        Larger backgrounds are slower with diminishing returns.
    chunk_size : int
        Number of validation rows processed per SHAP call (controls memory).
    out_beeswarm : str
        Output PNG path for the beeswarm plot (global direction + magnitude).
    out_bar : str
        Output PNG path for the mean |SHAP| importance bar chart.

    Returns
    -------
    out_beeswarm, out_bar : tuple[str, str]
        File paths of the saved PNGs.

    Notes
    -----
    • We disable AMP & keep model.eval() for stable gradients.
    • For multiclass models, SHAP returns a list of arrays (one per class) with
      shape (N, F). We select, per-row, the SHAP vector corresponding to the
      model’s *predicted* class to obtain a single (N, F) matrix for plotting.
      This provides an intuitive “global” view aligned with model decisions.
    """
    model.eval()

    # ---- Wrap model so SHAP sees a single (B,F) tensor -> logits (B,C) ----
    class _Wrap(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Try both signatures: forward(x) or forward(None, x)
            try:
                return self.base(x)
            except TypeError:
                return self.base(None, x)

    wrapped = _Wrap(model).to(device)

    # ---- Background (TRAIN) ----
    X_bg = _collect_X(train_loader, background_n).to(device)

    # ---- Explainer (DeepExplainer preferred for MLPs; GradientExplainer fallback) ----
    try:
        explainer = shap.DeepExplainer(wrapped, X_bg)
    except Exception:
        explainer = shap.GradientExplainer(wrapped, X_bg)

    # ---- Collect ALL validation features ----
    Xv = _collect_X(val_loader, None).to(device)

    # ---- Compute SHAP in chunks; build a single (N,F) matrix for global plots ----
    sv_rows, X_rows = [], []
    N = Xv.size(0)
    for i in range(0, N, chunk_size):
        x = Xv[i : i + chunk_size]

        # SHAP requires full-precision grads; disable autocast for stability
        with torch.cuda.amp.autocast(enabled=False):
            shap_values = explainer.shap_values(x)  # list(len=C)[(n,F)]  or  (n,F)
            logits = wrapped(x).float()

        if isinstance(shap_values, list):
            # Multiclass: select SHAP for the predicted class row-wise
            sv_stack = np.stack(shap_values, axis=-1)  # (n, F, C)
            pred = logits.argmax(dim=1).cpu().numpy()  # (n,)
            sel = sv_stack[np.arange(len(pred)), :, pred]  # (n, F)
        else:
            # Binary/single-output: already (n, F)
            sel = shap_values

        sv_rows.append(sel)
        X_rows.append(x.detach().cpu().numpy())

    sv_all = np.concatenate(sv_rows, axis=0)  # (N_val, F)
    X_all = np.concatenate(X_rows, axis=0)    # (N_val, F)

    # ---- Global beeswarm: shows distribution (+/- direction & density) ----
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv_all, X_all,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig(out_beeswarm, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved global beeswarm → {out_beeswarm}")

    # ---- Global bar: mean absolute SHAP (magnitude-only ranking) ----
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv_all, X_all,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig(out_bar, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved global importance bar → {out_bar}")

    return out_beeswarm, out_bar
