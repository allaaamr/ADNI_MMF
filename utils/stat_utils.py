import json, os, numpy as np, torch
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

def collect_logits_and_targets(model, val_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the model on the entire validation loader and collects raw logits + true labels.

    Why needed:
    - We want to compute metrics (accuracy, AUROC) consistently after training.
    - Storing logits + labels per fold allows later statistical tests and calibration.
    """
    model.eval()
    ys, lgs = [], []
    with torch.no_grad():
        for batch in val_loader:
            data_MRI, data_clinical, y  = batch
            data_MRI = data_MRI.to(device, non_blocking=True)   # (B,1,D,H,W)
            data_clinical = data_clinical.to(device, non_blocking=True)    
            y = y.to(device, non_blocking=True)   

            logits = model(data_MRI, data_clinical)
            lgs.append(logits.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    return np.concatenate(lgs, axis=0), np.concatenate(ys, axis=0)


def compute_multiclass_metrics(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Computes accuracy and macro-AUROC for multiclass classification.

    Why needed:
    - Accuracy is intuitive but can be misleading in imbalanced medical data.
    - AUROC is threshold-independent and more robust for medical classification tasks.
    - Both are standard metrics in medical AI papers, so we log them per fold.
    """
    y_pred = logits.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    try:
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        auroc_macro = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        # AUROC may fail if a class is missing in val set; in that case we record NaN.
        auroc_macro = np.nan
    return {"acc": float(acc), "auroc_macro": float(auroc_macro)}


def mean_ci(x: np.ndarray, alpha=0.05):
    """
    Computes mean, standard deviation, and 95% confidence interval for a metric.

    Why needed:
    - Instead of reporting only mean ± std, reviewers often ask for confidence intervals.
    - This allows more rigorous reporting across cross-validation folds.
    """
    x = np.asarray(x, dtype=float)
    m = float(np.nanmean(x)); s = float(np.nanstd(x, ddof=1))
    n = int(np.sum(np.isfinite(x)))
    if n <= 1:
        return m, s, (np.nan, np.nan)
    tval = stats.t.ppf(1 - alpha/2, df=n-1)
    half = tval * s / np.sqrt(n)
    return m, s, (m - half, m + half)

def save_json(d: Dict, path: str):
    """
    Saves a Python dict as JSON to disk (pretty-printed).

    Why needed:
    - Central place to dump metrics, summaries, and results for reproducibility.
    - Keeps track of performance per fold and summary statistics for later inspection.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: 
        json.dump(d, f, indent=2)
