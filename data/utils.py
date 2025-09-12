# 3d_alzheimer_classifier.py
import os, math, random, glob
import numpy as np
import nibabel as nib
from typing import Sequence, Tuple, List
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
from typing import Tuple, List
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from pathlib import Path



def _normalize_id(x):
    """Make ID matching robust: strip spaces, lower-case, remove obvious separators."""
    if pd.isna(x): return None
    s = str(x).strip()
    return s


def _match_scan(images_dir: str, pid: str):
    """
    Assumes layout:
      Images/<PID>/[the single MRI file].nii.gz  (or .nii)
    If the exact <PID> folder isn’t found, tries a case-insensitive match.
    """
    root = Path(images_dir)
    pid_str = str(pid).strip()
    if not pid_str:
        return None

    # 1) exact subfolder
    subdir = root / pid_str
    if not subdir.exists() or not subdir.is_dir():
        # 2) case-insensitive fallback
        matches = [d for d in root.iterdir() if d.is_dir() and d.name.lower() == pid_str.lower()]
        if not matches:
            return None
        subdir = matches[0]

    # Expect exactly one scan file in the PID folder
    nii_files = sorted(subdir.glob("*.nii.gz")) + sorted(subdir.glob("*.nii"))
    if len(nii_files) == 0:
        # no scan found in this PID folder
        return None
    if len(nii_files) > 1:
        # safety: pick the shortest filename (usually the main scan)
        nii_files.sort(key=lambda p: len(p.name))

    return nii_files[0]


from typing import List, Tuple
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# def build_splits_from_csv_ids(
#     csv_file: str,
#     id_col: str = "PTID",
#     label_col: str = "DIAGNOSIS",
#     val_size: float = 0.2,
#     seed: int = 42,
# ) -> Tuple[List[str], List[int], List[str], List[int]]:
#     df = pd.read_csv(csv_file)

#     # resolve columns case-insensitively
#     cmap = {c.lower(): c for c in df.columns}
#     id_col = cmap.get(id_col.lower(), id_col)
#     label_col = cmap.get(label_col.lower(), label_col)

#     df = df[[id_col, label_col]].dropna()

#     # map labels robustly to {0,1,2}
#     y = df[label_col]
#     if set(map(str, pd.unique(y))) <= {"0","1","2"}:
#         y = y.astype(int)
#     elif set(pd.unique(y)) <= {"CN","MCI","AD"}:
#         y = y.map({"CN":0, "MCI":1, "AD":2}).astype(int)
#     elif set(map(int, pd.unique(y))) <= {1,2,3}:
#         y = y.astype(int) - 1
#     else:
#         raise ValueError(f"Unrecognized labels in {label_col}")

#     ptids = df[id_col].astype(str).tolist()
#     y = y.tolist()

#     # stratified split on labels
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
#     (tr_idx, va_idx), = sss.split(ptids, y)

#     tr_ids  = [ptids[i] for i in tr_idx]
#     tr_lbls = [y[i] for i in tr_idx]
#     va_ids  = [ptids[i] for i in va_idx]
#     va_lbls = [y[i] for i in va_idx]

#     # quick sanity
#     def _dist(lbls):
#         from collections import Counter
#         c, tot = Counter(lbls), len(lbls)
#         return {k: f"{v} ({v/tot:.1%})" for k,v in sorted(c.items())}
#     print(f"[INFO] Train: {len(tr_ids)} | Val: {len(va_ids)}")
#     print(f"[INFO] Train label dist: {_dist(tr_lbls)}")
#     print(f"[INFO] Val   label dist: {_dist(va_lbls)}")

#     return tr_ids, tr_lbls, va_ids, va_lbls



import os
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def build_and_save_stratified_splits_from_csv_ids(
    csv_file: str,
    out_dir: str = "cv_splits",
    n_splits: int = 5,
    id_col: str = "PTID",
    label_col: str = "DIAGNOSIS",
    seed: int = 42,
) -> List[Tuple[List[str], List[int], List[str], List[int]]]:
    """
    Creates Stratified K-Fold splits (balanced by label), saves each fold's
    train/val IDs to CSV, and returns all splits in-memory.

    Returns:
        splits: list of tuples [(train_ids, train_labels, val_ids, val_labels), ...]
                length == n_splits
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_file)

    # resolve columns case-insensitively
    cmap = {c.lower(): c for c in df.columns}
    id_col = cmap.get(id_col.lower(), id_col)
    label_col = cmap.get(label_col.lower(), label_col)

    df = df[[id_col, label_col]].dropna()

    # robust label mapping -> ints {0,1,2}
    y = df[label_col]
    if set(map(str, pd.unique(y))) <= {"0", "1", "2"}:
        y = y.astype(int)
    elif set(pd.unique(y)) <= {"CN", "MCI", "AD"}:
        y = y.map({"CN": 0, "MCI": 1, "AD": 2}).astype(int)
    elif set(map(int, pd.unique(y))) <= {1, 2, 3}:
        y = y.astype(int) - 1
    else:
        raise ValueError(f"Unrecognized labels in {label_col}")

    ptids = df[id_col].astype(str).tolist()
    y = y.tolist()

    # Stratified K-Fold (balanced label ratios across folds)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits: List[Tuple[List[str], List[int], List[str], List[int]]] = []

    def _dist(lbls):
        from collections import Counter
        c, tot = Counter(lbls), len(lbls)
        return {k: f"{v} ({v/tot:.1%})" for k, v in sorted(c.items())}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(ptids, y), start=1):
        tr_ids  = [ptids[i] for i in tr_idx]
        tr_lbls = [y[i]     for i in tr_idx]
        va_ids  = [ptids[i] for i in va_idx]
        va_lbls = [y[i]     for i in va_idx]

        # Save CSVs for this fold
        fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        pd.DataFrame({id_col: tr_ids, label_col: tr_lbls}).to_csv(
            os.path.join(fold_dir, f"train_fold_{fold_idx}.csv"), index=False
        )
        pd.DataFrame({id_col: va_ids, label_col: va_lbls}).to_csv(
            os.path.join(fold_dir, f"val_fold_{fold_idx}.csv"), index=False
        )

        print(f"[FOLD {fold_idx}] Train: {len(tr_ids)} | Val: {len(va_ids)}")
        print(f"[FOLD {fold_idx}] Train label dist: {_dist(tr_lbls)}")
        print(f"[FOLD {fold_idx}] Val   label dist: {_dist(va_lbls)}")

        splits.append((tr_ids, tr_lbls, va_ids, va_lbls))

    return splits


# def build_splits_from_csv(
#     images_dir: str,
#     csv_file: str,
#     args,
#     id_col: str = "PTID",
#     label_col: str = "DIAGNOSIS",     # be strict: ADNI often uses upper-case
#     val_size: float = 0.2,
# ) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[str]]:
#     """
#     Match CSV rows to scan files and produce train/val splits.
#     - Deduplicates by scan path (avoid same file in both splits)
#     - Maps labels to {0,1,2} robustly
#     - Group-stratifies by patient (PTID) to avoid leakage
#     Returns:
#       train_paths, train_labels, val_paths, val_labels, train_ptids, val_ptids
#     """
#     df = pd.read_csv(csv_file)
#     # tolerant column resolving (case-insensitive)
#     cmap = {c.lower(): c for c in df.columns}
#     if id_col.lower() not in cmap:
#         raise ValueError(f"{id_col=} not found. CSV has: {df.columns.tolist()}")
#     if label_col.lower() not in cmap:
#         raise ValueError(f"{label_col=} not found. CSV has: {df.columns.tolist()}")
#     id_col = cmap[id_col.lower()]
#     label_col = cmap[label_col.lower()]

#     # keep required cols only, drop missing
#     df = df[[id_col, label_col]].dropna()

#     # ---- robust label mapping to {0,1,2} ----
#     y_raw = df[label_col]
#     uniq = pd.unique(y_raw)
#     # common ADNI patterns
#     mapping = None
#     if set(map(str, uniq)) <= {"0","1","2"}:
#         df[label_col] = y_raw.astype(int)
#     elif set(uniq) <= {"CN","MCI","AD"}:
#         mapping = {"CN":0, "MCI":1, "AD":2}
#         df[label_col] = y_raw.map(mapping).astype(int)
#     elif set(map(int, uniq)) <= {1,2,3}:
#         # map 1,2,3 -> 0,1,2
#         df[label_col] = y_raw.astype(int) - 1
#     else:
#         raise ValueError(f"Unrecognized label values: {sorted(map(str, uniq))}. "
#                          "Add a mapping to {0,1,2}.")

#     # ---- build (path, label, ptid), dedup by path ----
#     rows, missing, dup = [], [], 0
#     seen_paths = set()
#     for pid, lab in zip(df[id_col].astype(str), df[label_col].astype(int)):
#         norm_pid = _normalize_id(pid)
#         p = _match_scan(images_dir, norm_pid)  # MUST return a single full path or None
#         if p is None:
#             missing.append(norm_pid); continue
#         p = str(p)
#         if p in seen_paths:
#             dup += 1; continue    # avoid path-level duplicates across splits
#         seen_paths.add(p)
#         rows.append((p, lab, norm_pid))

#     if not rows:
#         raise RuntimeError("No scans matched the CSV IDs. Check ID normalization and filenames.")
#     if missing:
#         print(f"[WARN] {len(missing)} IDs had no matching scan (e.g. {missing[:5]})")
#     if dup:
#         print(f"[INFO] Skipped {dup} duplicate scan-path entries")

#     paths = [r[0] for r in rows]
#     labels = [r[1] for r in rows]
#     groups = [r[2] for r in rows]  # PTIDs for group split

#     # ---- group-stratified split (preferred) ----
#     n_splits = max(2, int(round(1.0 / val_size))) if val_size > 0 else 2
#     if len(np.unique(labels)) < 2:
#         # edge-case: single class, fallback to simple split
#         tr_idx, va_idx = train_test_split(
#             np.arange(len(paths)), test_size=val_size, random_state=args.seed
#         )
#     else:
#         sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
#         # take the first fold as validation
#         tr_idx, va_idx = next(sgkf.split(paths, labels, groups))

#     train_paths = [paths[i] for i in tr_idx]
#     val_paths   = [paths[i] for i in va_idx]
#     train_labels= [labels[i] for i in tr_idx]
#     val_labels  = [labels[i] for i in va_idx]
#     train_ptids = [groups[i] for i in tr_idx]
#     val_ptids   = [groups[i] for i in va_idx]

#     # ---- distribution check ----
#     def _dist(lbls):
#         from collections import Counter
#         c, tot = Counter(lbls), len(lbls)
#         return {k: f"{v} ({v/tot:.1%})" for k, v in sorted(c.items())}

#     print(f"[INFO] Total matched: {len(paths)} | Train: {len(train_paths)} | Val: {len(val_paths)}")
#     print(f"[INFO] Train label dist: {_dist(train_labels)}")
#     print(f"[INFO] Val   label dist: {_dist(val_labels)}")

#     return train_paths, train_labels, val_paths, val_labels




# def set_seed(seed: int = 42):
#     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def zscore_nonzero(x: np.ndarray, clip: float = 5.0):
    """Z-score normalize using nonzero voxels to avoid background skew."""
    nz = x[np.nonzero(x)]
    if nz.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    m, s = nz.mean(), nz.std()
    s = s if s > 1e-6 else 1.0
    x = (x - m) / s
    if clip is not None:
        x = np.clip(x, -clip, clip)
    return x.astype(np.float32)

def trilinear_resize_torch(vol: torch.Tensor, out_size: Tuple[int,int,int]) -> torch.Tensor:
    """
    vol: (1, 1, D, H, W) float32 tensor
    returns same shape with spatial resized to out_size
    """
    return F.interpolate(vol, size=out_size, mode="trilinear", align_corners=False)


# ---- add near the other utils ----
def brain_bbox(arr: np.ndarray, min_voxels: int = 1000) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """
    Rough brain bbox:
      1) try nonzero mask
      2) if too small (some datasets aren't exactly zero background), use a percentile threshold
    Returns (mins), (maxs) inclusive bounds along (D,H,W).
    """
    assert arr.ndim == 3
    mask = arr > 0
    if mask.sum() < min_voxels:
        thr = np.percentile(arr, 85.0)  # robust foreground fallback
        mask = arr > thr
    if mask.sum() == 0:
        # fallback to full volume
        D,H,W = arr.shape
        return (0,0,0), (D-1,H-1,W-1)
    dz, hy, wx = np.where(mask)
    return (int(dz.min()), int(hy.min()), int(wx.min())), (int(dz.max()), int(hy.max()), int(wx.max()))

def center_crop_from_bbox(arr: np.ndarray, bbox_min, bbox_max, crop_size: Tuple[int,int,int]) -> np.ndarray:
    """
    Center a fixed-size crop around the center of the bbox (clamped to volume).
    crop_size: (d,h,w)
    """
    D,H,W = arr.shape
    d,h,w = crop_size
    cz = (bbox_min[0] + bbox_max[0]) // 2
    ch = (bbox_min[1] + bbox_max[1]) // 2
    cw = (bbox_min[2] + bbox_max[2]) // 2
    sz = max(0, min(D - d, cz - d // 2))
    sh = max(0, min(H - h, ch - h // 2))
    sw = max(0, min(W - w, cw - w // 2))
    return arr[sz:sz+d, sh:sh+h, sw:sw+w]

# def to_channels_last_3d(t: torch.Tensor) -> torch.Tensor:
#     memfmt = getattr(torch, "channels_last_3d", None)
#     return t.contiguous(memory_format=memfmt) if memfmt is not None else t.contiguous()

def as_channels_last_3d(x: torch.Tensor) -> torch.Tensor:
    memfmt = getattr(torch, "channels_last_3d", None)
    if memfmt is None or x.dim() != 5:   # need (N,C,D,H,W)
        return x.contiguous()
    return x.contiguous(memory_format=memfmt)
