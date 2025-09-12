import os, math, random, glob
import numpy as np
import nibabel as nib
from typing import Sequence, Tuple, List
from sklearn.preprocessing import StandardScaler
from nibabel.orientations import OrientationError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
from data.utils import *
from data.utils import _match_scan , _normalize_id 
from scipy.ndimage import zoom
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import re

class MMFDataset(Dataset):
    def __init__(
        self,
        args: Namespace,
        ptids: Sequence[str],
        labels: Sequence[int],
        *,
        id_col: str = "PTID",
        label_col: str = "DIAGNOSIS",
        images_dir: str = None,
        crop_size=(144,144,144),
        target_size=(112,112,112),
        augment=False,
        cache_dir: str = "mri_cache",
        train: bool = True,
        scaler: StandardScaler = None,
    ):
        assert len(ptids) == len(labels)
        self.ptids  = [str(x) for x in ptids]
        self.labels = list(map(int, labels))
        self.mode   = args.mode
        self.images_dir = images_dir or args.images_dir
        self.crop_size  = crop_size
        self.target_size = target_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir: self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ---- Clinical features aligned to ptids (no leakage) ----
        df_all = pd.read_csv(str(args.csv_file))
        cmap = {c.lower(): c for c in df_all.columns}
        id_col = cmap.get(id_col.lower(), id_col)
        label_col = cmap.get(label_col.lower(), label_col)

        df_split = df_all.set_index(id_col).loc[self.ptids]               # preserves order
        print(df_split.columns)
        X = df_split.drop(columns=[label_col, id_col, "Unnamed: 0"], errors="ignore")   # drop id/label
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        self.feature_cols = list(X.columns)

        if train:
            self.scaler = StandardScaler().fit(X.values.astype(np.float32))
        else:
            self.scaler = scaler
            assert self.scaler is not None, "Provide scaler from the train dataset"

        X_scaled = self.scaler.transform(X.values.astype(np.float32))
        self.clinical = torch.tensor(X_scaled, dtype=torch.float32)
        
        # optional: cache resolved paths (lazy)
        self._path_cache = {}

    def __len__(self): return len(self.ptids)

    # your PTID→path resolver (use your own)
    def _path_from_ptid(self, pid: str) -> str:
        p = self._path_cache.get(pid)
        if p is None:
            p = _match_scan(self.images_dir, _normalize_id(pid))  # you already have these helpers
            if p is None:
                raise FileNotFoundError(f"No MRI found for {pid} in {self.images_dir}")
            p = str(p)
            self._path_cache[pid] = p
        return p

  
    def _cache_path(self, path: str) -> Path:
        stem = Path(path).stem
        cs = "x".join(map(str, self.crop_size))
        ts = "x".join(map(str, self.target_size))
        return self.cache_dir / f"{stem}_crop{cs}_size{ts}.pt"
    # def _load_canonical(path: str) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    #     """Load NIfTI as closest canonical (RAS+) and return array + voxel sizes."""
    #     img = nib.as_closest_canonical(nib.load(path))
    #     arr = img.get_fdata(dtype=np.float32)               # (D,H,W)
    #     zooms = img.header.get_zooms()[:3]
    #     return arr, zooms
    
    @staticmethod
    def _resample_iso(arr: np.ndarray, src_zooms, target_iso: float = 2.0) -> np.ndarray:
        """Trilinear isotropic resample. Use ~1.5–2.0 mm for speed/accuracy trade-off."""
        factors = [src_zooms[i] / float(target_iso) for i in range(3)]
        # order=1 (linear) is standard for MRI; prefilter False for speed
        return zoom(arr, zoom=factors, order=1, prefilter=False)

    @staticmethod
    def _crop_foreground(arr: np.ndarray, margin: int = 8) -> np.ndarray:
        """Crop to brain bbox with a safety margin (in voxels)."""
        bbox_min, bbox_max = brain_bbox(arr)                # your existing helper
        # expand with margin and clamp
        mn = np.maximum(np.array(bbox_min) - margin, 0)
        mx = np.minimum(np.array(bbox_max) + margin, np.array(arr.shape) - 1)
        d = tuple(slice(mn[i], mx[i] + 1) for i in range(3))
        return arr[d]

    @staticmethod
    def _znorm_brain(arr: np.ndarray, clip: float = 5.0) -> np.ndarray:
        """Z-score using nonzero voxels (your zscore_nonzero logic); then clip."""
        z = zscore_nonzero(arr, clip=clip)                  # you already have this
        return z

    @staticmethod
    def _resize_torch(arr: np.ndarray, target_size=(112,112,112)) -> torch.Tensor:
        """Final resize to model input size."""
        v = torch.from_numpy(arr)[None, None]               # (1,1,D,H,W)
        v = F.interpolate(v, size=target_size, mode="trilinear", align_corners=False)
        return v.squeeze(0).contiguous()            
    
        # def _resize_torch(self, arr: np.ndarray) -> torch.Tensor:
    #     v = torch.from_numpy(arr)[None,None]  # (1,1,D,H,W)
    #     v = F.interpolate(v, size=self.target_size, mode="trilinear", align_corners=False)
    #     return v.squeeze(0).contiguous()      # (1,d,h,w)        # (1,d,h,w)

    def _get_mri_tensor(self, pid: str) -> torch.Tensor:
        path = self._path_from_ptid(pid)

        if self.cache_dir:
            cp = self._cache_path(path)
            if cp.exists():
                return torch.load(cp)

        # ----- RECOMMENDED ORDER -----
        arr, zooms = self._to_canonical_safe(path)                  # 1) canonicalize
        arr = self._resample_iso(arr, zooms, target_iso=2.0)     # 2) isotropic resample
        arr = self._crop_foreground(arr, margin=8)               # 3) crop with margin
        arr = self._znorm_brain(arr, clip=5.0)                   # 4) z-score inside crop
        v = self._resize_torch(arr, target_size=self.target_size) # 5) final resize
    # --------------------------------

        if self.cache_dir:
            torch.save(v, cp)

        return v

    def __getitem__(self, idx: int):
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        pid = self.ptids[idx]

        if self.mode == "radio":
            mri = self._get_mri_tensor(pid)
            return mri, torch.tensor(1), y
        elif self.mode == "clinical":
            return torch.tensor(1), self.clinical[idx], y
        else:  # fusion
            mri = self._get_mri_tensor(pid)
            return mri, self.clinical[idx], y


# # at top of file (if not already)
# from nibabel.orientations import OrientationError
# from typing import Tuple
    def _to_canonical_safe(self, path: str) -> Tuple[np.ndarray, Tuple[float,float,float]]:
        """
        Load NIfTI robustly:
        - squeeze singleton 4th dim if present (… ,1)
        - canonicalize to RAS+ only if 3D (avoid dropping coords)
        - return (array_float32, voxel_sizes_xyz)
        """
        img = nib.load(path)
        arr = np.asanyarray(img.dataobj)  # lazy read

        # Squeeze a singleton non-spatial dim if present
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]                 # (D,H,W,1) -> (D,H,W)
            img = nib.Nifti1Image(arr, img.affine, img.header)
        elif arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]                      # (1,D,H,W) -> (D,H,W)
            img = nib.Nifti1Image(arr, img.affine, img.header)

        zooms = img.header.get_zooms()[:3]

        if arr.ndim == 3:
            try:
                img_c = nib.as_closest_canonical(img)
                arr   = np.asanyarray(img_c.dataobj)
                zooms = img_c.header.get_zooms()[:3]
            except OrientationError:
                # keep original orientation if canonicalization would drop coordinates
                pass

        arr = np.asarray(arr, dtype=np.float32, order="C")
        return arr, (float(zooms[0]), float(zooms[1]), float(zooms[2]))



    def _load_nii(self, path: str) -> np.ndarray:
        img = nib.load(path); arr = img.get_fdata(dtype=np.float32)
        return zscore_nonzero(arr, clip=5.0)

    # def _preprocess_np(self, arr: np.ndarray) -> np.ndarray:
    #     bbox_min, bbox_max = brain_bbox(arr)
    #     D,H,W = arr.shape
    #     d,h,w = min(self.crop_size[0],D), min(self.crop_size[1],H), min(self.crop_size[2],W)
    #     return center_crop_from_bbox(arr, bbox_min, bbox_max, (d,h,w))

    # def _resize_torch(self, arr: np.ndarray) -> torch.Tensor:
    #     v = torch.from_numpy(arr)[None,None]  # (1,1,D,H,W)
    #     v = F.interpolate(v, size=self.target_size, mode="trilinear", align_corners=False)
    #     return v.squeeze(0).contiguous()      # (1,d,h,w)

    # def _get_mri_tensor(self, pid: str) -> torch.Tensor:
    #     path = self._path_from_ptid(pid)
    #     if self.cache_dir:
    #         cp = self._cache_path(path)
    #         if cp.exists(): return torch.load(cp)

    #     v = self._resize_torch(self._preprocess_np(self._load_nii(path)))
    #     if self.cache_dir: torch.save(v, cp)
    #     return v



# class MMFDataset(Dataset):
#     """

#     """
#     def __init__(
#         self,
#         args: Namespace,
#         mri_paths: Sequence[str],
#         labels: Sequence[int],
#         crop_size: Tuple[int,int,int] = (144,144,144),   # crop around brain first
#         target_size: Tuple[int,int,int] = (112,112,112), # then resize smaller (speed!)
#         augment: bool = False,
#         cache_dir: str = "mri_cache",
#         print_shapes: bool = True,
#         train: bool = True,
#         scaler: StandardScaler=None
#     ):
#         assert len(mri_paths) == len(labels)
#         self.paths = list(mri_paths)
#         self.labels = list(map(int, labels))
#         self.crop_size = crop_size
#         self.target_size = target_size
#         self.augment = augment
#         self.print_shapes = print_shapes
#         self._printed = set()  # avoid spamming shapes for every access
#         self.mode= args.mode
#                 # --- Prepare clinical features: drop PTID and Diagnosis (case-insensitive) ---
#         df =  pd.read_csv(str(args.csv_file))

#         # map lowercase -> original column names
#         lower_map = {c.lower(): c for c in df.columns}
#         drop_cols = []
#         for key in ("ptid", "diagnosis",):
#             if key in lower_map:
#                 drop_cols.append(lower_map[key])
#         drop_cols.append("Unnamed: 0")
#         # drop, coerce to numeric, fill NAs
#         X = df.drop(columns=drop_cols, errors="ignore")

#         # assert len(X) == len(self.paths), (
#         #     "clinical_df row count must match mri_paths/labels. "
#         #     f"Got clinical={len(X)}, mri={len(self.paths)}."
#         # )
#         assert X.shape[1] > 0, "No clinical feature columns remain after dropping PTID/Diagnosis."

#         if train:
#             self.scaler = StandardScaler().fit(X.values.astype(np.float32))
#             X_scaled = self.scaler.transform(X.values.astype(np.float32))
#         else:
#             self.scaler  = scaler
#             print(self.scaler)
#             X_scaled = self.scaler.transform(X.values)

#         self.feature_cols = list(X.columns)
#         self.clinical = torch.tensor(X_scaled, dtype=torch.float32)


#         self.cache_dir = Path(cache_dir) if cache_dir else None
#         if self.cache_dir:
#             self.cache_dir.mkdir(parents=True, exist_ok=True)

#     def __len__(self): return len(self.paths)

#     def _cache_path(self, path: str) -> Path:
#         stem = Path(path).stem
#         cs = "x".join(map(str, self.crop_size))
#         ts = "x".join(map(str, self.target_size))
#         return self.cache_dir / f"{stem}_crop{cs}_size{ts}.pt"

#     def _load_nii(self, path: str) -> np.ndarray:
#         img = nib.load(path)  # nibabel already memmaps compressed dataobj when possible
#         arr = img.get_fdata(dtype=np.float32)  # (D,H,W)
#         arr = zscore_nonzero(arr, clip=5.0)
#         return arr

#     def _preprocess_np(self, arr: np.ndarray) -> np.ndarray:
#         # (1) print once
#         # if self.print_shapes and id(arr) not in self._printed:
#         #     print(f"[shape] original: {arr.shape}")
#         #     self._printed.add(id(arr))

#         # (2) brain-centered crop
#         bbox_min, bbox_max = brain_bbox(arr)
#         # ensure crop size does not exceed volume
#         D,H,W = arr.shape
#         d = min(self.crop_size[0], D)
#         h = min(self.crop_size[1], H)
#         w = min(self.crop_size[2], W)
#         arr = center_crop_from_bbox(arr, bbox_min, bbox_max, (d,h,w))
#         # if self.print_shapes:
#         #     print(f"[shape] after crop: {arr.shape}")

#         return arr

#     def _resize_torch(self, arr: np.ndarray) -> torch.Tensor:
#         # to torch and resize (trilinear)
#         v = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
#         v = F.interpolate(v, size=self.target_size, mode="trilinear", align_corners=False)
#         v = v.squeeze(0)                                    # (1,d,h,w)  <-- 4-D (C,D,H,W)
#         # DO NOT call channels_last_3d here (needs 5-D)
#         v = v.contiguous()
#         # if self.print_shapes:
#             # print(f"[shape] after resize: {tuple(v.shape[1:])}")
#         return v

#     def _get_mri_tensor(self, idx: int) -> torch.Tensor:
#         path = self.paths[idx]
#         if self.cache_dir:
#             cp = self._cache_path(path)
#             if cp.exists():
#                 return torch.load(cp)                        # (1,d,h,w)
#         arr = self._load_nii(path)
#         arr = self._preprocess_np(arr)
#         v = self._resize_torch(arr)                          # (1,d,h,w)
#         if self.cache_dir:
#             torch.save(v, cp)
#         return v
    
    
#     def __getitem__(self, idx: int):
#         y = torch.tensor(self.labels[idx], dtype=torch.long)

#         if self.mode == "radio":
#             mri = self._get_mri_tensor(idx)
#             return mri, torch.tensor(1), y

#         elif self.mode == "clinical":
#             clinical_tensor = self.clinical[idx]
#             return torch.tensor(1), clinical_tensor, y

#         else:  # fusion
#             mri = self._get_mri_tensor(idx)
#             clinical_tensor = self.clinical[idx]
#             return mri, clinical_tensor, y


