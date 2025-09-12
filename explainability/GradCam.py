# explainability.py
import os
from typing import Optional, Tuple, Union, Callable

import torch
import torch.nn.functional as F
import numpy as np

try:
    import nibabel as nib  # optional, for .nii/.nii.gz
except Exception:
    nib = None

import matplotlib.pyplot as plt


def _strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        out[k[len("module."):]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            out[k] = v
    return out


def _resolve_module_by_name(root: torch.nn.Module, name: str) -> torch.nn.Module:
    cur = root
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _to_5d(t: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    if t.ndim == 3:      # (D,H,W)
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 4:    # (C,D,H,W) or (1,D,H,W)
        t = t.unsqueeze(0)
    elif t.ndim != 5:    # (1,1,D,H,W)
        raise ValueError(f"Expected 3D/4D/5D input, got shape {tuple(t.shape)}")
    return t.float()


def _minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + eps)


class GradCAM3D:
    """
    3D Grad-CAM with a pluggable forward_fn.
    If your model expects (mri, clinical), pass forward_fn=lambda m, x: m(x, clinical).
    """
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Union[str, torch.nn.Module],
        device: Optional[str] = None,
        forward_fn: Optional[Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = None,
    ):
        self.model = model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.forward_fn = forward_fn  # optional; can also be provided per-call in generate()

        # Resolve the hook layer
        self.target_layer = _resolve_module_by_name(self.model, target_layer) \
                            if isinstance(target_layer, str) else target_layer

        self._acts = None
        self._grads = None

        def fwd_hook(_module, _in, out):
            self._acts = out  # (B,C,D',H',W')
            def bwd_hook(grad):
                self._grads = grad
            out.register_hook(bwd_hook)

        self._hook_handle = self.target_layer.register_forward_hook(fwd_hook)

    def remove_hooks(self):
        try:
            self._hook_handle.remove()
        except Exception:
            pass

    @torch.no_grad()
    def _upsample_to_input(self, cam: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
        return F.interpolate(cam, size=size, mode="trilinear", align_corners=False)

    def generate(
        self,
        x5d: torch.Tensor,                     # (1,1,D,H,W)
        class_idx: Optional[int] = None,
        apply_relu: bool = True,
        forward_fn: Optional[Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Returns:
          heatmap_3d: (D,H,W) in [0,1]
          logits: (1, num_classes)
          pred_idx: int
        """
        self._acts = None
        self._grads = None
        x5d = x5d.to(self.device)

        fn = forward_fn or self.forward_fn or (lambda m, x: m(x))

        with torch.enable_grad():
            logits = fn(self.model, x5d)  # shape (1, num_classes)
            if logits.ndim != 2:
                raise RuntimeError("Model must return logits of shape (B, num_classes).")
            pred_idx = int(logits.argmax(dim=1).item()) if class_idx is None else int(class_idx)
            loss = logits[:, pred_idx].sum()
            self.model.zero_grad(set_to_none=True)
            loss.backward()

        if self._acts is None or self._grads is None:
            raise RuntimeError("Hooks missed activations/gradients. Check target_layer path.")

        grads = self._grads  # (1,C,D',H',W')
        acts  = self._acts   # (1,C,D',H',W')
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # (1,C,1,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True)    # (1,1,D',H',W')
        if apply_relu:
            cam = F.relu(cam)
        cam = _minmax(cam)

        D, H, W = x5d.shape[2:]
        cam_up = self._upsample_to_input(cam, (D, H, W)).squeeze(0).squeeze(0)
        cam_up = _minmax(cam_up)

        return cam_up.detach().cpu(), logits.detach().cpu(), pred_idx


class MRIGradCAMRunner:
    """
    High-level runner that supports models expecting (mri_img, clinical_tensor).
    """
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        target_layer: Union[str, torch.nn.Module],
        device: Optional[str] = None,
        normalize_fn: Optional[callable] = None,
        clinical_tensor: Optional[Union[float, int, np.ndarray, torch.Tensor]] = None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_fn = normalize_fn
        self._load_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)

        self.clinical = self._prep_clinical(clinical_tensor) if clinical_tensor is not None else None
        self.cam = GradCAM3D(self.model, target_layer=target_layer, device=self.device)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state = _strip_module_prefix(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[GradCAM] Non-strict load:\n  Missing:", missing, "\n  Unexpected:", unexpected)

    @staticmethod
    def _default_norm(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean()
        x = x / (x.std() + 1e-6)
        return _minmax(x)

    def _prep_input(self, vol: Union[str, np.ndarray, torch.Tensor], dtype=torch.float32) -> torch.Tensor:
        if isinstance(vol, str):
            if nib is None:
                raise RuntimeError("nibabel required for NIfTI. pip install nibabel")
            arr = np.asanyarray(nib.load(vol).get_fdata())
        else:
            arr = vol
        x = _to_5d(arr).to(dtype)
        if self.normalize_fn is not None:
            x = self.normalize_fn(x)
        else:
            x = self._default_norm(x)
        return x

    def _prep_clinical(self, c: Union[float, int, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(c, (float, int)):
            c = torch.tensor([[float(c)]], dtype=torch.float32)       # (1,1)
        elif isinstance(c, np.ndarray):
            c = torch.from_numpy(c).float()
        elif isinstance(c, torch.Tensor):
            c = c.float()
        else:
            raise ValueError("Unsupported clinical type")
        if c.ndim == 1:   # (F,) -> (1,F)
            c = c.unsqueeze(0)
        return c.to(self.device)

    @staticmethod
    def _overlay_axial(volume_5d: torch.Tensor, heatmap_3d: torch.Tensor,
                       slice_idx: Optional[int] = None, alpha: float = 0.35,
                       out_png: Optional[str] = None, title: Optional[str] = None) -> str:
        vol = volume_5d.squeeze().cpu().numpy()     # (D,H,W)
        cam = heatmap_3d.squeeze().cpu().numpy()    # (D,H,W)
        D = vol.shape[0]
        idx = slice_idx if slice_idx is not None else (D // 2)
        base = (vol[idx] - vol[idx].min()) / (vol[idx].ptp() + 1e-8)
        hm   = (cam[idx] - cam[idx].min()) / (cam[idx].ptp() + 1e-8)

        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=160)
        ax.imshow(base, cmap="gray", interpolation="nearest")
        ax.imshow(hm, cmap="jet", interpolation="nearest", alpha=alpha)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10)
        out_png = out_png or "gradcam_overlay.png"
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return out_png

    def _mk_forward_fn(self, clinical_tensor: torch.Tensor):
        def fn(m, x):
            return m(x, clinical_tensor)
        return fn

    def run_on_path(self, nifti_path: str, class_idx: Optional[int] = None,
                    out_png: str = "gradcam_overlay.png", slice_idx: Optional[int] = None,
                    clinical_override: Optional[Union[float, int, np.ndarray, torch.Tensor]] = None) -> str:
        x = self._prep_input(nifti_path)
        clin = self._prep_clinical(clinical_override) if clinical_override is not None else self.clinical
        if clin is None:
            raise ValueError("Clinical tensor is required. Provide clinical_tensor in __init__ or clinical_override here.")
        heatmap, logits, pred = self.cam.generate(
            x, class_idx=class_idx, forward_fn=self._mk_forward_fn(clin)
        )
        title = f"Grad-CAM | class={class_idx if class_idx is not None else pred}"
        return self._overlay_axial(_minmax(x), heatmap, slice_idx=slice_idx, out_png=out_png, title=title)

    def run_on_tensor(self, volume: Union[np.ndarray, torch.Tensor], class_idx: Optional[int] = None,
                      out_png: str = "gradcam_overlay.png", slice_idx: Optional[int] = None,
                      clinical_override: Optional[Union[float, int, np.ndarray, torch.Tensor]] = None) -> str:
        x = self._prep_input(volume)
        clin = self._prep_clinical(clinical_override) if clinical_override is not None else self.clinical
        if clin is None:
            raise ValueError("Clinical tensor is required. Provide clinical_tensor in __init__ or clinical_override here.")
        heatmap, logits, pred = self.cam.generate(
            x, class_idx=class_idx, forward_fn=self._mk_forward_fn(clin)
        )
        title = f"Grad-CAM | class={class_idx if class_idx is not None else pred}"
        return self._overlay_axial(_minmax(x), heatmap, slice_idx=slice_idx, out_png=out_png, title=title)
