# explainability/xai_runner.py
from typing import Optional, Union, Tuple
import torch
import numpy as np
from .GradCam import MRIGradCAMRunner   # uses your existing Grad-CAM class

def _pull_tensor(obj):
    """Accept a torch tensor or a .pt dict and return a (D,H,W) or (1,D,H,W) tensor."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in ["image", "volume", "x", "data", "tensor"]:
            if k in obj:
                v = obj[k]
                return v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
    # else: assume it's a tensor-like
    return torch.as_tensor(obj)

def _choose_slice_from_cam(cam_3d: torch.Tensor, plane: str, strategy: str) -> int:
    """
    cam_3d: (D,H,W) in [0,1]; plane in {"axial","coronal","sagittal"}
    strategy: "middle" | "cammax"
    """
    D, H, W = cam_3d.shape
    if strategy == "middle":
        return {"axial": D//2, "coronal": H//2, "sagittal": W//2}[plane]

    # "cammax": pick slice with highest total CAM energy along that plane
    cam = cam_3d
    if plane == "axial":     # sum over (H,W) → argmax over D
        return int(cam.sum(dim=(1,2)).argmax().item())
    if plane == "coronal":   # sum over (D,W) → argmax over H
        return int(cam.sum(dim=(0,2)).argmax().item())
    if plane == "sagittal":  # sum over (D,H) → argmax over W
        return int(cam.sum(dim=(0,1)).argmax().item())
    raise ValueError("plane must be axial|coronal|sagittal")

def _overlay_any_plane(x_5d: torch.Tensor, cam_3d: torch.Tensor,
                       plane: str, idx: int, alpha: float = 0.35,
                       out_png: str = "gradcam.png", title: Optional[str] = None) -> str:
    """Save a PNG overlay for axial/coronal/sagittal slice."""
    import matplotlib.pyplot as plt
    vol = x_5d.squeeze().cpu().numpy()    # (D,H,W)
    cam = cam_3d.squeeze().cpu().numpy()  # (D,H,W)

    if plane == "axial":
        base, hm = vol[idx],        cam[idx]
    elif plane == "coronal":
        base, hm = vol[:, idx, :],  cam[:, idx, :]
    elif plane == "sagittal":
        base, hm = vol[:, :, idx],  cam[:, :, idx]
    else:
        raise ValueError("plane must be axial|coronal|sagittal")

    base = (base - base.min()) / (base.ptp() + 1e-8)
    hm   = (hm   - hm.min())   / (hm.ptp()   + 1e-8)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=160)
    ax.imshow(base, cmap="gray", interpolation="nearest")
    ax.imshow(hm,   cmap="jet",  interpolation="nearest", alpha=alpha)
    ax.axis("off")
    if title: ax.set_title(title, fontsize=10)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_png

def run_gradcam_from_pt(
    *,
    model_ctor,                        # callable -> model (no args or partial with your args)
    checkpoint_path: str,
    mri_pt_path: str,
    target_module: Optional[Union[str, torch.nn.Module]] = None,
    clinical: Union[float, int, np.ndarray, torch.Tensor] = 1.0,
    out_png: str = "gradcam.png",
    plane: str = "axial",              # "axial" | "coronal" | "sagittal"
    slice_idx: Optional[int] = None,   # if None, uses strategy
    strategy: str = "cammax",          # "cammax" | "middle"
    class_idx: Optional[int] = None,   # None → argmax
    device: Optional[str] = None
) -> Tuple[str, int]:
    """
    Returns: (out_png_path, predicted_class)
    """
    # 1) model + target layer
    model = model_ctor()
    if target_module is None:
        # sensible default for your ResNet3D
        target_module = model.layer4[-1].conv2

    # 2) MRI tensor (already preprocessed .pt)
    mri_obj = torch.load(mri_pt_path, map_location="cpu")
    mri = _pull_tensor(mri_obj).float()      # (D,H,W) or (1,D,H,W) or (1,1,D,H,W)

    # 3) Build runner (disable extra normalization)
    runner = MRIGradCAMRunner(
        model=model,
        checkpoint_path=checkpoint_path,
        target_layer=target_module,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        normalize_fn=lambda x: x,
        clinical_tensor=torch.as_tensor(clinical, dtype=torch.float32).view(1, -1),
    )

    # 4) Produce CAM once, then choose slice/plane
    x_5d = runner._prep_input(mri)  # (1,1,D,H,W)
    heatmap, logits, pred = runner.cam.generate(
        x_5d, class_idx=class_idx, forward_fn=runner._mk_forward_fn(runner.clinical)
    )

    if slice_idx is None:
        slice_idx = _choose_slice_from_cam(heatmap, plane=plane, strategy=strategy)

    title = f"Grad-CAM | class={int(pred)} | {plane} idx={int(slice_idx)}"
    png = _overlay_any_plane(x_5d, heatmap, plane=plane, idx=int(slice_idx),
                             alpha=0.35, out_png=out_png, title=title)
    return png, int(pred)
