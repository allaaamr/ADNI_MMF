import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ResNet3D import ResNet3D
from models.ClinicalMLP import ClinicalMLP2

def _find_last_conv3d(module: nn.Module) -> nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            last = m
    if last is None:
        raise RuntimeError("No nn.Conv3d layer found in ResNet3D.")
    return last

class _Hook:
    """Keeps last layer output; call .close() to remove hook."""
    def __init__(self, layer: nn.Module):
        self.out = None
        self.h = layer.register_forward_hook(self._fn)
    def _fn(self, module, inp, out):
        self.out = out
    def close(self):
        self.h.remove()

class HybridFusion(nn.Module):
    """
    Hybrid / mid-level fusion:
      - Grab MRI conv feature map via hook (B, C, d, h, w)
      - Embed to d_model, flatten to tokens (T = d*h*w)
      - Clinical MLP -> d_model (query)
      - Cross-attention: clinical attends to MRI tokens
      - 2-layer head -> logits
    """
    def __init__(self, num_classes: int = 3, clinical_in: int = 8, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Backbones unchanged
        self.resnet3d     = ResNet3D(num_classes=num_classes, in_channels=1, layers=(2,2,2,2), widths=(32,64,128,256))
        self.clinical_mlp = ClinicalMLP2(num_features=clinical_in, num_classes=num_classes)  # we won't use its logits

        # Hook target: last Conv3d
        self._target = _find_last_conv3d(self.resnet3d)
        self._hook = _Hook(self._target)

        # We don't assume C; infer at first forward with Lazy layers
        # MRI feature projection to d_model (1x1x1 conv is cheap)
        self.mri_proj = nn.LazyConv3d(out_channels=d_model, kernel_size=1, bias=False)

        # Clinical embedding to d_model (ignore its built-in classifier)
        self.cli_embed = nn.Sequential(
            nn.Linear(clinical_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )

        # Cross-attention: query=clinical(1,B,d), key/value=MRI tokens (T,B,d)
        # Use batch_first=False for widest PyTorch compatibility
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False)

        # 2-layer head to logits
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, data_mri: torch.Tensor, data_clinical: torch.Tensor) -> torch.Tensor:
        # 1) Run MRI backbone to populate hook with conv features
        #    (ResNet3D may accept (mri) or (mri, clinical); support both)

        _ = self.resnet3d(data_mri, data_clinical)   # ignore its logits


        feat = self._hook.out                        # (B, C, d, h, w)
        if feat is None:
            raise RuntimeError("MRI conv features not captured. Ensure forward reached the hooked Conv3d.")

        # 2) Project to d_model and flatten to tokens
        #    tokens: (T, B, d_model), where T = d*h*w
        f = self.mri_proj(feat)                      # (B, d_model, d, h, w)
        B, Dm, d, h, w = f.shape
        T = d * h * w
        tokens = f.view(B, Dm, T).permute(2, 0, 1)   # (T, B, d_model)

        # 3) Clinical embedding -> query (1, B, d_model)
        q = self.cli_embed(data_clinical).unsqueeze(0)  # (1, B, d_model)

        # 4) Cross-attention (fp32 for stability)
        with torch.cuda.amp.autocast(enabled=False):
            attn_out, _ = self.attn(q.float(), tokens.float(), tokens.float())  # (1, B, d_model)

        # 5) Classifier head
        logits = self.head(attn_out.squeeze(0))      # (B, num_classes)
        return logits
