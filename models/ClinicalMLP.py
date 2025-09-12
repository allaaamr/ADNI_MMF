import torch
import torch.nn as nn
import torch.nn.functional as F

from xgboost import XGBClassifier
import numpy as np, joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report



class ClinicalMLP2(nn.Module):
    """
    MLP for 8-D clinical features.
    - LayerNorm instead of BatchNorm (small batches)
    - GELU activations
    - Light residual block for stability
    """
    def __init__(self, num_features: int = 8, num_classes: int = 3,
                 hidden_sizes=(64, 64), dropout: float = 0.2):
        super().__init__()
        self.in_norm = nn.LayerNorm(num_features)

        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(h))
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # lightweight residual projection (optional)
        self.proj = nn.Linear(num_features, in_dim) if in_dim != num_features else nn.Identity()

        self.head = nn.Linear(in_dim, num_classes)

        # Kaiming init for GELU is fine with 'fan_in'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data_mri=None, data_clinical=None):
        x = self.in_norm(data_clinical)
        h = self.backbone(x)
        # residual (project input to hidden dim if needed)
        h = h + self.proj(x)
        return self.head(h)

def _collect_xy_from_loader(loader):
    Xs, ys = [], []
    for _, x_cli, y in loader:         # clinical mode yields (dummy, clinical, y)
        Xs.append(x_cli.cpu().numpy())
        ys.append(y.view(-1).cpu().numpy())
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y
def train_xgb_from_loaders(train_loader, val_loader, num_classes=3, class_weights=None):
    Xtr, ytr = _collect_xy_from_loader(train_loader)
    Xva, yva = _collect_xy_from_loader(val_loader)

    # optional per-sample weights from inverse class frequency
    sw = None
    if class_weights is not None:
        wmap = {i: float(class_weights[i]) for i in range(num_classes)}
        sw = np.vectorize(wmap.get)(ytr)

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_estimators=4000,              # early stopping will pick best
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=8
    )

    clf.fit(
        Xtr, ytr,
        sample_weight=sw,
        eval_set=[(Xtr, ytr), (Xva, yva)],
        verbose=200,
        early_stopping_rounds=200
    )

    pred  = clf.predict(Xva)
    proba = clf.predict_proba(Xva)

    print("VAL acc:", round(accuracy_score(yva, pred), 3),
          "bal-acc:", round(balanced_accuracy_score(yva, pred), 3),
          "f1-macro:", round(f1_score(yva, pred, average='macro'), 3))
    print(classification_report(yva, pred, digits=3))
    # print("Confusion matrix:\n", confusion_matrix(yva, pred))

    joblib.dump(clf, "best_clinical_xgb.pkl")
    print("Saved XGBoost → best_clinical_xgb.pkl")

    return clf
