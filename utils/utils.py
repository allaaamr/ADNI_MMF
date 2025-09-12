import os, math, random, glob
import numpy as np
import nibabel as nib
from typing import Sequence, Tuple, List
from data.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
from argparse import Namespace
from models.ResNet3D import *
from models.ClinicalMLP import *
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
from models.utils import *
from models.FT_Transformer import *
from models.LateFusion import *
from models.HybridFusion import *
import pickle


def train(datasets: tuple, args: Namespace, device):
    train_loader, val_loader = datasets

    counts = get_label_counts(train_loader, num_classes=3, device=device)
    K, N = counts.numel(), counts.sum().clamp_min(1)
    class_weights = (N / (K * counts.clamp_min(1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)



    
    if args.mode=="radio":
        model = ResNet3D(num_classes=3, in_channels=1, layers=(2,2,2,2), widths=(32,64,128,256)).to(device)
    if args.mode=="clinical":
        model = ClinicalMLP2(num_features=8, num_classes=3).to(device)
    if args.mode=="late_fusion":
        model = LateFusion()
    if args.mode=="hybrid":
        model=HybridFusion()

    print("model init")
    model = model.to(device)

    # ===== ONE LINE: initialize final bias to log-priors =====
    init_output_bias_from_counts(model, counts)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    scaler = None

    best_val_acc, best_path = 0.0, f"best_{args.mode}.pth"
    
    
    for epoch in range(1, args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion,  device, scaler)
        va_loss, va_acc = evaluate(model, val_loader,criterion, device)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": va_acc, "target_size": args.target_size}, best_path)
            print(f" Saved new best to {best_path} (val acc {va_acc:.3f})")

    return model

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):

    model.train()
    loss_meter, correct, total = 0.0, 0, 0
    for batch_idx, batch in enumerate(loader):

        optimizer.zero_grad(set_to_none=True)
            
        data_MRI, data_clinical, y  = batch

        data_MRI = data_MRI.to(device, non_blocking=True)   # (B,1,D,H,W)
        data_clinical = data_clinical.to(device, non_blocking=True)    
        y = y.to(device, non_blocking=True)    

        data_MRI = as_channels_last_3d(data_MRI)    


        # if scaler is not None:
        #     with torch.autocast(device_type="cuda", dtype=torch.float16):
        #         logits = model(data_MRI, data_clinical)
        #         loss = criterion(logits.float(), y)
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        logits = model( data_MRI, data_clinical)
        loss = criterion(logits.float(), y)
        loss.backward()
        optimizer.step()

        loss_meter += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return loss_meter/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter, correct, total = 0.0, 0, 0
    for batch_idx, batch in enumerate(loader):

        data_MRI, data_clinical, y  = batch

        data_MRI = data_MRI.to(device, non_blocking=True)   # (B,1,D,H,W)
        data_clinical = data_clinical.to(device, non_blocking=True)    
        y = y.to(device, non_blocking=True)    

        data_MRI = as_channels_last_3d(data_MRI)    

        #pdb.set_trace() 
        logits = model(data_mri =data_MRI, data_clinical=data_clinical)

        loss = criterion(logits, y)
        loss_meter += loss.item() * data_clinical.size(0)
        pred = logits.argmax(1)
        # print("pred ", pred)
        correct += (pred == y).sum().item()
        # print("correct ", correct)
        total += y.numel()
    return loss_meter/total, correct/total





def get_label_counts(loader, num_classes=3, device="cpu"):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for b in loader:
        y = b[-1]
        counts += torch.bincount(y, minlength=num_classes).cpu()
    return counts.to(device).float()

@torch.no_grad()
def init_output_bias_from_counts(model: nn.Module, counts: torch.Tensor):
    m = model.module if hasattr(model, "module") else model
    pri = (counts / counts.sum().clamp_min(1)).clamp_min(1e-6).log()
    last = None
    for mod in reversed(list(m.modules())):
        if isinstance(mod, nn.Linear) and mod.out_features == counts.numel() and mod.bias is not None:
            last = mod; break
    if last is not None:
        last.bias.copy_(pri.to(last.bias.device))


def pickle_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)