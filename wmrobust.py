"""Core library for "Robustness of Vision Watermark Detection Against Adversarial Attacks".

Everything the experiments need lives here so the runner script, the final notebook,
and the report-asset script all use exactly the same code:

* CIFAR-10 splits (identical indices/labels to the team's original notebook)
* the fixed +/-1 additive watermark  x_W = clip(x + lambda * W, 0, 1)
* a ResNet50 detector that takes images in [0, 1] (ImageNet normalization is a layer
  inside the model, so attack budgets epsilon are in plain pixel units)
* FGSM and PGD (L-infinity) attacks
* standard / FGSM / PGD adversarial training with per-epoch resumable checkpoints
* paired evaluation: every test original is scored both clean and watermarked

LLM disclosure: written with assistance from Claude (Anthropic), reviewed by the team.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 0
WATERMARK_SEED = 2026
VAL_FRACTION = 0.10


# --------------------------------------------------------------------------- utils
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


@contextlib.contextmanager
def eval_mode(model: nn.Module):
    """Temporarily put the model in eval mode (BatchNorm uses running stats)."""
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


# --------------------------------------------------------------------------- data
def load_cifar10(root: str = "./data"):
    from torchvision import datasets

    train = datasets.CIFAR10(root, train=True, download=True)
    test = datasets.CIFAR10(root, train=False, download=True)
    return train, test


def to_uint8_tensor(cifar_dataset, indices: np.ndarray) -> torch.Tensor:
    """CIFAR HWC uint8 numpy -> (N, 3, 32, 32) uint8 tensor (kept as uint8 to save memory)."""
    arr = cifar_dataset.data[np.asarray(indices, dtype=int)]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def stratified_indices(targets, selected_count=None, seed=42) -> np.ndarray:
    """Same as the team notebook: equal count per CIFAR class, then shuffled."""
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    if selected_count is None:
        selected_count = len(targets)
    if selected_count % (2 * len(classes)) != 0:
        raise ValueError("selected_count must be divisible by 20 for balanced watermark labels.")
    per_class = selected_count // len(classes)
    indices = np.concatenate([
        rng.choice(np.where(targets == cls)[0], size=per_class, replace=False) for cls in classes
    ])
    return rng.permutation(indices)


def make_splits(cifar_train, cifar_test, max_train=None, max_test=None,
                val_fraction=VAL_FRACTION, seed=SEED) -> Dict[str, np.ndarray]:
    """Reproduces the team notebook's split exactly (same seeds, same order of RNG calls).

    Originals are split before any watermarking, so no source image appears in two partitions.
    """
    all_train = stratified_indices(cifar_train.targets, max_train, seed=seed)
    test = stratified_indices(cifar_test.targets, max_test, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    targets = np.asarray(cifar_train.targets)
    train_list, val_list = [], []
    for cls in np.unique(targets):
        class_idx = all_train[targets[all_train] == cls].copy()
        rng.shuffle(class_idx)
        n_val = round(len(class_idx) * val_fraction)
        if n_val % 2:
            n_val += 1
        val_list.extend(class_idx[:n_val].tolist())
        train_list.extend(class_idx[n_val:].tolist())
    train = rng.permutation(np.array(train_list, dtype=int))
    val = rng.permutation(np.array(val_list, dtype=int))
    assert not set(train).intersection(val)
    return {"train": train, "val": val, "test": test}


def balanced_watermark_labels(object_labels: np.ndarray, seed: int) -> np.ndarray:
    """Exactly half of each CIFAR class gets the watermark (label 1). Same as team notebook."""
    object_labels = np.asarray(object_labels)
    labels = np.zeros(len(object_labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for cls in np.unique(object_labels):
        positions = np.flatnonzero(object_labels == cls)
        if len(positions) % 2:
            raise ValueError("Each source class must have an even count for a 50/50 split")
        rng.shuffle(positions)
        labels[positions[: len(positions) // 2]] = 1
    return labels


# --------------------------------------------------------------------------- watermark
def make_watermark_pattern(seed: int = WATERMARK_SEED) -> torch.Tensor:
    """Fixed pseudorandom pattern W in {-1, +1}^{3x32x32} (same generator call as team notebook)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (3, 32, 32), generator=g, dtype=torch.int64).float().mul(2).sub(1)


def add_watermark(x: torch.Tensor, strength: float, pattern: torch.Tensor) -> torch.Tensor:
    """x: (..., 3, 32, 32) float in [0, 1]. Returns clip(x + strength * W, 0, 1)."""
    return torch.clamp(x + float(strength) * pattern.to(x.device, x.dtype), 0.0, 1.0)


class WatermarkSplit:
    """A split held entirely on the device: uint8 originals + watermark labels.

    Batches come out as float images in [0, 1] with the watermark applied where label == 1.
    Holding the data on the GPU removes the per-image Python data loader bottleneck.
    """

    def __init__(self, images_u8: torch.Tensor, wm_labels, strength: float,
                 pattern: torch.Tensor, device: torch.device):
        self.device = device
        self.images = images_u8.to(device)
        self.labels = torch.as_tensor(np.asarray(wm_labels), dtype=torch.long, device=device)
        self.strength = float(strength)
        self.pattern = pattern.to(device)

    def __len__(self) -> int:
        return self.images.shape[0]

    def get(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx].float().div_(255.0)
        y = self.labels[idx]
        marked = add_watermark(x, self.strength, self.pattern)
        x = torch.where(y.view(-1, 1, 1, 1).bool(), marked, x)
        return x, y

    def batches(self, batch_size: int, shuffle: bool = False, seed: int = 0,
                limit: Optional[int] = None):
        n = len(self) if limit is None else min(limit, len(self))
        if shuffle:
            order = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))[:n]
            order = order.to(self.device)
        else:
            order = torch.arange(n, device=self.device)
        for start in range(0, n, batch_size):
            yield self.get(order[start:start + batch_size])


# --------------------------------------------------------------------------- model
class Detector(nn.Module):
    """ImageNet-pretrained ResNet50 with a 2-way head. Input: images in [0, 1]."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.net = models.resnet50(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


# --------------------------------------------------------------------------- attacks
def _input_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """d CE(model(x), y) / dx in full precision (no autocast: fp16 can underflow to zero gradients)."""
    x = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x).float(), y)
    (grad,) = torch.autograd.grad(loss, x)
    return grad


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Goodfellow et al. (2015): x_adv = clip(x + eps * sign(grad), 0, 1)."""
    if eps <= 0:
        return x.detach().clone()
    with eval_mode(model):
        grad = _input_gradient(model, x, y)
    return torch.clamp(x + eps * grad.sign(), 0.0, 1.0).detach()


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
        alpha: Optional[float] = None, steps: int = 10, random_start: bool = True) -> torch.Tensor:
    """Madry et al. (2018) L-inf PGD with uniform random start. Default step alpha = eps / 4
    (10 steps travel 2.5 * eps, the standard choice so the whole eps-ball is reachable)."""
    if eps <= 0:
        return x.detach().clone()
    alpha = eps / 4.0 if alpha is None else alpha
    x = x.detach()
    if random_start:
        x_adv = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)
    else:
        x_adv = x.clone()
    with eval_mode(model):
        for _ in range(steps):
            grad = _input_gradient(model, x_adv, y)
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp_(0.0, 1.0).detach()
    return x_adv


def random_sign_noise(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Non-adversarial control: x + eps * (random +/-1 per pixel), clipped. Same L-inf size and
    the same 'dense +/-eps' structure as FGSM, but with no information about the model."""
    if eps <= 0:
        return x.detach().clone()
    noise = torch.randint(0, 2, x.shape, device=x.device).to(x.dtype).mul_(2).sub_(1)
    return torch.clamp(x + eps * noise, 0.0, 1.0).detach()


def run_attack(name: str, model, x, y, eps: float, pgd_steps: int = 10) -> torch.Tensor:
    if name == "none" or eps <= 0:
        return x.detach().clone()
    if name == "noise":
        return random_sign_noise(x, eps)
    if name == "fgsm":
        return fgsm(model, x, y, eps)
    if name == "pgd":
        return pgd(model, x, y, eps, steps=pgd_steps)
    raise ValueError(f"unknown attack {name!r}")


# --------------------------------------------------------------------------- image quality
def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for images in [0, 1]; identical images are capped at 100 dB."""
    mse = (a - b).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window(channels: int, size: int = 11, sigma: float = 1.5, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image SSIM (Wang et al. 2004; 11x11 Gaussian window, sigma 1.5, valid region,
    averaged over channels). Matches skimage.metrics.structural_similarity with
    gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1."""
    c = a.shape[1]
    w = _gaussian_window(c, device=a.device).to(a.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a = F.conv2d(a, w, groups=c)
    mu_b = F.conv2d(b, w, groups=c)
    s_aa = F.conv2d(a * a, w, groups=c) - mu_a.pow(2)
    s_bb = F.conv2d(b * b, w, groups=c) - mu_b.pow(2)
    s_ab = F.conv2d(a * b, w, groups=c) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a.pow(2) + mu_b.pow(2) + c1) * (s_aa + s_bb + c2)
    return (num / den).flatten(1).mean(1)


# --------------------------------------------------------------------------- training
DEFAULT_TRAIN_CONFIG = dict(
    epochs=10,
    lr=1e-3,            # proposal: SGD, lr 0.001, momentum 0.9, cross-entropy
    momentum=0.9,
    batch_size=128,
    adv="none",         # "none" | "fgsm" | "pgd"
    adv_eps=0.0,
    pgd_steps=10,
    val_attack_n=1000,  # val images attacked each epoch for model selection (adv training only)
    seed=SEED,
)


@torch.no_grad()
def evaluate_split(model: nn.Module, split: WatermarkSplit, batch_size: int = 500,
                   limit: Optional[int] = None) -> Dict[str, float]:
    with eval_mode(model):
        correct = total = 0
        loss_sum = 0.0
        for x, y in split.batches(batch_size, limit=limit):
            logits = model(x).float()
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return {"loss": loss_sum / total, "accuracy": correct / total}


def robust_accuracy_split(model, split: WatermarkSplit, attack: str, eps: float,
                          pgd_steps: int = 10, batch_size: int = 250, limit: Optional[int] = None,
                          seed: int = 0) -> float:
    torch.manual_seed(seed)
    correct = total = 0
    for x, y in split.batches(batch_size, limit=limit):
        x_adv = run_attack(attack, model, x, y, eps, pgd_steps)
        with torch.no_grad(), eval_mode(model):
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train_detector(model: nn.Module, train: WatermarkSplit, val: WatermarkSplit, config: dict,
                   device: torch.device, partial_path: Optional[Path] = None,
                   log: Callable[[str], None] = print) -> Tuple[nn.Module, List[dict]]:
    """Standard (adv="none") or adversarial training.

    Adversarial training follows the team's design: each step trains on the clean batch
    concatenated with its adversarial version (a 50/50 mix, Goodfellow et al. 2015).
    Model selection replaces the noisy last-batch early-stopping rule: we train a fixed
    number of epochs and keep the epoch with the best validation score (clean accuracy for
    standard training; mean of clean and robust accuracy at the training epsilon for
    adversarial training). State is saved after every epoch so an interrupted run resumes.
    """
    cfg = {**DEFAULT_TRAIN_CONFIG, **config}
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: List[dict] = []
    best_score, best_state, start_epoch = -math.inf, None, 0

    if partial_path is not None and Path(partial_path).exists():
        state = torch.load(partial_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best_score, best_state = state["history"], state["best_score"], state["best_state"]
        start_epoch = state["epoch"] + 1
        log(f"  resumed from {partial_path} at epoch {start_epoch + 1}")

    adv = cfg["adv"]
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train.batches(cfg["batch_size"], shuffle=True, seed=cfg["seed"] * 1000 + epoch):
            if adv != "none":
                x_adv = run_attack(adv, model, x, y, cfg["adv_eps"], cfg["pgd_steps"])
                x, y = torch.cat([x, x_adv]), torch.cat([y, y])
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()

        val_clean = evaluate_split(model, val)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "val_loss": val_clean["loss"],
            "val_accuracy": val_clean["accuracy"],
        }
        if adv != "none":
            row["val_robust_accuracy"] = robust_accuracy_split(
                model, val, adv, cfg["adv_eps"], cfg["pgd_steps"], limit=cfg["val_attack_n"])
            score = 0.5 * (row["val_accuracy"] + row["val_robust_accuracy"])
        else:
            score = row["val_accuracy"]
        row["selection_score"] = score
        row["seconds"] = time.time() - t0
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log("  epoch {epoch:02d}/{n} | train loss {train_loss:.4f} acc {train_accuracy:.2%} | "
            "val loss {val_loss:.4f} acc {val_accuracy:.2%}{rob} | {seconds:.0f}s".format(
                n=cfg["epochs"],
                rob=(f" robust {row['val_robust_accuracy']:.2%}" if adv != "none" else ""),
                **row))
        if partial_path is not None:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(), "history": history, "epoch": epoch,
                        "best_score": best_score, "best_state": best_state}, partial_path)

    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda r: r["selection_score"])["epoch"]
    log(f"  selected epoch {best_epoch} (score {best_score:.4f})")
    return model, history


# --------------------------------------------------------------------------- paired evaluation
def detection_metrics(pred_clean: np.ndarray, pred_marked: np.ndarray) -> Dict[str, float]:
    """Metrics on the paired test set: each original appears once clean (label 0) and once
    watermarked (label 1), so the set is exactly balanced."""
    tn = int((pred_clean == 0).sum()); fp = int((pred_clean == 1).sum())
    tp = int((pred_marked == 1).sum()); fn = int((pred_marked == 0).sum())
    n = len(pred_clean)
    return {
        "accuracy": (tp + tn) / (2 * n),
        "TPR": tp / n,
        "FPR": fp / n,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "pair_accuracy": float(((pred_clean == 0) & (pred_marked == 1)).mean()),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def watermark_fidelity(test_u8: torch.Tensor, strength: float, pattern: torch.Tensor,
                       device: torch.device, batch_size: int = 1000) -> Dict[str, float]:
    """How visible is the watermark itself? PSNR/SSIM of watermarked vs original test images."""
    ps, ss = [], []
    for s in range(0, len(test_u8), batch_size):
        x = test_u8[s:s + batch_size].to(device).float().div_(255.0)
        xw = add_watermark(x, strength, pattern)
        ps.append(psnr(xw, x).cpu()); ss.append(ssim(xw, x).cpu())
    return {"wm_psnr": torch.cat(ps).mean().item(), "wm_ssim": torch.cat(ss).mean().item()}


def evaluate_paired(model: nn.Module, test_u8: torch.Tensor, strength: float,
                    pattern: torch.Tensor, attack_grid: Sequence[Tuple[str, float]],
                    device: torch.device, batch_size: int = 250, pgd_steps: int = 10,
                    seed: int = 0, log: Callable[[str], None] = print,
                    example_count: int = 6) -> Tuple[pd.DataFrame, dict]:
    """Evaluate a detector clean and under each (attack, eps) in attack_grid.

    The attacker uses the true label: watermarked images are pushed toward "clean"
    (evasion) and clean images toward "watermarked" (forgery). Reported per configuration:
      accuracy, TPR, FPR, precision, pair_accuracy (both versions of an original correct),
      evasion_asr = P(attacked watermarked image -> "clean" | detected before attack)  [proposal Sec. 4]
      forgery_asr = P(attacked clean image -> "watermarked" | correctly clean before attack)
      psnr / ssim of attacked image vs. its unattacked input, and the max L-inf change.
    Returns (rows DataFrame, examples dict of numpy arrays for figures).
    """
    pattern = pattern.to(device)
    n = len(test_u8)
    with eval_mode(model):
        # clean predictions (computed once, reused for attack success rates)
        clean_pc, clean_pm = [], []
        with torch.no_grad():
            for s in range(0, n, batch_size):
                x0 = test_u8[s:s + batch_size].to(device).float().div_(255.0)
                x1 = add_watermark(x0, strength, pattern)
                clean_pc.append(model(x0).argmax(1).cpu()); clean_pm.append(model(x1).argmax(1).cpu())
        clean_pc = torch.cat(clean_pc).numpy(); clean_pm = torch.cat(clean_pm).numpy()

    rows = [{"attack": "none", "eps": 0.0, **detection_metrics(clean_pc, clean_pm),
             "evasion_asr": float("nan"), "forgery_asr": float("nan"),
             "psnr": float("nan"), "ssim": float("nan"), "linf": 0.0, "seconds": 0.0}]
    examples = {"clean": None, "marked": None}
    ex_x0 = test_u8[:example_count].to(device).float().div_(255.0)
    ex_x1 = add_watermark(ex_x0, strength, pattern)
    examples["clean"] = ex_x0.cpu().numpy(); examples["marked"] = ex_x1.cpu().numpy()
    with torch.no_grad(), eval_mode(model):
        examples["p_marked_clean"] = model(ex_x0).softmax(1)[:, 1].cpu().numpy()
        examples["p_marked_marked"] = model(ex_x1).softmax(1)[:, 1].cpu().numpy()

    for attack, eps in attack_grid:
        t0 = time.time()
        torch.manual_seed(seed)
        adv_pc, adv_pm, ps, ss, linf = [], [], [], [], 0.0
        for s in range(0, n, batch_size):
            x0 = test_u8[s:s + batch_size].to(device).float().div_(255.0)
            x1 = add_watermark(x0, strength, pattern)
            x = torch.cat([x0, x1])
            y = torch.cat([torch.zeros(len(x0), dtype=torch.long, device=device),
                           torch.ones(len(x1), dtype=torch.long, device=device)])
            x_adv = run_attack(attack, model, x, y, eps, pgd_steps)
            linf = max(linf, (x_adv - x).abs().max().item())
            with torch.no_grad(), eval_mode(model):
                pred = model(x_adv).argmax(1).cpu()
            adv_pc.append(pred[: len(x0)]); adv_pm.append(pred[len(x0):])
            ps.append(psnr(x_adv, x).cpu()); ss.append(ssim(x_adv, x).cpu())
        assert linf <= eps + 1e-5, f"attack exceeded its budget: {linf} > {eps}"
        adv_pc = torch.cat(adv_pc).numpy(); adv_pm = torch.cat(adv_pm).numpy()
        detected = clean_pm == 1
        clean_ok = clean_pc == 0
        row = {
            "attack": attack, "eps": float(eps), **detection_metrics(adv_pc, adv_pm),
            "evasion_asr": float(((adv_pm == 0) & detected).sum() / max(detected.sum(), 1)),
            "forgery_asr": float(((adv_pc == 1) & clean_ok).sum() / max(clean_ok.sum(), 1)),
            "psnr": torch.cat(ps).mean().item(), "ssim": torch.cat(ss).mean().item(),
            "linf": linf, "seconds": time.time() - t0,
        }
        rows.append(row)
        log(f"  {attack:>4} eps={eps:<6} acc {row['accuracy']:.2%} TPR {row['TPR']:.2%} "
            f"FPR {row['FPR']:.2%} evasion ASR {row['evasion_asr']:.2%} "
            f"PSNR {row['psnr']:.1f}dB | {row['seconds']:.0f}s")

        # keep a few attacked watermarked examples for the figures
        ex_y = torch.ones(example_count, dtype=torch.long, device=device)
        torch.manual_seed(seed)
        ex_adv = run_attack(attack, model, ex_x1, ex_y, eps, pgd_steps)
        with torch.no_grad(), eval_mode(model):
            p = model(ex_adv).softmax(1)[:, 1].cpu().numpy()
        examples[f"{attack}_{eps}"] = ex_adv.cpu().numpy()
        examples[f"p_marked_{attack}_{eps}"] = p
    return pd.DataFrame(rows), examples
