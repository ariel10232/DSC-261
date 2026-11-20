"""Train and evaluate every detector in the study. Safe to re-run: finished models and
results are skipped, and an interrupted training run resumes from its last epoch.

Usage (on a GPU machine, e.g. DSMLP):
    python run_experiments.py --quick                 # ~5 min end-to-end sanity check
    python run_experiments.py --exp all               # full study  -> runs/full/
    python run_experiments.py --exp baseline          # just one group
    python run_experiments.py --models pgd_e0.005_l0.02

Outputs in --out (default runs/full):
    checkpoints/<model>.pt     best weights + training history
    results/<model>.csv        clean + attacked metrics on the paired test set
    results/<model>_history.csv, results/<model>_examples.npz
    all_results.csv            every model's rows concatenated
    log.txt, run_meta.json

LLM disclosure: written with assistance from Claude (Anthropic), reviewed by the team.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import wmrobust as wm

WATERMARK_STRENGTH = 0.02          # lambda for the main experiments (PSNR ~34 dB)
ATTACK_EPS = [0.0025, 0.005, 0.01, 0.02, 0.03]
ATTACKS = ["fgsm", "pgd"]

# Adversarial-training budgets are kept below / at lambda/2 = 0.01: for eps >= lambda/2 an
# attacker can move a clean and a watermarked copy of the same image to the same point, so
# no detector can be robust there (see report Sec. 3.7).
MODEL_SPECS = {
    "std_l0.02":         dict(strength=0.02,  adv="none", adv_eps=0.0),
    "fgsm_e0.005_l0.02": dict(strength=0.02,  adv="fgsm", adv_eps=0.005),
    "fgsm_e0.01_l0.02":  dict(strength=0.02,  adv="fgsm", adv_eps=0.01),
    "pgd_e0.005_l0.02":  dict(strength=0.02,  adv="pgd",  adv_eps=0.005),
    "pgd_e0.01_l0.02":   dict(strength=0.02,  adv="pgd",  adv_eps=0.01),
    "std_l0.005":        dict(strength=0.005, adv="none", adv_eps=0.0),
    "std_l0.01":         dict(strength=0.01,  adv="none", adv_eps=0.0),
    "std_l0.05":         dict(strength=0.05,  adv="none", adv_eps=0.0),
}
GROUPS = {
    "baseline": ["std_l0.02"],
    "fgsm_at": ["fgsm_e0.005_l0.02", "fgsm_e0.01_l0.02"],
    "pgd_at": ["pgd_e0.005_l0.02", "pgd_e0.01_l0.02"],
    "lambda_sweep": ["std_l0.005", "std_l0.01", "std_l0.05"],
}
GROUPS["all"] = GROUPS["baseline"] + GROUPS["fgsm_at"] + GROUPS["pgd_at"] + GROUPS["lambda_sweep"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="all", choices=sorted(GROUPS))
    p.add_argument("--models", default=None, help="comma-separated model names (overrides --exp)")
    p.add_argument("--out", default=None, help="output dir (default runs/full, or runs/quick with --quick)")
    p.add_argument("--data", default="./data")
    p.add_argument("--quick", action="store_true", help="tiny subsets + 1 epoch, for a smoke test")
    p.add_argument("--epochs", type=int, default=None, help="default 10 (1 with --quick)")
    p.add_argument("--max-train", type=int, default=None, help="originals for train+val (multiple of 20)")
    p.add_argument("--max-test", type=int, default=None, help="test originals (multiple of 20)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=250)
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--eps", default=None, help="comma-separated attack eps (default %s)" % ATTACK_EPS)
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help="random init (offline testing only)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.epochs = args.epochs or 1
        args.max_train = args.max_train or 4000
        args.max_test = args.max_test or 1000
    epochs = args.epochs or 10
    out = Path(args.out or ("runs/quick" if args.quick else "runs/full"))
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    log_file = open(out / "log.txt", "a")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n"); log_file.flush()

    names = args.models.split(",") if args.models else GROUPS[args.exp]
    unknown = [n for n in names if n not in MODEL_SPECS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}; choose from {list(MODEL_SPECS)}")
    eps_list = [float(e) for e in args.eps.split(",")] if args.eps else ATTACK_EPS
    attack_grid = [(a, e) for a in ATTACKS for e in eps_list]

    device = wm.get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log(f"device: {wm.device_name(device)} | torch {torch.__version__} | models: {names}")

    cifar_train, cifar_test = wm.load_cifar10(args.data)
    splits = wm.make_splits(cifar_train, cifar_test, args.max_train, args.max_test)
    train_u8 = wm.to_uint8_tensor(cifar_train, splits["train"])
    val_u8 = wm.to_uint8_tensor(cifar_train, splits["val"])
    test_u8 = wm.to_uint8_tensor(cifar_test, splits["test"])
    train_targets = np.asarray(cifar_train.targets)
    train_labels = wm.balanced_watermark_labels(train_targets[splits["train"]], seed=wm.SEED + 10)
    val_labels = wm.balanced_watermark_labels(train_targets[splits["val"]], seed=wm.SEED + 11)
    pattern = wm.make_watermark_pattern()
    log(f"originals: train {len(train_u8):,} | val {len(val_u8):,} | test {len(test_u8):,} "
        f"(paired test = {2 * len(test_u8):,} images)")

    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "device": wm.device_name(device), "torch": torch.__version__, "python": platform.python_version(),
        "epochs": epochs, "batch_size": args.batch_size, "pgd_steps": args.pgd_steps,
        "attack_eps": eps_list, "attacks": ATTACKS, "pgd_alpha": "eps/4, random start",
        "n_train": len(train_u8), "n_val": len(val_u8), "n_test_originals": len(test_u8),
        "pretrained": not args.no_pretrained, "model_specs": MODEL_SPECS,
    })
    meta.setdefault("timings_sec", {})

    for name in names:
        spec = MODEL_SPECS[name]
        ckpt = out / "checkpoints" / f"{name}.pt"
        partial = out / "checkpoints" / f"{name}.partial.pt"
        res_csv = out / "results" / f"{name}.csv"
        log(f"=== {name}: {spec}")
        wm.set_seed(wm.SEED)
        model = wm.Detector(pretrained=not args.no_pretrained).to(device)

        if ckpt.exists() and not args.force_train:
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            log(f"  loaded {ckpt}")
        else:
            if args.force_train and partial.exists():
                partial.unlink()
            train = wm.WatermarkSplit(train_u8, train_labels, spec["strength"], pattern, device)
            val = wm.WatermarkSplit(val_u8, val_labels, spec["strength"], pattern, device)
            t0 = time.time()
            config = dict(epochs=epochs, batch_size=args.batch_size, adv=spec["adv"],
                          adv_eps=spec["adv_eps"], pgd_steps=args.pgd_steps)
            model, history = wm.train_detector(model, train, val, config, device, partial, log)
            meta["timings_sec"][f"{name}_train"] = time.time() - t0
            torch.save({"model": model.state_dict(), "spec": spec, "config": config,
                        "history": history}, ckpt)
            pd.DataFrame(history).to_csv(out / "results" / f"{name}_history.csv", index=False)
            if partial.exists():
                partial.unlink()
            log(f"  saved {ckpt} ({time.time() - t0:.0f}s)")
            meta_path.write_text(json.dumps(meta, indent=2))

        if res_csv.exists() and not args.force_eval:
            log(f"  results exist, skipping eval ({res_csv})")
            continue
        t0 = time.time()
        df, examples = wm.evaluate_paired(model, test_u8, spec["strength"], pattern, attack_grid,
                                          device, args.eval_batch_size, args.pgd_steps, log=log)
        fid = wm.watermark_fidelity(test_u8, spec["strength"], pattern, device)
        df.insert(0, "model", name)
        for k, v in [("strength", spec["strength"]), ("train_adv", spec["adv"]),
                     ("train_eps", spec["adv_eps"]), *fid.items()]:
            df[k] = v
        df.to_csv(res_csv, index=False)
        np.savez_compressed(out / "results" / f"{name}_examples.npz", **examples)
        meta["timings_sec"][f"{name}_eval"] = time.time() - t0
        meta_path.write_text(json.dumps(meta, indent=2))
        log(f"  wrote {res_csv} ({time.time() - t0:.0f}s)")

    frames = [pd.read_csv(p) for p in sorted((out / "results").glob("*.csv"))
              if not p.name.endswith("_history.csv")]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "all_results.csv", index=False)
        log(f"combined {len(frames)} result files -> {out / 'all_results.csv'}")
    log("done")


if __name__ == "__main__":
    main()
