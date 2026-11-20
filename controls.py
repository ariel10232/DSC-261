"""Two control experiments on every saved detector (forward passes only; GPU optional).

1. Random noise. FGSM adds a dense +/-eps sign pattern. Adding *random* +/-eps sign noise of the
   same size (no knowledge of the model) tests whether the detector reacts to "dense +/-noise" in
   general rather than to the specific watermark pattern W.

2. Oracle attacker (the worst case in the report's proposition). An attacker who knows both the
   clean original x0 and the watermarked copy x1 tries two moves per image, each at most eps
   per pixel, and wins if either one fools the detector:
       toward the midpoint z = (x0 + x1) / 2:   x -> x + clip(z - x, -eps, eps)
       toward the other copy:                   x0 -> x0 + clip(x1 - x0, -eps, eps), and vice versa
   At eps >= lambda/2 both copies can reach z exactly, so at most one of them can be classified
   correctly (accuracy <= 50%, pair accuracy = 0). At eps >= lambda the watermarked copy can be
   turned back into the original. Comparing PGD with this attacker shows how far a gradient attack
   is from the true worst case.

    python controls.py --run runs/full        # -> runs/full/controls.csv

LLM disclosure: written with assistance from Claude (Anthropic), reviewed by the team.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import wmrobust as wm
from run_experiments import ATTACK_EPS, MODEL_SPECS


@torch.no_grad()
def oracle_eval(model, test_u8, strength, pattern, eps, device, batch_size=500):
    """Paired metrics under the oracle attacker described in the module docstring."""
    pc, pm, cpc, cpm = [], [], [], []
    for s in range(0, len(test_u8), batch_size):
        x0 = test_u8[s:s + batch_size].to(device).float().div_(255.0)
        x1 = wm.add_watermark(x0, strength, pattern)
        z = (x0 + x1) / 2
        cpc.append(model(x0).argmax(1).cpu()); cpm.append(model(x1).argmax(1).cpu())
        # clean copy: attack wins if any move makes the detector say "watermarked" (1)
        c_mid = model(x0 + (z - x0).clamp(-eps, eps)).argmax(1)
        c_swp = model(x0 + (x1 - x0).clamp(-eps, eps)).argmax(1)
        pc.append(((c_mid == 1) | (c_swp == 1)).long().cpu())
        # watermarked copy: still detected only if every move leaves the prediction at 1
        m_mid = model(x1 + (z - x1).clamp(-eps, eps)).argmax(1)
        m_swp = model(x1 + (x0 - x1).clamp(-eps, eps)).argmax(1)
        pm.append(((m_mid == 1) & (m_swp == 1)).long().cpu())
    pc, pm, cpc, cpm = (torch.cat(t).numpy() for t in (pc, pm, cpc, cpm))
    row = {"attack": "oracle", "eps": float(eps), **wm.detection_metrics(pc, pm)}
    row["evasion_asr"] = float(((pm == 0) & (cpm == 1)).sum() / max((cpm == 1).sum(), 1))
    row["forgery_asr"] = float(((pc == 1) & (cpc == 0)).sum() / max((cpc == 0).sum(), 1))
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/full")
    p.add_argument("--data", default="./data")
    p.add_argument("--max-test", type=int, default=None)
    args = p.parse_args(argv)
    run = Path(args.run)
    device = wm.get_device()
    cifar_train, cifar_test = wm.load_cifar10(args.data)
    splits = wm.make_splits(cifar_train, cifar_test, max_test=args.max_test)
    test_u8 = wm.to_uint8_tensor(cifar_test, splits["test"])
    pattern = wm.make_watermark_pattern().to(device)
    eps_list = sorted(set(ATTACK_EPS) | {0.025, 0.04, 0.05})  # also covers lambda/2 and lambda for lambda=0.05
    frames = []
    for name, spec in MODEL_SPECS.items():
        ckpt = run / "checkpoints" / f"{name}.pt"
        if not ckpt.exists():
            print("skip (no checkpoint):", name)
            continue
        model = wm.Detector(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        model.eval()
        torch.manual_seed(0)
        df, _ = wm.evaluate_paired(model, test_u8, spec["strength"], pattern,
                                   [("noise", e) for e in eps_list], device, example_count=1,
                                   log=lambda m, n=name: print(n, m.strip()))
        rows = df[df.attack == "noise"].to_dict("records")
        for e in eps_list:
            r = oracle_eval(model, test_u8, spec["strength"], pattern, e, device)
            print(f"{name} oracle eps={e:<6} acc {r['accuracy']:.2%} pair {r['pair_accuracy']:.2%} "
                  f"evasion ASR {r['evasion_asr']:.2%}")
            rows.append(r)
        out = pd.DataFrame(rows)
        out.insert(0, "model", name)
        out["strength"], out["train_adv"], out["train_eps"] = spec["strength"], spec["adv"], spec["adv_eps"]
        frames.append(out)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(run / "controls.csv", index=False)
    print("wrote", run / "controls.csv")


if __name__ == "__main__":
    main()
