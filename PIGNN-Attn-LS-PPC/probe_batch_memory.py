#!/usr/bin/env python
"""Measure peak GPU memory per batch size for the GridSFM / LUMINA PF drivers.

Sizing 31 grids by extrapolating from one working batch is how the GridFM
campaign lost two rounds of jobs: the estimate was 166 KB/node against a true
479 KB/node. This runs the real forward+backward at several batch sizes and
reports peak allocation, so the dispatcher's budget is measured rather than
guessed.

    python probe_batch_memory.py --model gridsfm --PARQUET big.parquet --batches 1,2,4,8
"""

from __future__ import annotations

import argparse
import importlib
import sys

import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("gridsfm", "lumina"))
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--model_config", default="",
                   help="local LUMINA config.json (required for --model lumina)")
    p.add_argument("--vmin", type=float, default=0.9)
    p.add_argument("--vmax", type=float, default=1.1)
    return p.parse_args()


def main():
    a = parse()
    if not torch.cuda.is_available():
        print("no CUDA; this probe measures GPU memory"); return 1
    dev = "cuda"

    from Dataset_optimized_complex_columns import ChanghunDataset
    from collate_blockdiag_optimized_complex_columns import collate_blockdiag

    drv = importlib.import_module(f"train_valid_test_{a.model}")

    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8,
                         share_grid=True, share_ybus=True, lazy_row_groups=True,
                         row_group_cache_size=2, complex_dtype="complex128")
    n_bus = int(ds[0]["N"])
    print(f"[data] {n_bus} buses")

    # The drivers read a handful of knobs off the args namespace; build the
    # smallest object that satisfies them rather than re-parsing their CLI.
    class A:
        pass
    args = A()
    args.vmin, args.vmax = a.vmin, a.vmax
    args.rate_a = 1.0
    args.treat_voltage_mismatch_as_transformer = False
    args.gen_setpoint_mode = "known"
    args.no_minmax_scaling = False
    args.task = "pf"
    args.init_mode = "scratch"
    args.pretrained_checkpoint = ""
    args.init_checkpoint = ""
    # LUMINA reads its architecture from the released config.json even when
    # training from scratch. Point --model_config at a local copy: compute nodes
    # have no route to the Hub, and hf_hub_download would fail there.
    args.hf_repo_id = "argonne/LUMINA-2M"
    args.model_config = a.model_config

    model = drv.make_model(args, dev)
    fwd = drv.gridsfm_forward if a.model == "gridsfm" else drv.lumina_forward

    for b in [int(x) for x in a.batches.split(",")]:
        if b > len(ds):
            continue
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch = collate_blockdiag([ds[i] for i in range(b)])
            V = fwd(model, batch, args, dev, opf_space=None)
            V = V[0] if isinstance(V, tuple) else V
            ref = batch["V_newton"].to(dev)
            loss = ((V[..., 0] - ref[..., 0].to(V.dtype)) ** 2).mean()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"[mem] batch {b:3d}  B*N {b*n_bus:8d}  peak {peak:7.2f} GiB  "
                  f"{peak * 2**20 / (b * n_bus):8.1f} KiB/node")
            model.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[mem] batch {b:3d}  B*N {b*n_bus:8d}  OOM")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
