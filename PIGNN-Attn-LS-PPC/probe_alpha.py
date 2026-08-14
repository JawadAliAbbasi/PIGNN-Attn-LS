#!/usr/bin/env python
"""Report what the line search actually does, per layer.

A line search that always returns alpha=1 is not a line search, and one that
always returns 0 never applies the increment at all -- both would leave
`graphkit_ls` indistinguishable from something else while still training and
producing plausible numbers. This measures which is happening.
"""

from __future__ import annotations

import argparse

import torch

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from train_valid_test_gridfm import make_gridfm_inputs, split_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--BATCH", type=int, default=8)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=48)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8,
                         share_grid=True, share_ybus=True, lazy_row_groups=True,
                         row_group_cache_size=4, complex_dtype="complex128")
    _, _, test_ds = split_dataset(ds, 0.3333, 0.3333, 42)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=a.BATCH, shuffle=False,
                                         collate_fn=collate_blockdiag, num_workers=0)

    from gridfm_graphkit_adapter import make_args, forward_graphkit_parquet
    from genco_ls import GNS_heterogeneous_LS
    model = GNS_heterogeneous_LS(
        make_args("PowerFlow", a.hidden_size, a.num_layers, a.n_heads, 0.0),
        line_search=True).to(dev)
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=dev))
        print(f"[probe] loaded {a.ckpt}")
    else:
        print("[probe] UNTRAINED weights -- mechanics only, not behaviour")
    model.eval()

    per_layer = None
    n_seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= a.batches:
                break
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            forward_graphkit_parquet(model, bd, dev, task_name="PowerFlow",
                                     feature_transform="signed_log",
                                     vn_feature_mode="log")
            if per_layer is None:
                per_layer = [[] for _ in model.alpha_history]
            for i, al in enumerate(model.alpha_history):
                per_layer[i].append(al.float().cpu())
            n_seen += 1

    print(f"[probe] {n_seen} batches, {len(per_layer)} layers, "
          f"{a.BATCH} graphs per batch")
    print(f"{'layer':>5} {'mean':>8} {'min':>8} {'max':>8} "
          f"{'frac a=0':>9} {'frac a=1':>9}")
    all_a = []
    for i, chunks in enumerate(per_layer):
        al = torch.cat(chunks)
        all_a.append(al)
        print(f"{i:5d} {al.mean():8.4f} {al.min():8.4f} {al.max():8.4f} "
              f"{(al == 0).float().mean():9.3f} {(al == 1).float().mean():9.3f}")
    al = torch.cat(all_a)
    z, o = (al == 0).float().mean(), (al == 1).float().mean()
    print(f"\noverall: mean {al.mean():.4f}  frac 0 {z:.3f}  frac 1 {o:.3f}  "
          f"frac strictly between {(1 - z - o):.3f}")
    if z > 0.99:
        print("VERDICT: the increment is essentially never accepted -- the model "
              "is stuck at its initial state.")
    elif o > 0.99:
        print("VERDICT: alpha is always 1 -- the line search is inert and this "
              "is graphkit_inc with extra cost.")
    else:
        print("VERDICT: the line search is active.")


if __name__ == "__main__":
    main()
