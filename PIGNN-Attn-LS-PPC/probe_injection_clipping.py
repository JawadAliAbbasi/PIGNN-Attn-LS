#!/usr/bin/env python
"""How much of the bus injection do the GridSFM/LUMINA PF graphs actually carry?

Both drivers put the injection on a `load` node as

    xl[:, 0] = clamp(-S.real, min=0)
    xl[:, 1] = clamp(-S.imag, min=0)

and give a `generator` node only to slack and PV buses. A PQ bus whose net
injection is positive -- generation exceeding load, which is what distributed
generation looks like in these scenarios -- therefore reaches the model as a
zero load and no generator at all. The physics loss still scores it against the
true S, so the target depends on information the input does not contain.

This measures, per grid, how many PQ buses and how much injection are lost.

    python probe_injection_clipping.py --PARQUET grid.parquet --n 200
"""

from __future__ import annotations

import argparse

import torch

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--label", default="")
    return p.parse_args()


def main():
    a = parse()
    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8, share_grid=True,
                         share_ybus=True, lazy_row_groups=True, row_group_cache_size=4,
                         complex_dtype="complex128")
    n = min(a.n, len(ds))
    b = collate_blockdiag([ds[i] for i in range(n)])
    bt = b["bus_type"].squeeze(0)
    S = b["S_start"].squeeze(0)

    pq = (bt != 1) & (bt != 2)
    # what the driver would hand the model
    pd_ = torch.clamp(-S.real, min=0.0)
    qd_ = torch.clamp(-S.imag, min=0.0)
    lost_p = (-S.real) - (-pd_)          # nonzero exactly where the clamp bit
    lost_q = (-S.imag) - (-qd_)

    pq_pos_p = pq & (S.real > 0)
    pq_pos_q = pq & (S.imag > 0)
    tot = int(pq.sum())
    print(f"[{a.label or 'grid'}] {n} scenarios, {int(bt.numel()/n)} buses each")
    print(f"  PQ buses                              : {tot} ({100*tot/bt.numel():.1f}% of all)")
    print(f"  PQ buses with P injection > 0 (clamped): {int(pq_pos_p.sum())} "
          f"({100*int(pq_pos_p.sum())/max(tot,1):.1f}% of PQ)")
    print(f"  PQ buses with Q injection > 0 (clamped): {int(pq_pos_q.sum())} "
          f"({100*int(pq_pos_q.sum())/max(tot,1):.1f}% of PQ)")
    ap = S.real[pq].abs().sum()
    lp = S.real[pq_pos_p].abs().sum()
    aq = S.imag[pq].abs().sum()
    lq = S.imag[pq_pos_q].abs().sum()
    print(f"  |P| mass on PQ lost to the clamp      : {float(lp):.4g} / {float(ap):.4g} pu"
          f"  ({100*float(lp)/max(float(ap),1e-12):.1f}%)")
    print(f"  |Q| mass on PQ lost to the clamp      : {float(lq):.4g} / {float(aq):.4g} pu"
          f"  ({100*float(lq)/max(float(aq),1e-12):.1f}%)")
    if int(pq_pos_p.sum()):
        print(f"  largest clamped P injection           : {float(S.real[pq_pos_p].max()):.4g} pu")


if __name__ == "__main__":
    main()
