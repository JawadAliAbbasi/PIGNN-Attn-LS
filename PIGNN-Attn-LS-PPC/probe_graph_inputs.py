#!/usr/bin/env python
"""Check that the PF graph a driver builds actually carries the scenario.

Reconstructs the bus injection from the graph the driver hands the model
(load nodes + generator nodes, mapped back through their link edges) and
compares it against S_start, which is what the physics loss scores against.
Any gap is information the target depends on and the input does not contain.

    python probe_graph_inputs.py --model gridsfm --PARQUET g.parquet --mode signed
"""

from __future__ import annotations

import argparse
import importlib

import torch

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("gridsfm", "lumina"))
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--mode", default="signed", choices=("signed", "legacy"))
    p.add_argument("--n", type=int, default=4)
    return p.parse_args()


class A:
    pass


def main():
    a = parse()
    drv = importlib.import_module(f"train_valid_test_{a.model}")
    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8, share_grid=True,
                         share_ybus=True, lazy_row_groups=True, row_group_cache_size=2,
                         complex_dtype="complex128")
    b = collate_blockdiag([ds[i] for i in range(a.n)])

    args = A()
    args.vmin, args.vmax, args.rate_a = 0.9, 1.1, 1.0
    args.treat_voltage_mismatch_as_transformer = False
    args.gen_setpoint_mode = "known"
    args.pf_injection = a.mode
    make = drv.make_gridsfm_graphs if a.model == "gridsfm" else drv.make_lumina_graphs
    graphs, _ = make(b, args, opf_space=None)

    S = b["S_start"].squeeze(0)
    bt = b["bus_type"].squeeze(0)
    sizes = b["sizes"].tolist()
    off = 0
    tot_gap_p = tot_gap_q = tot_p = 0.0
    for gi, d in enumerate(graphs):
        n = sizes[gi]
        s = S[off:off + n]
        t = bt[off:off + n]
        pq = (t != 1) & (t != 2)
        # what the graph says about each bus
        inj_p = torch.zeros(n, dtype=torch.float64)
        inj_q = torch.zeros(n, dtype=torch.float64)
        if "load" in d.node_types and d["load"].x.numel():
            lb = d["load", "load_link", "bus"].edge_index[1]
            inj_p.index_add_(0, lb, -d["load"].x[:, 0].double())
            inj_q.index_add_(0, lb, -d["load"].x[:, 1].double())
        # PQ buses are the ones the load node must fully describe
        gap_p = (inj_p[pq] - s.real[pq].double()).abs()
        gap_q = (inj_q[pq] - s.imag[pq].double()).abs()
        tot_gap_p += float(gap_p.sum()); tot_gap_q += float(gap_q.sum())
        tot_p += float(s.real[pq].abs().sum())
        off += n

    print(f"[{a.model} / {a.mode}] over {a.n} scenarios, PQ buses only")
    print(f"  sum |injection in graph - S_start| : P {tot_gap_p:.4g} pu, Q {tot_gap_q:.4g} pu")
    print(f"  sum |S_start.P| on PQ              : {tot_p:.4g} pu")
    print(f"  fraction of P unrepresented        : {100 * tot_gap_p / max(tot_p, 1e-12):.2f}%")


if __name__ == "__main__":
    main()
