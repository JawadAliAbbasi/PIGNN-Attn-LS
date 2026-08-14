#!/usr/bin/env python
"""Show why the PIGNN PF campaign learned nothing on eleven grids.

armijo_mode="fixed" -- the parser default, and what dispatch_pf_pignn.sh used
since it never passes --armijo_mode -- discards the step entirely when neither
the line search nor its fallback decreases the mismatch:

    if not accepted:                       # GNSMsg_SelfAttention_armijo.py
        ...
        if ok:                             # fallback also has to pass
            v, th, m = v2, th2, m + a * dm
        # else: v, th, m are left untouched

The network's dv/dth then never enter the state. If that happens at every one
of the K solver steps, the returned voltage is the input V_start, which is a
constant: out.requires_grad is False, the loss is detached, and the driver's
zero-grad guard turns the optimiser step into a no-op.

This probe runs one real batch through both modes and reports whether the
output is still attached to the parameters.

    python probe_armijo_detach.py --PARQUET grid.parquet --n 4
"""

from __future__ import annotations

import argparse

import torch

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--d_hi", type=int, default=24)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--num_attn_layers", type=int, default=8)
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--modes", default="fixed,geometric,geometric_safe")
    return p.parse_args()


def main():
    a = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8,
                         share_grid=True, share_ybus=True, lazy_row_groups=True,
                         row_group_cache_size=4, complex_dtype="complex128")
    bd = collate_blockdiag([ds[i] for i in range(a.n)])
    bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bd.items()}

    net = dict(bd)
    for k in ("Branch_tau", "Branch_shift_deg"):
        net[k] = net[k].float()
    for k in ("Branch_y_series_from", "Branch_y_series_to", "Branch_y_series_ft",
              "Branch_y_shunt_from", "Branch_y_shunt_to", "Y_shunt_bus", "S_start"):
        if torch.is_tensor(net.get(k)):
            net[k] = net[k].to(torch.complex64)
    net["V_start"] = net["V_start"].float()
    Y = bd["Ybus"]
    if Y.is_sparse:
        Y = Y.coalesce()
        net["Ybus"] = torch.sparse_coo_tensor(Y.indices(),
                                              Y.values().to(torch.complex64),
                                              Y.shape).coalesce()
    else:
        net["Ybus"] = Y.to(torch.complex64)

    print(f"[data] {a.n} scenarios, {int(bd['sizes'].sum())} buses total")

    for mode in a.modes.split(","):
        torch.manual_seed(0)
        model = GNSMsg_EdgeSelfAttn(
            d=a.d, d_hi=a.d_hi, K=a.K, pinn=True, gamma=0.9, v_limit=False,
            n_heads=a.n_heads, num_attn_layers=a.num_attn_layers,
            use_armijo=True, armijo_mode=mode,
            dtheta_max=0.3, dvm_frac=0.1, physics_loss_form="logcosh",
        ).to(dev)
        model.train()
        out = model(
            bus_type=net["bus_type"],
            Branch_f_bus=net["Branch_f_bus"], Branch_t_bus=net["Branch_t_bus"],
            Branch_status=net["Branch_status"], Branch_tau=net["Branch_tau"],
            Branch_shift_deg=net["Branch_shift_deg"],
            Branch_y_series_from=net["Branch_y_series_from"],
            Branch_y_series_to=net["Branch_y_series_to"],
            Branch_y_series_ft=net["Branch_y_series_ft"],
            Branch_y_shunt_from=net["Branch_y_shunt_from"],
            Branch_y_shunt_to=net["Branch_y_shunt_to"],
            Is_trafo=net.get("Is_trafo"),
            Y=net["Ybus"], S=net["S_start"], V0=net["V_start"],
            Y_shunt_bus=net.get("Y_shunt_bus"),
            n_nodes_per_graph=bd["sizes"].to(dev),
            vn_log=net.get("vn_log"),
        )
        V = (out[0] if isinstance(out, (tuple, list)) else out)[..., :2]

        # Did the solver move at all, and is the result still differentiable?
        moved = float((V[..., 0] - net["V_start"][..., 0].to(V.dtype)).abs().max())
        attached = V.requires_grad
        mse = ((V[..., 0] - bd["V_newton"][..., 0].to(V.dtype)) ** 2).mean()
        gnorm = None
        if mse.requires_grad:
            model.zero_grad(set_to_none=True)
            mse.backward()
            gnorm = sum(float(p.grad.norm()) for p in model.parameters()
                        if p.grad is not None)
        print(f"[{mode:15s}] |V| moved max {moved:.3e}   "
              f"output attached to params: {attached}   "
              f"grad norm: {'DETACHED - no gradient' if gnorm is None else f'{gnorm:.3e}'}")


if __name__ == "__main__":
    main()
