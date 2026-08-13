#!/usr/bin/env python
"""Re-score a saved PF checkpoint with MAE and the regression diagnostics.

Adds what the original runs did not report -- per-channel MAE and the
predicted-vs-reference regression of prediction_diagnostics -- without
retraining. Reads the same parquet and the same test split (seed 42, thirds),
so the RMSE it prints must reproduce the training run's final RMSE; that
agreement is the check that the split and the checkpoint line up.

    python rescore_pf.py --PARQUET f.parquet --ckpt run_best.pt --grid case118
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from train_valid_test_gridfm import (GridFMHeteroSurrogate, make_gridfm_inputs,
                                     split_dataset, cap_subset)
from prediction_diagnostics import voltage_mae, voltage_regression, angle_wrap


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--grid", required=True)
    # mirrorpin is the same architecture and the same weights as mirror; the
    # only difference is that the head's correction is masked off where the
    # quantity is a given input. Loading a mirrorpin checkpoint as "mirror"
    # succeeds silently and scores a different function, so the variant has to
    # be named explicitly.
    p.add_argument("--impl", choices=("mirror", "mirrorpin", "graphkit"),
                   default="mirror")
    p.add_argument("--BATCH", type=int, default=16)
    p.add_argument("--hidden_size", type=int, default=48)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--train_ratio", type=float, default=0.3333)
    p.add_argument("--valid_ratio", type=float, default=0.3333)
    p.add_argument("--seed_value", type=int, default=42)
    p.add_argument("--vn_feature_mode", default="log")
    p.add_argument("--feature_transform", default="signed_log")
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--json_out", default="")
    return p.parse_args()


def main():
    a = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8,
                         share_grid=True, share_ybus=True, lazy_row_groups=True,
                         row_group_cache_size=4, complex_dtype="complex128")
    _, _, test_ds = split_dataset(ds, a.train_ratio, a.valid_ratio, a.seed_value)
    if a.max_test_samples:
        test_ds = cap_subset(test_ds, a.max_test_samples)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=a.BATCH, shuffle=False,
                                         collate_fn=collate_blockdiag, num_workers=0)

    sample = collate_blockdiag([ds[0]])
    sd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in sample.items()}
    if a.impl == "graphkit":
        from gridfm_graphkit_adapter import build_graphkit_model, forward_graphkit_parquet
        model = build_graphkit_model(task_name="PowerFlow", hidden_size=a.hidden_size,
                                     num_layers=a.num_layers,
                                     attention_head=a.n_heads).to(dev)
    else:
        x, _, ea, _, _, _ = make_gridfm_inputs(sd, dev, a.vn_feature_mode,
                                               a.feature_transform, opf_space=None)
        model = GridFMHeteroSurrogate(
            input_bus_dim=x["bus"].shape[-1], input_gen_dim=x["gen"].shape[-1],
            edge_dim=ea[("bus", "connects", "bus")].shape[-1],
            hidden_size=a.hidden_size, heads=a.n_heads, num_layers=a.num_layers,
            pin_known=(a.impl == "mirrorpin"),
        ).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()

    # Pool the whole test set before summarising: a regression slope cannot be
    # averaged over batches, and neither can a median.
    P, R, BT = [], [], []
    with torch.no_grad():
        for batch in loader:
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if a.impl == "graphkit":
                V = forward_graphkit_parquet(model, bd, dev, task_name="PowerFlow",
                                             feature_transform=a.feature_transform,
                                             vn_feature_mode=a.vn_feature_mode)
            else:
                x, ei, ea, _, _, _ = make_gridfm_inputs(bd, dev, a.vn_feature_mode,
                                                        a.feature_transform, opf_space=None)
                V = model(x, ei, ea)
            P.append(V.squeeze(0).cpu())
            R.append(bd["V_newton"].squeeze(0).float().cpu())
            BT.append(bd["bus_type"].squeeze(0).cpu())
    Vp = torch.cat(P, 0).unsqueeze(0)
    Vr = torch.cat(R, 0).unsqueeze(0)
    bt = torch.cat(BT, 0)

    mae = voltage_mae(Vp, Vr)
    reg = voltage_regression(Vp, Vr)
    dmag = (Vp[..., 0] - Vr[..., 0]).to(torch.float64)
    dang = angle_wrap((Vp[..., 1] - Vr[..., 1]).to(torch.float64))
    rmse = float(((dmag ** 2).mean() + (dang ** 2).mean()) ** 0.5)

    # The two implementations do not predict the same thing. The released model
    # masks |V| to PQ buses for power flow -- at PV and slack it copies the
    # input, which is exact by construction -- while the mirror predicts every
    # bus. Pooled |V| error therefore flatters graphkit. These PQ-restricted
    # figures are the like-for-like comparison.
    #
    # The parquet pipeline does NOT use the MATPOWER numbering: here 1 = slack,
    # 2 = PV, and everything else is PQ (verified on case118: 1 slack, 53 PV,
    # 64 PQ). train_valid_test_gridfm builds its one-hots the same way
    # (REF_H from bt == 1, PV_H from bt == 2, PQ_H from the rest).
    pq = (bt != 1) & (bt != 2)
    n_pq = int(pq.sum())
    if n_pq > 1:
        mae_pq = voltage_mae(Vp[:, pq], Vr[:, pq])
        reg_pq = voltage_regression(Vp[:, pq], Vr[:, pq])
        pq_extra = {
            "frac_pq": float(n_pq) / float(bt.numel()),
            "mae_vmag_pq": mae_pq["mae_vmag_pu"],
            "rmse_vmag_pq": mae_pq["rmse_vmag_pu"],
            "slope_vmag_pq": reg_pq["vmag"]["slope"],
            "R2_vmag_pq": reg_pq["vmag"]["R2"],
        }
    else:
        pq_extra = {"frac_pq": float(n_pq) / float(bt.numel())}

    out = {"grid": a.grid, "impl": a.impl, "n_bus_samples": int(Vp.shape[1]),
           **pq_extra,
           "rmse_combined": rmse, **mae,
           "slope_vmag": reg["vmag"]["slope"], "R2_vmag": reg["vmag"]["R2"],
           "std_pred_vmag": reg["vmag"]["std_pred"], "std_ref_vmag": reg["vmag"]["std_ref"],
           "slope_sin": reg["theta_sin"]["slope"], "R2_sin": reg["theta_sin"]["R2"],
           "slope_cos": reg["theta_cos"]["slope"], "R2_cos": reg["theta_cos"]["R2"]}
    print("RESCORE " + json.dumps(out))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
