#!/usr/bin/env python
"""Re-score a saved PIGNN-Attn-LS PF checkpoint with the collapse diagnostics.

The GridFM side of this comparison is scored by rescore_pf.py; this is the
same pass for GNSMsg_EdgeSelfAttn so the slope/R^2 and the PQ-restricted
figures are computed identically for all four implementations.

The test split is reproduced rather than re-derived: train_valid_test.py and
train_valid_test_gridfm.py both call random_split with the same lengths and the
same seeded generator, so the split here is the one every campaign held out, and
the RMSE this prints must reproduce the training run's.

    python rescore_pignn.py --PARQUET f.parquet --ckpt run_best.ckpt --grid case118
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import random_split

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn
from prediction_diagnostics import voltage_mae, voltage_regression, angle_wrap


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--BATCH", type=int, default=8)
    # Defaults match dispatch_pf_pignn.sh. A mismatch shows up as a flood of
    # missing keys from load_state_dict, which is checked below rather than
    # warned about, because a partially loaded model still runs and would
    # silently produce meaningless numbers.
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--d_hi", type=int, default=24)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--num_attn_layers", type=int, default=8)
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--use_armijo", action="store_true")
    # These change the forward pass but are NOT stored in the checkpoint, so a
    # wrong value loads cleanly and scores a different function. The defaults
    # below are train_valid_test.py's parser defaults, which are what the
    # campaign ran with -- deliberately not GNSMsg_EdgeSelfAttn's own class
    # defaults, which differ: the class defaults v_limit to True while the
    # driver's --vlimit is a store_true that the dispatcher never sets, and
    # physics_loss_form to "mse" against the driver's "logcosh". Taking the
    # class defaults gave |V| RMSE 1.47e-01 against a logged 8.76e-04.
    #
    # Note the model file on the training cluster is older than this repo's and
    # has no solver_update_mode/preconditioner_* arguments, so this must not
    # pass them: the checkpoints were trained by that version.
    p.add_argument("--vlimit", action="store_true")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--DthetaMax", type=float, default=0.3)
    p.add_argument("--DvmFrac", type=float, default=0.1)
    p.add_argument("--physics_loss_form", default="logcosh")
    p.add_argument("--armijo_mode", default="fixed")
    p.add_argument("--train_ratio", type=float, default=0.3333)
    p.add_argument("--valid_ratio", type=float, default=0.3333)
    p.add_argument("--seed_value", type=int, default=42)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--native_dtype", action="store_true",
                   help="pass the batch through undowncast, as the training driver does")
    p.add_argument("--json_out", default="")
    return p.parse_args()


def to_net(bd):
    """Cast a batch to the dtypes the network runs in, leaving bd untouched.

    The dataset decodes complex128 and the metrics need it, but the network is
    float32/complex64; casting in place would degrade the residual diagnostics.
    """
    net = dict(bd)
    for k in ("Branch_tau", "Branch_shift_deg"):
        if torch.is_tensor(net.get(k)):
            net[k] = net[k].float()
    for k in ("Branch_y_series_from", "Branch_y_series_to", "Branch_y_series_ft",
              "Branch_y_shunt_from", "Branch_y_shunt_to", "Y_shunt_bus", "S_start"):
        if torch.is_tensor(net.get(k)):
            net[k] = net[k].to(torch.complex64)
    net["V_start"] = net["V_start"].float()
    Y = bd["Ybus"]
    if Y.is_sparse:
        Y = Y.coalesce()
        net["Ybus"] = torch.sparse_coo_tensor(
            Y.indices(), Y.values().to(torch.complex64), Y.shape).coalesce()
    else:
        net["Ybus"] = Y.to(torch.complex64)
    return net


def main():
    a = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8,
                         share_grid=True, share_ybus=True, lazy_row_groups=True,
                         row_group_cache_size=4, complex_dtype="complex128")
    n = len(ds)
    n_tr = int(a.train_ratio * n)
    n_va = int(a.valid_ratio * n)
    _, _, test_ds = random_split(
        ds, [n_tr, n_va, n - n_tr - n_va],
        generator=torch.Generator().manual_seed(a.seed_value))
    if a.max_test_samples:
        from torch.utils.data import Subset
        test_ds = Subset(test_ds.dataset, test_ds.indices[:a.max_test_samples])
    loader = torch.utils.data.DataLoader(test_ds, batch_size=a.BATCH, shuffle=False,
                                         collate_fn=collate_blockdiag, num_workers=0)

    model = GNSMsg_EdgeSelfAttn(d=a.d, d_hi=a.d_hi, K=a.K, pinn=True,
                                gamma=a.gamma, v_limit=a.vlimit,
                                n_heads=a.n_heads,
                                num_attn_layers=a.num_attn_layers,
                                use_armijo=a.use_armijo,
                                armijo_mode=a.armijo_mode,
                                dtheta_max=a.DthetaMax, dvm_frac=a.DvmFrac,
                                physics_loss_form=a.physics_loss_form).to(dev)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"state_dict mismatch on {a.grid}: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. The architecture flags do not match "
            f"the checkpoint; scoring it would produce meaningless numbers. "
            f"First missing: {list(missing)[:3]}")
    model.eval()

    P, R, BT = [], [], []
    with torch.no_grad():
        for batch in loader:
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            # The driver's own test loop passes the batch through with no dtype
            # cast at all -- it keeps the dataset's complex128/float64. Casting
            # down to complex64 the way predict_pignn.py does is not neutral for
            # a K=40 iterative solver, so this matches the driver instead.
            net = bd if a.native_dtype else to_net(bd)
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
                # Without this the model treats the whole block-diagonal batch
                # as a single graph, and the per-graph reductions inside it mix
                # scenarios together. It loads and runs, and the error looks
                # like an untrained network: omitting it gave |V| RMSE 2.1e-01
                # against a logged 8.8e-04.
                n_nodes_per_graph=bd["sizes"].to(dev) if "sizes" in bd else None,
                vn_log=net.get("vn_log"),
            )
            V = (out[0] if isinstance(out, (tuple, list)) else out)[..., :2]
            P.append(V.squeeze(0).float().cpu())
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

    # Same convention as rescore_pf.py and as train_valid_test.py's own masking:
    # 1 = slack, 2 = PV, everything else PQ.
    pq = (bt != 1) & (bt != 2)
    n_pq = int(pq.sum())
    extra = {"frac_pq": float(n_pq) / float(bt.numel())}
    if n_pq > 1:
        mae_pq = voltage_mae(Vp[:, pq], Vr[:, pq])
        reg_pq = voltage_regression(Vp[:, pq], Vr[:, pq])
        extra.update({
            "mae_vmag_pq": mae_pq["mae_vmag_pu"],
            "rmse_vmag_pq": mae_pq["rmse_vmag_pu"],
            "slope_vmag_pq": reg_pq["vmag"]["slope"],
            "R2_vmag_pq": reg_pq["vmag"]["R2"],
        })

    out = {"grid": a.grid, "impl": "pignn", "n_bus_samples": int(Vp.shape[1]),
           **extra, "rmse_combined": rmse, **mae,
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
