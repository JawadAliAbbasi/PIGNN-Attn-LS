#!/usr/bin/env bash
set -euo pipefail

# PIGNN-Attn-LS (GNSMsg_EdgeSelfAttn) as an AC power-flow surrogate on the
# Phase-0 backbone corpus -- the same 31 grids the GridFM mirror and the
# released gridfm_graphkit model were run on, so the three are comparable.
#
# Targets ALEX, not helma. alex compute nodes can read /home/vault directly, so
# there is no staging dance: each job copies its parquet straight from vault to
# node-local disk.
#
#   GRIDS=all bash dispatch_pf_pignn.sh
#   GRIDS="case300 LVN_heo1" EPOCHS=100 bash dispatch_pf_pignn.sh

HOST="${HOST:-alex}"
HOME_BASE="${HOME_BASE:-/home/hpc/iwi5/iwi5295h}"
BASE="${BASE:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PY="${PY:-${HOME_BASE}/conda-envs/gridfm-py312/bin/python}"
VAULT="${VAULT:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out}"

PARTITION="${PARTITION:-a40}"
GRES="${GRES:-gpu:a40:1}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS="${CPUS:-16}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-42}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GROUP="${GROUP:-pf_pignn_${PARTITION}_${STAMP}}"

# Architecture. Taken from this model's OWN capacity sweep on case145
# (results.pdf 4.4), not from GridFM:
#
#   * H=8, d_hi=24, L=8 (65k params) was Pareto-best -- theta 1.77 deg,
#     |V| 1.77e-3.
#   * d_hi=64 "diverged completely (physics loss ~1e5, theta 40-88 deg)".
#     GridFM runs at hidden 48, so copying its width would put this model
#     next to the configuration its own sweep measured as failing. Width is
#     not transferable between the two: GridFM is a plain message-passing
#     encoder, this is an unrolled solver with K correction steps whose
#     gradients accumulate through all of them.
#   * Deeper is not better here: L=8 beat L=10 and L=12 on angle.
#   * d_hi <= 16 with L <= 2 plateaus at theta ~5-6 deg -- capacity, not K,
#     was the binding constraint on case145.
D="${D:-4}"
D_HI="${D_HI:-24}"
HEADS="${HEADS:-8}"
LAYERS="${LAYERS:-8}"
K="${K:-40}"

# Armijo line search is OFF for this campaign, by request. Two things from the
# earlier results to keep in mind while reading the output: without Armijo,
# case39 diverged outright in the previous study, and with it LVN_heo1 got no
# gradient at all because every candidate step was rejected. Neither setting is
# safe everywhere on this corpus.
USE_ARMIJO="${USE_ARMIJO:-0}"
# train_valid_test.py defaults --armijo_mode to "fixed", which discards the
# solver step whenever neither the line search nor its fallback decreases the
# mismatch. On grids where that happens every iteration the returned voltage is
# the constant input, the loss arrives detached, and the driver's zero-grad
# guard turns training into a no-op -- measured on 11 of the 31 grids in the
# 20260812 campaign, which ran without this flag. Set ARMIJO_MODE=geometric_safe
# to keep the graph edge alive. Empty means "do not pass the flag", which
# preserves the original behaviour for reproducing that campaign.
ARMIJO_MODE="${ARMIJO_MODE:-}"

MANIFEST=(
  "case4gs|4|case4gs_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|48706591"   # 36000 rows, 0.05 GB
  "case5|5|case5_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|54879945"   # 36000 rows, 0.05 GB
  "case6ww|6|case6ww_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|63186318"   # 36000 rows, 0.06 GB
  "case9|9|case9_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|68078868"   # 36000 rows, 0.06 GB
  "case14|14|case14_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|96943508"   # 36000 rows, 0.09 GB
  "case24_ieee_rts|24|case24_ieee_rts_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|147104522"   # 36000 rows, 0.14 GB
  "GBreducednetwork|29|GBreducednetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|228966544"   # 36000 rows, 0.21 GB
  "case30|30|case30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|165257198"   # 36000 rows, 0.15 GB
  "case_ieee30|30|case_ieee30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|166379288"   # 36000 rows, 0.15 GB
  "case33bw|33|case33bw_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|167773699"   # 36000 rows, 0.16 GB
  "case39|39|case39_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|197445907"   # 36000 rows, 0.18 GB
  "case57|57|case57_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|288682793"   # 35920 rows, 0.27 GB
  "case89pegase|89|case89pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|510206319"   # 36000 rows, 0.48 GB
  "SimBench|94|SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet|411958895"   # 36000 rows, 0.38 GB
  "case118|118|case118_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|601335980"   # 36000 rows, 0.56 GB
  "case145|145|case145_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|135348914"   # 7885 rows, 0.13 GB
  "iceland|189|iceland_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|764733355"   # 35605 rows, 0.71 GB
  "case_illinois200|200|case_illinois200_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|890231521"   # 36000 rows, 0.83 GB
  "case300|300|case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|698798974"   # 27994 rows, 0.65 GB
  "LVN_heo1|722|LVN_heo1_ppcY_backbone_dc_compile_cNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet|1458388848"   # 36000 rows, 1.36 GB
  "case1354pegase|1354|case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_32627_NR_branchrows_directSI_rg20.parquet|2929450219"   # 32627 rows, 2.73 GB
  "case1888rte|1888|case1888rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|4443938557"   # 36000 rows, 4.14 GB
  "GBnetwork|2224|GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_34424_NR_branchrows_directSI_rg20.parquet|4641628366"   # 34424 rows, 4.32 GB
  "case2848rte|2848|case2848rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet|6677724778"   # 36000 rows, 6.22 GB
  "case2869pegase|2869|case2869pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_35998_NR_branchrows_directSI_rg20.parquet|6786556795"   # 35998 rows, 6.32 GB
  "case3120sp|3120|case3120sp_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_35465_NR_branchrows_directSI_rg20.parquet|7348582034"   # 35465 rows, 6.84 GB
  "ENTSO_E_RealGridTest|6051|ENTSO_E_RealGridTest_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet|12827080698"   # 36000 rows, 11.95 GB
  "case6470rte|6470|case6470rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_25551_NR_branchrows_directSI_rg20.parquet|10565362456"   # 25551 rows, 9.84 GB
  "case6495rte|6495|case6495rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_23332_NR_branchrows_directSI_rg20.parquet|9677738572"   # 23332 rows, 9.01 GB
  "case6515rte|6515|case6515rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_20135_NR_branchrows_directSI_rg20.parquet|8378596594"   # 20135 rows, 7.80 GB
  "case9241pegase|9241|case9241pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_18280_NR_branchrows_directSI_rg20.parquet|11155229512"   # 18280 rows, 10.39 GB
)

# Batch size is bounded by GPU activation memory. Measured through the driver
# itself on case9241pegase (a40, 48 GB), K=40, d_hi=24, H=8, L=8:
#
#   batch 1 -> 11.4 GB      batch 4 -> 44.8 GB
#   batch 2 -> 21.7 GB      batch 8 -> CUDA OOM
#
# i.e. ~1.24 MB per (bus x batch), which is 2.6x GridFM's 479 KB per node on
# the same corpus -- the K unrolled correction steps each keep their
# activations for the backward pass. 28k nodes puts the peak near 34 GB with
# room for fragmentation on a 48 GB a40. Re-measure if K, d_hi, H or L change.
BN_BUDGET="${BN_BUDGET:-28000}"
BATCH_CAP="${BATCH_CAP:-32}"
pick_batch() {
  local n="$1"
  local b=$(( BN_BUDGET / n ))
  [ "$b" -lt 1 ] && b=1
  [ "$b" -gt "$BATCH_CAP" ] && b="$BATCH_CAP"
  echo "$b"
}

retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 20 ] && return 1; sleep 30; done; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT

GRIDS="${GRIDS:-all}"
if [ "${GRIDS}" = "all" ]; then
  SELECTED=(); for e in "${MANIFEST[@]}"; do SELECTED+=("${e%%|*}"); done
else
  read -r -a SELECTED <<< "${GRIDS}"
fi

lookup() {
  local want="$1"
  for e in "${MANIFEST[@]}"; do
    if [ "${e%%|*}" = "${want}" ]; then echo "${e#*|}"; return; fi
  done
}
for g in "${SELECTED[@]}"; do
  if [ -z "$(lookup "${g}")" ]; then
    echo "unknown grid '${g}'. Known:" >&2
    for e in "${MANIFEST[@]}"; do echo "  ${e%%|*}" >&2; done
    exit 1
  fi
done

JOB_OUT="${BASE}/sbatch/Job_out"
LOG_DIR="${BASE}/results/logs/${GROUP}"
CKPT_DIR="${BASE}/results/ckpt/${GROUP}"
SB_DIR="${BASE}/sbatch/${GROUP}"
retry ssh "${HOST}" "mkdir -p '${SB_DIR}' '${LOG_DIR}' '${CKPT_DIR}' '${JOB_OUT}'"

missing=()
for g in "${SELECTED[@]}"; do
  info="$(lookup "${g}")"; rest="${info#*|}"; file="${rest%|*}"
  retry ssh "${HOST}" "test -s '${VAULT}/${file}'" 2>/dev/null || missing+=("${g}: ${file}")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Not present in ${VAULT}:" >&2; printf '  %s\n' "${missing[@]}" >&2; exit 1
fi

echo "GROUP=${GROUP}"
ALL=()
for g in "${SELECTED[@]}"; do
  info="$(lookup "${g}")"; buses="${info%%|*}"
  rest="${info#*|}"; file="${rest%|*}"
  batch="${BATCH:-$(pick_batch "${buses}")}"
  for seed in ${SEEDS}; do
    if [ "$(echo ${SEEDS} | wc -w)" -gt 1 ]; then
      label="pf_pignn_${g}_b${batch}_s${seed}"
    else
      label="pf_pignn_${g}_b${batch}"
    fi
    armijo_flag=""
    [ "${USE_ARMIJO}" = "1" ] && armijo_flag="--use_armijo"
    [ -n "${ARMIJO_MODE}" ] && armijo_flag="${armijo_flag} --armijo_mode ${ARMIJO_MODE}"
    cat > "${STAGE}/${label}.sh" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${label}
#SBATCH --output=${JOB_OUT}/${label}.out
#SBATCH --error=${JOB_OUT}/${label}.err
#SBATCH --gres=${GRES}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}

set -euo pipefail
export PYTHONPATH=${BASE}:\${PYTHONPATH:-}
cd ${BASE}

echo "[gpu] \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[grid] ${g} (${buses} buses), batch ${batch}, K=${K}, d_hi=${D_HI}, H=${HEADS}, L=${LAYERS}"

# alex sees /home/vault from the compute nodes, so the parquet comes straight
# from the master copy; the local copy is for random-read speed, not visibility.
LOCAL_PARQUET="\${TMPDIR}/${g}.parquet"
cp "${VAULT}/${file}" "\${LOCAL_PARQUET}"

srun ${PY} -u train_valid_test.py \\
  --PARQUET "\${LOCAL_PARQUET}" \\
  --run_name ${label} \\
  --log_to_file --log_dir ${LOG_DIR} --ckpt_dir ${CKPT_DIR} \\
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \\
  --lazy_parquet --row_group_cache_size 4 \\
  --dataset_complex_dtype complex128 --BLOCK_DIAG \\
  --BATCH ${batch} --EPOCHS ${EPOCHS} --LR ${LR} --VAL_EVERY 1 \\
  --d ${D} --d_hi ${D_HI} --n_heads ${HEADS} --num_attn_layers ${LAYERS} --K ${K} \\
  --PINN --mse_weight 1.0 --physics_loss_form logcosh \\
  --seed_value ${seed} ${armijo_flag}
EOF
    retry scp -q "${STAGE}/${label}.sh" "${HOST}:${SB_DIR}/"
    jid="$(retry ssh "${HOST}" "cd '${SB_DIR}' && sbatch --parsable '${label}.sh'")"
    echo "   ${jid}  ${label}"
    ALL+=("${label}")
  done
done

echo "Submitted ${#ALL[@]} PIGNN PF jobs."
echo "Logs: ${LOG_DIR}"
