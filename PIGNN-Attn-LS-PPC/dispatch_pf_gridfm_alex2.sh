#!/usr/bin/env bash
set -euo pipefail

# GridFM as an AC power-flow surrogate on the PPC test cases.
#
# The model is our own re-implementation (GridFMHeteroSurrogate in
# train_valid_test_gridfm.py), not the released gridfm_graphkit model -- see the
# report's Section 1. --task pf is the script's default: given an injection,
# predict the voltages. No OPF decision space is read.
#
# Defaults target helma. helma compute nodes see ONLY /home/hpc, so the parquet
# must sit under $HOME before the job starts; the job then copies it to the
# node-local disk (14 TB) so training reads from local NVMe rather than NFS.
#
#   GRIDS="case14 case118" bash dispatch_pf_gridfm.sh
#   GRIDS=all bash dispatch_pf_gridfm.sh
#   GRIDS="case2848rte" BATCH=16 EPOCHS=60 bash dispatch_pf_gridfm.sh
#
# Grid names are matched against the manifest below, not globbed on the cluster,
# so a typo fails here rather than after the job is queued.

HOST="${HOST:-alex2}"
HOME_BASE="${HOME_BASE:-/home/hpc/b313dc/b313dc11}"
BASE="${BASE:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PY="${PY:-${HOME_BASE}/conda-envs/gridfm-py312/bin/python}"
DATA="${DATA:-${HOME_BASE}/PIGNN-Attn-LS/data/pf}"
# Master copy. helma compute nodes cannot see /home/vault, and $HOME has only
# ~42 GB under its soft quota against ~61 GB of parquet, so the full set lives
# in vault and each grid is rolled into $HOME just before its job is queued.
VAULT="${VAULT:-/home/vault/b313dc/b313dc11/data/pf}"
# Ceiling on how much staged parquet may sit in $HOME at once. The corpus is
# ~86 GB against ~42 GB of soft-quota headroom, so grids are submitted as space
# frees rather than all at once. Each job deletes its own staged copy as soon as
# it has the file on node-local disk, so usage drops a few minutes after a job
# starts -- the dispatcher does not wait for training to finish.
WAVE_BUDGET_GB="${WAVE_BUDGET_GB:-30}"
# Give up waiting for room after this long and submit anyway (seconds).
STAGE_WAIT_MAX="${STAGE_WAIT_MAX:-14400}"

PARTITION="${PARTITION:-a100}"
GRES="${GRES:-gpu:a100:1}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
# 16, not the h100 campaign's 32: alex a100 nodes are 128 cores across 8 GPUs,
# so a 1-GPU job may claim at most 16 or sbatch refuses it outright with
# "Too many tasks/CPUs per GPU requested".
CPUS="${CPUS:-16}"
# helma rejects --mem on GPU jobs ("Memory corresponding to number of GPUs
# will be allocated automatically"), so memory is not requestable here: the
# batch size in pick_batch is what keeps the job inside the node's share.
EPOCHS="${EPOCHS:-40}"
LR="${LR:-5e-4}"
# Seeds to train per grid. PFDelta ("PFDelta: A Benchmark Dataset for Power
# Flow", section 4.4) trains every model three times from different random
# initialisations and reports the standard deviation across runs as the error
# bar. This project has needed exactly that: the same configuration gave GridFM
# 1.040e-3 on one cluster and 4.300e-3 on another for case14, and GridSFM's
# scratch runs swapped rank between clusters -- differences a single run cannot
# distinguish from a real effect.
#
# SEEDS=42 (the default) reproduces the campaigns run so far. SEEDS="42 43 44"
# submits one job per seed per grid; label and run_name carry the seed so the
# runs do not collide, and summarize_rescore.py reports mean +/- sd when it
# finds more than one seed for a grid.
SEEDS="${SEEDS:-42}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GROUP="${GROUP:-pf_gridfm_${PARTITION}_${STAMP}}"

# Model size. The defaults match the OPF runs so PF and OPF results stay
# comparable; override for a capacity sweep.
# "mirror" = our local GridFMHeteroSurrogate, "graphkit" = the released
# gridfm_graphkit GNS_heterogeneous (per-layer physics feedback + PhysicsDecoderPF).
# Neither has published weights, so both are scratch-only.
IMPL="${IMPL:-mirror}"
# Power flow only: hold |V| at PV/slack and the angle at the slack to the given
# inputs instead of predicting a correction there. The first mirror campaign ran
# without this and its |V| error included buses where the answer was an input --
# measured at 2.3e-3 pu mean on case118, over 46% of buses. PIN=1 labels its runs
# "mirrorpin" so both campaigns can coexist.
PIN="${PIN:-0}"
PIN_FLAG=""
IMPL_TAG="${IMPL}"
if [ "${PIN}" = "1" ]; then PIN_FLAG="--pin_known"; IMPL_TAG="${IMPL}pin"; fi
HIDDEN="${HIDDEN:-48}"
LAYERS="${LAYERS:-12}"
HEADS="${HEADS:-8}"

# grid | buses | parquet basename | bytes
#
# This is the Phase-0 backbone corpus: the pandapower-solved set tagged _ppNR_
# in the filename. LVN_heo1 carries _cNR_ instead because pandapower cannot
# solve that grid, so it was produced with the custom NR oracle. The solver tag
# is the reliable selector -- do NOT pick files by size or by the _A_ tag, which
# is a different (older, custom-NR) scenario set entirely.
#
# A u0clean tag appears in the SimBench / ENTSO-E / LVN_heo1 names but not the
# earlier ones, so glob on the _ppNR_ substring rather than on a full pattern.
#
# Bus counts are measured from each file's bus_number column, not assumed.
# Files rewritten to 20 rows/group carry _rg20; SimBench, ENTSO-E and LVN_heo1
# were already at 20 and are used as-is.
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

# Batch size.
#
# collate_blockdiag now builds a SPARSE block-diagonal Y-bus, so the old
# quadratic wall is gone. Measured on case9241pegase (9241 buses, the largest
# grid in the corpus) with the sparse path:
#
#   batch   B*N      nnz        peak RSS   GPU
#       1   9241      37,655     4.2 GB      7 MB
#      64  591,424  2,409,920    4.2 GB    422 MB
#
# Host RSS is flat -- it is dataset overhead, not batch -- and GPU grows
# linearly from a tiny base. Under the previous dense layout the same grid
# needed 81 GB for the Y-bus alone at batch 8 and was OOM-killed.
#
# What binds now is the MODEL's activation memory on the GPU, not the Y-bus.
# Measured directly (probe_model_mem.py: real fwd+bwd of GridFMHeteroSurrogate
# at hidden 48 / 12 layers / 8 heads, case9241pegase):
#
#   batch    B*N      peak GPU
#       1    9,241      4.3 GB
#       4   36,964     16.9 GB
#       8   73,928     33.6 GB
#      12  110,892     50.3 GB
#      16  147,856     67.0 GB
#      24  221,784     OOM (93.3 GB usable)
#
# Dead linear at ~464 KB per node. An earlier version of this comment guessed
# 166 KB from a single OOM point and was wrong by 2.8x, which is why batch 27
# also died -- extrapolating from the point where a job *failed* measures the
# allocator's ceiling, not the model's demand. 120k nodes puts the peak near
# 54 GB with real headroom.
#
# Re-measure if HIDDEN/LAYERS/HEADS change: the constant belongs to the model
# shape, not to the data.
# 50000, not the h100 campaign's 120000: an alex a100 is a 40 GB card against
# the h100's ~94 GB usable, and 50000 nodes at the measured 479 KB/node is
# ~24 GB, the same ~60% utilisation the h100 runs had. Note the consequence for
# comparability: the cap of 64 binds for every grid below ~780 buses, so those
# train at exactly the helma batch, and only the largest grids differ.
BN_BUDGET="${BN_BUDGET:-50000}"
BATCH_CAP="${BATCH_CAP:-64}"
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

lookup() {  # grid -> "buses|file", empty if unknown
  local want="$1"
  for e in "${MANIFEST[@]}"; do
    if [ "${e%%|*}" = "${want}" ]; then echo "${e#*|}"; return; fi
  done
}

# Validate the whole selection before touching the cluster.
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

# Everything must exist in vault; a grid missing there is a staging error and is
# worth failing on before anything is queued.
missing=()
for g in "${SELECTED[@]}"; do
  info="$(lookup "${g}")"; rest="${info#*|}"; file="${rest%|*}"
  retry ssh "${HOST}" "test -s '${VAULT}/${file}'" 2>/dev/null || missing+=("${g}: ${file}")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Not present in ${VAULT} on ${HOST}:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

# Copy vault -> $HOME. Only the login node can do this: vault is invisible on
# compute nodes.
stage_one() {
  local file="$1"
  retry ssh "${HOST}" "
    set -e
    mkdir -p '${DATA}'
    if [ -s '${DATA}/${file}' ] && \
       [ \"\$(stat -c %s '${DATA}/${file}')\" = \"\$(stat -c %s '${VAULT}/${file}')\" ]; then
      exit 0
    fi
    cp '${VAULT}/${file}' '${DATA}/${file}.part'
    mv '${DATA}/${file}.part' '${DATA}/${file}'
  "
}

unstage() { retry ssh "${HOST}" "rm -f '${DATA}/$1'"; }

wait_for_jobs() {  # block until none of the given job ids are in the queue
  local ids="$*"
  [ -z "${ids}" ] && return 0
  local pat; pat="$(echo "${ids}" | tr ' ' '|')"
  while true; do
    local n
    n="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${HOST}" \
          "squeue -u \$USER -h -o %i 2>/dev/null | grep -cE '^(${pat})\$'" 2>/dev/null || echo "")"
    [ -z "${n}" ] && { sleep 60; continue; }   # transient ssh failure, not "done"
    [ "${n}" = "0" ] && return 0
    echo "    ${n} job(s) still running ..."
    sleep 120
  done
}

echo "GROUP=${GROUP}"
ALL=()

# $HOME usage in bytes of the staging directory.
staged_bytes() {
  # Must not abort the dispatcher: the NHR gateway refuses connections often
  # enough that a bare ssh under `set -e` will kill a multi-hour run. An empty
  # result means "unknown", and the caller retries.
  ssh -o BatchMode=yes -o ConnectTimeout=20 "${HOST}" \
    "du -sb '${DATA}' 2>/dev/null | cut -f1" 2>/dev/null || true
}

budget=$(( WAVE_BUDGET_GB * 1073741824 ))
for g in "${SELECTED[@]}"; do
  info="$(lookup "${g}")"; buses="${info%%|*}"
  rest="${info#*|}"; file="${rest%|*}"; nbytes="${info##*|}"
  batch="${BATCH:-$(pick_batch "${buses}")}"
  for seed in ${SEEDS}; do
  # Only tag the label when sweeping, so single-seed runs keep the names the
  # existing results and reports already use.
  if [ "$(echo ${SEEDS} | wc -w)" -gt 1 ]; then
    label="pf_${IMPL_TAG}_${g}_b${batch}_s${seed}"
  else
    label="pf_${IMPL_TAG}_${g}_b${batch}"
  fi

  # Wait until this grid fits. Jobs delete their own staged copy once they have
  # it on local disk, so usage falls as jobs start -- no need to wait for a
  # whole group to finish training.
  # alex compute nodes can read /home/vault, so nothing is staged into $HOME
  # and nothing has to be waited for.
  echo "   ${g} ($(( nbytes / 1073741824 )) GB) read directly from vault"
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
echo "[grid] ${g} (${buses} buses), batch ${batch}"

# Training reads randomly; NFS random reads are slow, node-local NVMe is not.
LOCAL_PARQUET="\${TMPDIR}/${g}.parquet"
cp "${VAULT}/${file}" "\${LOCAL_PARQUET}"
# Nothing to clean up: the source is the vault master copy, which must survive.

srun ${PY} -u train_valid_test_gridfm.py \\
  --PARQUET "\${LOCAL_PARQUET}" --task pf --gridfm_impl ${IMPL} ${PIN_FLAG} \\
  --run_name ${label} \\
  --log_to_file --log_dir ${LOG_DIR} --ckpt_dir ${CKPT_DIR} \\
  --PER_UNIT --target_S_base 1e8 --share_grid --lazy_parquet --row_group_cache_size 4 \\
  --dataset_complex_dtype complex128 \\
  --BATCH ${batch} --EPOCHS ${EPOCHS} --LR ${LR} --VAL_EVERY 1 \\
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value ${seed} \\
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \\
  --hidden_size ${HIDDEN} --num_layers ${LAYERS} --n_heads ${HEADS} \\
  --zero_init_head --vn_feature_mode log --feature_transform signed_log
EOF
  retry scp -q "${STAGE}/${label}.sh" "${HOST}:${SB_DIR}/"
  jid="$(retry ssh "${HOST}" "cd '${SB_DIR}' && sbatch --parsable '${label}.sh'")"
  echo "   ${jid}  ${label}"
  ALL+=("${label}")
  done
done

echo "Submitted ${#ALL[@]} PF jobs."
echo "Logs: ${LOG_DIR}"
