#!/usr/bin/env bash
set -euo pipefail

# GridSFM and LUMINA as AC power-flow surrogates on the Phase-0 backbone corpus
# -- the same 31 grids, the same parquets, the same 1/3 split and seed as the
# GridFM and PIGNN campaigns, so all six are comparable.
#
# Both drivers use the released packages (gridsfm.GridTransformerBackbone,
# lumina_inference HGT) rather than a re-implementation, and both separate the
# OPF path behind `opf_space is not None`. Verified before this campaign:
#   * GridSFM receives the |V| setpoint through the released interface --
#     head_V(cat([h_bus, v_setpoint])) -- and LUMINA through the generator row's
#     vg slot, which is where the PGLib schema puts it.
#   * Reactive power at slack/PV is zeroed by both, since it is the solver's
#     unknown there, not an input.
#   * S_start does not leak the answer: measured on case118, the slack active
#     power differs from the solved injection by 26.5 pu, while P at PV/PQ
#     matches to 3e-9 (it is specified) and Q at PQ matches to 1e-8.
#
# The two models run under different accounts on the same cluster so they do not
# compete for one account's queue limit:
#
#   MODEL=gridsfm bash dispatch_pf_sfm_lumina.sh     # alex,  a40,  iwi5295h
#   MODEL=lumina  bash dispatch_pf_sfm_lumina.sh     # alex2, a100, b313dc11
#   MODEL=lumina GRIDS="case118 SimBench" bash dispatch_pf_sfm_lumina.sh

MODEL="${MODEL:-gridsfm}"

case "${MODEL}" in
  gridsfm)
    HOST="${HOST:-alex}"
    HOME_BASE="${HOME_BASE:-/home/hpc/iwi5/iwi5295h}"
    VAULT="${VAULT:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out}"
    PARTITION="${PARTITION:-a40}"; GRES="${GRES:-gpu:a40:1}"
    DRIVER="train_valid_test_gridsfm.py"
    MANIFEST_SRC="${MANIFEST_SRC:-dispatch_pf_pignn.sh}"
    EXTRA="${EXTRA:---init_mode scratch}"
    ;;
  lumina)
    HOST="${HOST:-alex2}"
    HOME_BASE="${HOME_BASE:-/home/hpc/b313dc/b313dc11}"
    VAULT="${VAULT:-/home/vault/b313dc/b313dc11/data/pf}"
    PARTITION="${PARTITION:-a100}"; GRES="${GRES:-gpu:a100:1}"
    DRIVER="train_valid_test_lumina.py"
    MANIFEST_SRC="${MANIFEST_SRC:-dispatch_pf_gridfm.sh}"
    EXTRA="${EXTRA:---init_mode scratch --gen_setpoint_mode known}"
    ;;
  *) echo "MODEL must be gridsfm or lumina" >&2; exit 1 ;;
esac

BASE="${BASE:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PY="${PY:-${HOME_BASE}/conda-envs/gridfm-py312/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GROUP="${GROUP:-pf_${MODEL}_${PARTITION}_${STAMP}}"
LOG_DIR="${BASE}/results/logs/${GROUP}"
CKPT_DIR="${BASE}/results/ckpt/${GROUP}"

EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-42}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS="${CPUS:-16}"

# Batch is chosen from a measured bus-sample budget, not extrapolated from one
# working configuration. probe_batch_memory.py reports peak GPU memory per batch
# on a large grid; BN_BUDGET is B*N at the largest batch that fit, with headroom.
# Estimating instead of measuring is what cost the GridFM campaign two rounds of
# OOM-killed jobs (166 KB/node estimated against 479 KB/node measured).
BN_BUDGET="${BN_BUDGET:-0}"
BATCH_CAP="${BATCH_CAP:-8}"
BATCH_MIN="${BATCH_MIN:-1}"
if [ "${BN_BUDGET}" = "0" ]; then
  echo "BN_BUDGET is unset. Run probe_batch_memory.py --model ${MODEL} first and" >&2
  echo "pass the measured budget, e.g. BN_BUDGET=20000 MODEL=${MODEL} bash $0" >&2
  exit 1
fi

pick_batch() {  # pick_batch <n_buses>
  # Two statements on purpose: in `local n="$1" b=$((BN_BUDGET/n))` bash expands
  # every word before `local` assigns any of them, so n is still unset inside
  # the arithmetic and the division is by zero.
  local n="$1"
  local b=$(( BN_BUDGET / n ))
  [ "${b}" -gt "${BATCH_CAP}" ] && b="${BATCH_CAP}"
  [ "${b}" -lt "${BATCH_MIN}" ] && b="${BATCH_MIN}"
  echo "${b}"
}

retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 20 ] && return 1; sleep 30; done; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT

# Each account's vault holds the same corpus under its own manifest, so the
# grid -> filename mapping is read from that account's dispatcher rather than
# duplicated here, where it could drift.
MANIFEST=()
while IFS= read -r _l; do [ -n "${_l}" ] && MANIFEST+=("${_l}"); done < <(
  sed -n '/^MANIFEST=(/,/^)/p' "$(dirname "$0")/${MANIFEST_SRC}" \
  | grep '^  "' | sed 's/^  "//; s/".*$//')
[ "${#MANIFEST[@]}" -gt 0 ] || { echo "no manifest in ${MANIFEST_SRC}" >&2; exit 1; }

lookup() { local w="$1"; for e in "${MANIFEST[@]}"; do
  [ "${e%%|*}" = "${w}" ] && { echo "${e#*|}"; return; }; done; }

GRIDS="${GRIDS:-all}"
if [ "${GRIDS}" = "all" ]; then
  SELECTED=(); for e in "${MANIFEST[@]}"; do SELECTED+=("${e%%|*}"); done
else
  read -r -a SELECTED <<< "${GRIDS}"
fi

retry ssh "${HOST}" "mkdir -p '${LOG_DIR}' '${CKPT_DIR}' '${BASE}/sbatch/Job_out'"
echo "[${MODEL}] host ${HOST}  partition ${PARTITION}  group ${GROUP}"
echo "[${MODEL}] BN_BUDGET ${BN_BUDGET}  cap ${BATCH_CAP}  epochs ${EPOCHS}  lr ${LR}"

n=0
for g in "${SELECTED[@]}"; do
  info="$(lookup "${g}")" || true
  [ -z "${info}" ] && { echo "unknown grid ${g}, skipped" >&2; continue; }
  buses="${info%%|*}"; rest="${info#*|}"; file="${rest%|*}"
  batch="$(pick_batch "${buses}")"

  for seed in ${SEEDS}; do
    label="pf_${MODEL}_${g}_b${batch}_s${seed}"
    cat > "${STAGE}/${label}.sh" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${label}
#SBATCH --output=${BASE}/sbatch/Job_out/${label}.out
#SBATCH --error=${BASE}/sbatch/Job_out/${label}.err
#SBATCH --gres=${GRES}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}

set -euo pipefail
export PYTHONPATH=${BASE}:\${PYTHONPATH:-}
cd ${BASE}
# alex compute nodes read /home/vault directly, so the parquet goes straight to
# node-local disk with no \$HOME staging step.
cp "${VAULT}/${file}" "\${TMPDIR}/p.parquet"

srun ${PY} -u ${DRIVER} \\
  --PARQUET "\${TMPDIR}/p.parquet" --task pf \\
  --run_name ${label} \\
  --log_to_file --log_dir ${LOG_DIR} --ckpt_dir ${CKPT_DIR} \\
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \\
  --lazy_parquet --row_group_cache_size 4 \\
  --dataset_complex_dtype complex128 \\
  --BATCH ${batch} --EPOCHS ${EPOCHS} --LR ${LR} --VAL_EVERY 1 \\
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value ${seed} \\
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \\
  ${EXTRA}
EOF
    retry scp -q "${STAGE}/${label}.sh" "${HOST}:${BASE}/sbatch/"
    jid="$(retry ssh "${HOST}" "cd '${BASE}/sbatch' && sbatch --parsable '${label}.sh'")"
    echo "   ${jid}  ${label}  (${buses} buses)"
    n=$((n+1))
  done
done
echo "Submitted ${n} ${MODEL} PF jobs -> ${LOG_DIR}"
