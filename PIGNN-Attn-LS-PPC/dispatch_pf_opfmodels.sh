#!/usr/bin/env bash
set -euo pipefail

# GridSFM and LUMINA as AC power-flow surrogates over the Phase-0 backbone
# corpus -- the same 31 grids, parquet, split and seed the GridFM and PIGNN
# campaigns used, so the four families stay comparable.
#
#   MODEL=gridsfm GRIDS=all bash dispatch_pf_opfmodels.sh    # alex, a40
#   MODEL=lumina  GRIDS=all bash dispatch_pf_opfmodels.sh    # helma, h100
#
# Both drivers call the released packages (gridsfm.GridTransformerBackbone,
# lumina_inference HGT); neither is a reimplementation. --task pf selects the
# PF graph builder, which differs from the OPF one in that generators carry the
# known injection with a synthesised band instead of the real decision space.

MODEL="${MODEL:-gridsfm}"

case "${MODEL}" in
  gridsfm)
    HOST="${HOST:-alex}"
    HOME_BASE="${HOME_BASE:-/home/hpc/iwi5/iwi5295h}"
    VAULT="${VAULT:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out}"
    PARTITION="${PARTITION:-a40}"; GRES="${GRES:-gpu:a40:1}"
    DRIVER="train_valid_test_gridsfm.py"
    MANIFEST_SRC="${MANIFEST_SRC:-dispatch_pf_pignn.sh}"
    # Measured on case1354pegase, a40: 1.43 GiB at batch 1 rising linearly to
    # 21.50 GiB at batch 16, i.e. ~1040 KiB per bus-sample. 30000 is ~30 GiB,
    # leaving headroom on a 48 GiB a40 for the optimiser and fragmentation.
    BN_BUDGET="${BN_BUDGET:-30000}"
    STAGE_VIA_HOME="${STAGE_VIA_HOME:-0}"
    EXTRA_FLAGS="${EXTRA_FLAGS:-}"
    ;;
  lumina)
    # helma, not alex2: the a100 queue was backed up and helma is idle. Its
    # compute nodes cannot see /home/vault, so the parquet is staged through
    # $HOME first, and --mem is rejected outright on GPU jobs.
    HOST="${HOST:-helma}"
    HOME_BASE="${HOME_BASE:-/home/hpc/b313dc/b313dc11}"
    VAULT="${VAULT:-/home/vault/b313dc/b313dc11/data/pf}"
    PARTITION="${PARTITION:-h100}"; GRES="${GRES:-gpu:1}"
    DRIVER="train_valid_test_lumina.py"
    MANIFEST_SRC="${MANIFEST_SRC:-dispatch_pf_gridfm.sh}"
    # Measured on case1354pegase and case9241pegase, a100: ~145 KiB per
    # bus-sample, linear (5.06 GiB at 9241 buses x batch 4). An h100 has more
    # memory than the a100-40GB this was measured on, so 250000 (~35 GiB) is
    # deliberately below what the measurement permits.
    BN_BUDGET="${BN_BUDGET:-250000}"
    STAGE_VIA_HOME="${STAGE_VIA_HOME:-1}"
    LUMINA_CFG="${LUMINA_CFG:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json}"
    # LUMINA reads its architecture from the released config.json even when
    # training from scratch, and compute nodes have no route to the Hub.
    EXTRA_FLAGS="${EXTRA_FLAGS:---model_config ${LUMINA_CFG}}"
    ;;
  *) echo "MODEL must be gridsfm or lumina" >&2; exit 1 ;;
esac

BASE="${BASE:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PY="${PY:-${HOME_BASE}/conda-envs/gridfm-py312/bin/python}"
DATA="${DATA:-${HOME_BASE}/PIGNN-Attn-LS/data/pf}"
BATCH_CAP="${BATCH_CAP:-32}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-42}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS="${CPUS:-16}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GROUP="${GROUP:-pf_${MODEL}_${PARTITION}_${STAMP}}"
LOG_DIR="${BASE}/results/logs/${GROUP}"
CKPT_DIR="${BASE}/results/ckpt/${GROUP}"
STAGE_BUDGET_GB="${STAGE_BUDGET_GB:-40}"

retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 20 ] && return 1; sleep 30; done; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT

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

pick_batch() {  # buses -> batch
  # Split the declaration: `local n="$1" b=$((BN_BUDGET / n))` evaluates b
  # against an n that is not reliably visible yet, which silently yields a
  # divide-by-zero.
  local n="$1"
  local b=$(( BN_BUDGET / n ))
  [ "$b" -lt 1 ] && b=1
  [ "$b" -gt "${BATCH_CAP}" ] && b="${BATCH_CAP}"
  echo "$b"
}

staged_bytes() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "${HOST}" \
    "du -sb '${DATA}' 2>/dev/null | cut -f1" 2>/dev/null || true
}

retry ssh "${HOST}" "mkdir -p '${LOG_DIR}' '${CKPT_DIR}' '${BASE}/sbatch/Job_out' '${DATA}'"
echo "[${MODEL}] host=${HOST} partition=${PARTITION} group=${GROUP} budget B*N<=${BN_BUDGET}"

n=0
for g in "${SELECTED[@]}"; do
  info="$(lookup "${g}")" || true
  [ -z "${info}" ] && { echo "unknown grid ${g}" >&2; continue; }
  buses="${info%%|*}"; rest="${info#*|}"; file="${rest%|*}"; nbytes="${info##*|}"
  batch="$(pick_batch "${buses}")"

  if [ "${STAGE_VIA_HOME}" = "1" ]; then
    budget=$(( STAGE_BUDGET_GB * 1073741824 ))
    while true; do
      used="$(staged_bytes)"
      [ -z "${used}" ] && { sleep 30; continue; }
      [ $(( used + nbytes )) -le "${budget}" ] && break
      echo "   waiting for room: $(( used / 1073741824 )) GB staged, ${g} needs $(( nbytes / 1073741824 )) GB"
      sleep 120
    done
    retry ssh "${HOST}" "
      set -e
      if [ ! -s '${DATA}/${file}' ] || \
         [ \"\$(stat -c %s '${DATA}/${file}')\" != \"\$(stat -c %s '${VAULT}/${file}')\" ]; then
        cp '${VAULT}/${file}' '${DATA}/${file}.part' && mv '${DATA}/${file}.part' '${DATA}/${file}'
      fi"
    SRC="${DATA}/${file}"
    # The job frees the $HOME copy as soon as it has its own on node-local
    # disk. Without this the staged files accumulate, the dispatcher blocks
    # forever on STAGE_BUDGET_GB, and the corpus is three times that budget.
    CLEANUP="rm -f \"${DATA}/${file}\""
  else
    SRC="${VAULT}/${file}"
    CLEANUP="true"
  fi

  for seed in ${SEEDS}; do
    label="pf_${MODEL}_${g}_b${batch}"
    [ "${seed}" != "42" ] && label="${label}_s${seed}"
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
LOCAL_PARQUET="\${TMPDIR}/${g}.parquet"
cp "${SRC}" "\${LOCAL_PARQUET}"
${CLEANUP}

srun ${PY} -u ${DRIVER} \\
  --PARQUET "\${LOCAL_PARQUET}" --task pf \\
  --run_name ${label} \\
  --log_to_file --log_dir ${LOG_DIR} --ckpt_dir ${CKPT_DIR} \\
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \\
  --lazy_parquet --row_group_cache_size 4 \\
  --dataset_complex_dtype complex128 \\
  --init_mode scratch \\
  --BATCH ${batch} --EPOCHS ${EPOCHS} --LR ${LR} --VAL_EVERY 1 \\
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value ${seed} \\
  --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \\
  ${EXTRA_FLAGS}
EOF
    retry scp -q "${STAGE}/${label}.sh" "${HOST}:${BASE}/sbatch/"
    jid="$(retry ssh "${HOST}" "cd '${BASE}/sbatch' && sbatch --parsable '${label}.sh'")"
    echo "   ${jid}  ${label}  (${buses} buses, batch ${batch})"
    n=$((n+1))
  done
done
echo "Submitted ${n} ${MODEL} jobs -> ${LOG_DIR}"
