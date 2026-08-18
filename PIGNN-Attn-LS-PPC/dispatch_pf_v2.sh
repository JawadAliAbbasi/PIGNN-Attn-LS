#!/usr/bin/env bash
set -euo pipefail

# Four PF surrogates over the ppNR v2 corpus, one model per grid.
#
#   MODEL=pignn TIER=A bash dispatch_pf_v2.sh
#
# v2 is selected by the _u0clean_sgenpert_pvqvstart_ configuration tag, not by
# the _ppNR_ solver tag: several generations now share the latter, so selecting
# on it would silently mix corpora. manifest_ppnr_v2.sh carries the 31 files.
#
# The data lives in the iwi5295h vault and is readable from the b313dc11 account
# as well, so alex2 needs no copy. helma compute nodes cannot see /home/vault at
# all, so there the login node stages each parquet through $HOME and the job
# frees it once it has a node-local copy.
#
# Tiers place grids on hardware by size:
#   A  10 smallest      alex  a40
#   B  case39..200      alex2 a100
#   C  case300..1888rte alex  a100_80
#   D  GBnetwork..9241  helma h100

CONSTRAINT=""
MODEL="${MODEL:-pignn}"
TIER="${TIER:-A}"

case "${TIER}" in
  A) HOST=alex  ; HOME_BASE=/home/hpc/iwi5/iwi5295h ; PARTITION=a40     ; GRES=gpu:a40:1     ; MEM=48 ; STAGE_VIA_HOME=0
     GRIDS_DEF="case4gs case5 case6ww case9 case14 case24_ieee_rts GBreducednetwork case30 case_ieee30 case33bw" ;;
  B) HOST=alex2 ; HOME_BASE=/home/hpc/b313dc/b313dc11 ; PARTITION=a100  ; GRES=gpu:a100:1    ; MEM=40 ; STAGE_VIA_HOME=0
     GRIDS_DEF="case39 case57 case89pegase SimBench case118 case145 iceland case_illinois200" ;;
  C) HOST=alex  ; HOME_BASE=/home/hpc/iwi5/iwi5295h ; PARTITION=a100     ; GRES=gpu:a100:1    ; MEM=80 ; STAGE_VIA_HOME=0
     # a100_80 is a node feature, not a partition: the a100 partition holds both
     # 19 nodes of 40 GiB and 18 of 80 GiB, selected by --constraint.
     CONSTRAINT=a100_80
     GRIDS_DEF="case300 LVN_heo1 case1354pegase case1888rte" ;;
  D) HOST=helma ; HOME_BASE=/home/hpc/b313dc/b313dc11 ; PARTITION=h100  ; GRES=gpu:1         ; MEM=94 ; STAGE_VIA_HOME=1
     GRIDS_DEF="GBnetwork case2848rte case2869pegase case3120sp ENTSO_E_RealGridTest case6470rte case6495rte case6515rte case9241pegase" ;;
  *) echo "TIER must be A, B, C or D" >&2; exit 1 ;;
esac

# Per-model activation cost, measured rather than guessed, expressed as a
# bus-sample budget for a 48 GiB a40 and scaled to the tier's card below.
#   gridsfm ~1040 KiB/bus-sample (probe on case1354pegase)
#   lumina  ~145  KiB/bus-sample (probe on case1354pegase and case9241pegase)
#   graphkit ~479 KiB/bus-sample (from the GridFM campaign measurement)
#   pignn   K=40 unrolled; budget taken from the batches its first campaign ran
case "${MODEL}" in
  pignn)
    DRIVER=train_valid_test.py ; BASE_BUDGET=27000 ; CAP=32
    MODEL_FLAGS="--BLOCK_DIAG --PINN --d 4 --d_hi 24 --n_heads 8 --num_attn_layers 8 --K 40 \
--use_armijo --armijo_mode geometric_safe --mse_weight 1.0 --physics_loss_form logcosh"
    # armijo_mode=fixed discards the solver step when the line search fails,
    # leaving the output equal to the constant input and the loss detached; it
    # killed 11 of 31 runs in the v1 campaign. geometric_safe keeps the graph
    # edge alive. Measured: fixed -> no gradient, geometric_safe -> 2.707e-01.
    LR_DEF=1e-4 ;;
  graphkit)
    DRIVER=train_valid_test_gridfm.py ; BASE_BUDGET=60000 ; CAP=64
    MODEL_FLAGS="--task pf --gridfm_impl graphkit --hidden_size 48 --num_layers 12 --n_heads 8 \
--zero_init_head --vn_feature_mode log --feature_transform signed_log \
--mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh"
    LR_DEF=5e-4 ;;
  gridsfm)
    DRIVER=train_valid_test_gridsfm.py ; BASE_BUDGET=30000 ; CAP=32
    MODEL_FLAGS="--task pf --init_mode scratch --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh"
    LR_DEF=1e-4 ;;
  lumina)
    DRIVER=train_valid_test_lumina.py ; BASE_BUDGET=200000 ; CAP=32
    MODEL_FLAGS="--task pf --init_mode scratch --mse_weight 1.0 --physics_weight 1e-2 --physics_loss_form logcosh \
--model_config ${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/lumina_ckpt/lumina_config.json"
    LR_DEF=1e-4 ;;
  *) echo "MODEL must be pignn, graphkit, gridsfm or lumina" >&2; exit 1 ;;
esac

# The tier fixes a default partition; PARTITION/GRES stay overridable so a run
# that cannot finish inside the 24 h h100 limit can be sent to preempt, which
# allows 48 h on the same hardware at the cost of being preemptible.
PARTITION="${PARTITION_OVERRIDE:-${PARTITION}}"
GRES="${GRES_OVERRIDE:-${GRES}}"
BN_BUDGET="${BN_BUDGET:-$(( BASE_BUDGET * MEM / 48 ))}"
BATCH_CAP="${BATCH_CAP:-${CAP}}"
LR="${LR:-${LR_DEF}}"
VAULT="${VAULT:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out}"
BASE="${BASE:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PY="${PY:-${HOME_BASE}/conda-envs/gridfm-py312/bin/python}"
DATA="${DATA:-${HOME_BASE}/PIGNN-Attn-LS/data/pf}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-42}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS="${CPUS:-16}"
STAMP="${STAMP:-v2}"
GROUP="${GROUP:-pfv2_${MODEL}_${TIER}_${STAMP}}"
LOG_DIR="${BASE}/results/logs/${GROUP}"
CKPT_DIR="${BASE}/results/ckpt/${GROUP}"
STAGE_BUDGET_GB="${STAGE_BUDGET_GB:-25}"
GRIDS="${GRIDS:-${GRIDS_DEF}}"

retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 20 ] && return 1; sleep 30; done; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT

MANIFEST=()
while IFS= read -r _l; do [ -n "${_l}" ] && MANIFEST+=("${_l}"); done < <(
  sed -n '/^MANIFEST=(/,/^)/p' "$(dirname "$0")/manifest_ppnr_v2.sh" \
  | grep '^  "' | sed 's/^  "//; s/".*$//')
[ "${#MANIFEST[@]}" -eq 31 ] || { echo "manifest has ${#MANIFEST[@]} entries, expected 31" >&2; exit 1; }

lookup() { local w="$1"; for e in "${MANIFEST[@]}"; do
  [ "${e%%|*}" = "${w}" ] && { echo "${e#*|}"; return; }; done; }

pick_batch() {
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

CONSTRAINT_LINE=""
[ -n "${CONSTRAINT}" ] && CONSTRAINT_LINE="#SBATCH --constraint=${CONSTRAINT}"

retry ssh "${HOST}" "mkdir -p '${LOG_DIR}' '${CKPT_DIR}' '${BASE}/sbatch/Job_out' '${DATA}'"
echo "[${MODEL}/tier ${TIER}] host=${HOST} ${PARTITION} (${MEM} GiB) budget B*N<=${BN_BUDGET} lr=${LR} -> ${GROUP}"

n=0
for g in ${GRIDS}; do
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
    CLEANUP="rm -f \"${DATA}/${file}\""
  else
    SRC="${VAULT}/${file}"
    CLEANUP="true"
  fi

  label="pfv2_${MODEL}_${g}_b${batch}"
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
${CONSTRAINT_LINE}

set -euo pipefail
export PYTHONPATH=${BASE}:\${PYTHONPATH:-}
cd ${BASE}
LOCAL_PARQUET="\${TMPDIR}/${g}.parquet"
cp "${SRC}" "\${LOCAL_PARQUET}"
${CLEANUP}

srun ${PY} -u ${DRIVER} \\
  --PARQUET "\${LOCAL_PARQUET}" \\
  --run_name ${label} \\
  --log_to_file --log_dir ${LOG_DIR} --ckpt_dir ${CKPT_DIR} \\
  --PER_UNIT --target_S_base 1e8 --share_grid --share_ybus \\
  --lazy_parquet --row_group_cache_size 4 \\
  --dataset_complex_dtype complex128 \\
  --BATCH ${batch} --EPOCHS ${EPOCHS} --LR ${LR} --VAL_EVERY 1 \\
  --train_ratio 0.3333 --valid_ratio 0.3333 --seed_value ${SEED} \\
  ${MODEL_FLAGS}
EOF
  retry scp -q "${STAGE}/${label}.sh" "${HOST}:${BASE}/sbatch/"
  jid="$(retry ssh "${HOST}" "cd '${BASE}/sbatch' && sbatch --parsable '${label}.sh'")"
  echo "   ${jid}  ${label}  (${buses} buses, batch ${batch})"
  n=$((n+1))
done
echo "Submitted ${n} ${MODEL} tier-${TIER} jobs -> ${LOG_DIR}"
