#!/usr/bin/env bash
set -euo pipefail

# Re-score the finished PIGNN PF checkpoints for slope/R^2 and the
# PQ-restricted figures, so PIGNN can be compared against the three GridFM
# variants on the collapse diagnostic as well as on RMSE and residuals.
#
# alex compute nodes read /home/vault directly, so there is no staging dance.
#
#   bash dispatch_rescore_pignn.sh
#   GRIDS="SimBench case118" bash dispatch_rescore_pignn.sh

HOST="${HOST:-alex}"
HOME_BASE="${HOME_BASE:-/home/hpc/iwi5/iwi5295h}"
BASE="${BASE:-${HOME_BASE}/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC}"
PY="${PY:-${HOME_BASE}/conda-envs/gridfm-py312/bin/python}"
VAULT="${VAULT:-/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/out}"
GROUP="${GROUP:-pf_pignn_a40_20260812_112919}"
CKPT_DIR="${CKPT_DIR:-${BASE}/results/ckpt/${GROUP}}"
LOG_DIR="${LOG_DIR:-${BASE}/results/logs/${GROUP}}"
OUT_DIR="${OUT_DIR:-${BASE}/results/rescore_pignn/${GROUP}}"

PARTITION="${PARTITION:-a40}"
GRES="${GRES:-gpu:a40:1}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
CPUS="${CPUS:-16}"
# K=40 solver steps per scenario make this far slower than a feed-forward
# rescore, so the batch is the training batch rather than a fixed small one.
BATCH_OVERRIDE="${BATCH_OVERRIDE:-}"

retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 20 ] && return 1; sleep 30; done; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT

# Reuse the training dispatcher's manifest so the parquet names cannot drift.
DISPATCH_SRC="${DISPATCH_SRC:-$(dirname "$0")/dispatch_pf_pignn.sh}"
MANIFEST=()
while IFS= read -r _l; do [ -n "${_l}" ] && MANIFEST+=("${_l}"); done < <(
  sed -n '/^MANIFEST=(/,/^)/p' "${DISPATCH_SRC}" | grep '^  "' | sed 's/^  "//; s/".*$//')
[ "${#MANIFEST[@]}" -gt 0 ] || { echo "no manifest in ${DISPATCH_SRC}" >&2; exit 1; }

lookup() { local w="$1"; for e in "${MANIFEST[@]}"; do
  [ "${e%%|*}" = "${w}" ] && { echo "${e#*|}"; return; }; done; }

GRIDS="${GRIDS:-all}"
if [ "${GRIDS}" = "all" ]; then
  SELECTED=(); for e in "${MANIFEST[@]}"; do SELECTED+=("${e%%|*}"); done
else
  read -r -a SELECTED <<< "${GRIDS}"
fi

retry ssh "${HOST}" "mkdir -p '${OUT_DIR}' '${BASE}/sbatch/Job_out'"

n=0; skipped=""
for g in "${SELECTED[@]}"; do
  info="$(lookup "${g}")" || true
  [ -z "${info}" ] && { echo "unknown grid ${g}" >&2; continue; }
  rest="${info#*|}"; file="${rest%|*}"

  # A checkpoint exists from the first epoch onward -- the driver rewrites
  # best_model.ckpt whenever validation improves -- so its presence does NOT
  # mean the run finished. Gate on the training log's test block instead;
  # scoring a mid-training snapshot would produce numbers that look final.
  fin="$(retry ssh "${HOST}" "grep -alc 'Test physics-loss' '${LOG_DIR}'/pf_pignn_${g}_b*_training_log.txt 2>/dev/null | head -1")"
  if [ -z "${fin}" ]; then skipped="${skipped} ${g}"; continue; fi
  ck="$(retry ssh "${HOST}" "ls '${CKPT_DIR}' 2>/dev/null | grep -E '_${g}_b[0-9]+_[0-9]+_best_model\\.ckpt$' | head -1")"
  if [ -z "${ck}" ]; then skipped="${skipped} ${g}"; continue; fi
  # sed -E, not \+: BSD sed on the submitting machine does not support \+ and
  # silently yields an empty batch, which reaches sbatch as "--BATCH ".
  batch="${BATCH_OVERRIDE:-$(echo "${ck}" | sed -E 's/.*_b([0-9]+)_.*/\1/')}"
  if ! echo "${batch}" | grep -qE '^[0-9]+$'; then
    echo "could not read batch from ${ck}, skipping ${g}" >&2; continue
  fi

  label="rsp_${g}"
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
cp "${VAULT}/${file}" "\${TMPDIR}/p.parquet"

srun ${PY} -u rescore_pignn.py \\
  --PARQUET "\${TMPDIR}/p.parquet" \\
  --ckpt "${CKPT_DIR}/${ck}" \\
  --grid ${g} --BATCH ${batch} \\
  --json_out "${OUT_DIR}/${g}.json"
EOF
  retry scp -q "${STAGE}/${label}.sh" "${HOST}:${BASE}/sbatch/"
  jid="$(retry ssh "${HOST}" "cd '${BASE}/sbatch' && sbatch --parsable '${label}.sh'")"
  echo "   ${jid}  ${label}  (batch ${batch})"
  n=$((n+1))
done
echo "Submitted ${n} PIGNN rescore jobs -> ${OUT_DIR}"
[ -n "${skipped}" ] && echo "Skipped (no finished checkpoint):${skipped}"
