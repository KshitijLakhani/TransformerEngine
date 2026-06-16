#!/usr/bin/env bash
# Launch scripts/benchmark_te_jax_attention.py as one JAX process per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="$TE_ROOT/scripts/benchmark_te_jax_attention.py"

NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}"
COORDINATOR_ADDRESS="${COORDINATOR_ADDRESS:-127.0.0.1:23456}"
LOG_DIR="${LOG_DIR:-$(mktemp -d -t te_jax_attention_mp_XXXXXX)}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"

mkdir -p "$LOG_DIR"
echo "TE JAX attention multiprocess launch"
echo "  processes   : $NUM_PROCESSES"
echo "  coordinator : $COORDINATOR_ADDRESS"
echo "  log dir     : $LOG_DIR"
echo "  XLA_FLAGS   : ${XLA_FLAGS:-}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

for i in $(seq 0 $((NUM_PROCESSES - 1))); do
  LOG_FILE="$LOG_DIR/proc_${i}.log"
  CMD=(
    python3 -u "$BENCH"
    --distributed
    --coordinator-address "$COORDINATOR_ADDRESS"
    --num-processes "$NUM_PROCESSES"
    --process-id "$i"
    "$@"
  )
  if [ "$i" -eq 0 ]; then
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE" &
  else
    "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  fi
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    STATUS=1
  fi
done

if [ "$STATUS" -ne 0 ]; then
  echo "At least one process failed; logs retained at $LOG_DIR" >&2
  for f in "$LOG_DIR"/proc_*.log; do
    echo "===== $f =====" >&2
    tail -80 "$f" >&2 || true
  done
fi

exit "$STATUS"
