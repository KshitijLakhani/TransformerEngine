# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

TEST_CASES=(
"test_cp_all_gather"
"test_cp_ring"
"test_cp_load_balanced"
)

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

echo
echo "*** Executing tests in examples/jax/attention/test_context_parallel.py ***"

HAS_FAILURE=0

PIDS=()

cleanup() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Killing process $pid"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing process $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

for TEST_CASE in "${TEST_CASES[@]}"; do
  echo
  echo "=== Starting test: $TEST_CASE ..."

  for i in $(seq 0 $(($NUM_GPUS - 1))); do
    LOG_FILE="${TEST_CASE}_gpu_${i}.log"

    if [ $i -eq 0 ]; then
      echo "=== Live output from process 0 ==="
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs --junitxml=$XML_LOG_DIR/context_parallel_${TEST_CASE}.xml \
        "$TE_PATH/examples/jax/attention/test_context_parallel.py::TestContextParallel::$TEST_CASE" \
        --num-process=$NUM_GPUS \
        --process-id=$i 2>&1 | tee "$LOG_FILE" &
      PID=$!
      PIDS+=($PID)
    else
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs "$TE_PATH/examples/jax/attention/test_context_parallel.py::TestContextParallel::$TEST_CASE" \
        --num-process=$NUM_GPUS \
        --process-id=$i > "$LOG_FILE" 2>&1 &
      PID=$!
      PIDS+=($PID)
    fi
  done

  wait

  if grep -q "SKIPPED" "${TEST_CASE}_gpu_0.log"; then
    echo "... $TEST_CASE SKIPPED"
  elif grep -q "FAILED" "${TEST_CASE}_gpu_0.log"; then
    echo "... $TEST_CASE FAILED"
    HAS_FAILURE=1
  elif grep -q "PASSED" "${TEST_CASE}_gpu_0.log"; then
    echo "... $TEST_CASE PASSED"
  else
    echo "... $TEST_CASE INVALID"
    HAS_FAILURE=1
  fi

  wait
  rm ${TEST_CASE}_gpu_*.log
done

wait

cleanup

exit $HAS_FAILURE
