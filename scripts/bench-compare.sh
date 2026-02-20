#!/usr/bin/env bash
# Benchmark regression checker.
# Compares current benchmark output against a baseline (if present).
# Fails if any benchmark regresses more than the threshold.
set -euo pipefail

THRESHOLD_PERCENT="${BENCH_REGRESSION_THRESHOLD:-15}"
BENCH_FILE="${1:-bench-output.txt}"

if [ ! -f "$BENCH_FILE" ]; then
  echo "No benchmark file found at $BENCH_FILE"
  exit 0
fi

echo "=== Benchmark Results ==="
cat "$BENCH_FILE"
echo ""

# Check if we have a baseline to compare against
BASELINE_FILE="bench-baseline.txt"
if [ ! -f "$BASELINE_FILE" ]; then
  echo "No baseline found at $BASELINE_FILE — skipping regression check."
  echo "To set a baseline, run: cp $BENCH_FILE $BASELINE_FILE"
  exit 0
fi

echo "=== Comparing against baseline ==="
REGRESSION_FOUND=0

while IFS= read -r line; do
  # Parse bencher format: "test <name> ... bench: <ns> ns/iter (+/- <var>)"
  if echo "$line" | grep -q "bench:"; then
    NAME=$(echo "$line" | awk '{print $2}')
    CURRENT_NS=$(echo "$line" | grep -oP 'bench:\s+\K[0-9,]+' | tr -d ',')

    # Find matching baseline
    BASELINE_NS=$(grep "$NAME" "$BASELINE_FILE" | grep -oP 'bench:\s+\K[0-9,]+' | tr -d ',' || true)

    if [ -n "$BASELINE_NS" ] && [ "$BASELINE_NS" -gt 0 ] 2>/dev/null; then
      CHANGE=$(( (CURRENT_NS - BASELINE_NS) * 100 / BASELINE_NS ))
      if [ "$CHANGE" -gt "$THRESHOLD_PERCENT" ]; then
        echo "REGRESSION: $NAME regressed by ${CHANGE}% (baseline: ${BASELINE_NS}ns, current: ${CURRENT_NS}ns)"
        REGRESSION_FOUND=1
      else
        echo "OK: $NAME changed by ${CHANGE}%"
      fi
    fi
  fi
done < "$BENCH_FILE"

if [ "$REGRESSION_FOUND" -eq 1 ]; then
  echo ""
  echo "FAILED: Benchmark regression(s) exceeded ${THRESHOLD_PERCENT}% threshold."
  exit 1
fi

echo ""
echo "PASSED: No benchmark regressions detected."
exit 0
