#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/e/课业/EECS583/583final/PolyBenchC-4.2.1"
RUN="$ROOT/run_bl.sh"              # make run_bl.sh accept an optional cfile (see note below)
JOIN="$ROOT/join_bl.py"            # you said you now have it
OUT="$ROOT/polybench_bl_dataset.csv"

rm -f "$OUT" "$ROOT/bl_failures.txt"
FIRST=1

# find PolyBench kernels: exclude utilities, polybench.c, and *.orig.c
mapfile -t FILES < <(find "$ROOT" -type f -name "*.c" \
  ! -name "polybench.c" \
  ! -name "*.orig.c" \
  ! -path "*/utilities/*" \
  | sort)

echo "Found ${#FILES[@]} .c files"

for cfile in "${FILES[@]}"; do
  echo "=== $cfile ==="

  # Run per-kernel pipeline (do NOT fail the whole batch)
  if ! "$RUN" "$cfile" >/dev/null 2>&1; then
    echo "[WARN] run failed: $cfile" | tee -a "$ROOT/bl_failures.txt"
    continue
  fi

  kdir="$(dirname "$cfile")"
  static="$kdir/static.csv"
  prof="$kdir/bl_profile.txt"
  joined="$kdir/dataset.csv"

  if [ ! -s "$static" ] || [ ! -s "$prof" ]; then
    echo "[WARN] missing outputs for: $cfile" | tee -a "$ROOT/bl_failures.txt"
    continue
  fi

  if ! python3 "$JOIN" "$static" "$prof" "$joined" >/dev/null 2>&1; then
    echo "[WARN] join failed: $cfile" | tee -a "$ROOT/bl_failures.txt"
    continue
  fi

  kernel="$(basename "$cfile" .c)"

  if [ $FIRST -eq 1 ]; then
    # write header with kernel column
    head -n 1 "$joined" | sed 's/^/kernel,/' > "$OUT"
    FIRST=0
  fi

  # append data rows with kernel prefix
  tail -n +2 "$joined" | sed "s/^/$kernel,/" >> "$OUT"
done

echo "Wrote: $OUT"
echo "Failures (if any): $ROOT/bl_failures.txt"
