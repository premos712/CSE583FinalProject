#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
ROOT="/mnt/e/课业/EECS583/583final/PolyBenchC-4.2.1"
UTILITIES_DIR="$ROOT/utilities"
PASS_SO="/mnt/e/课业/EECS583/583final/llvm-pgo-dump/build/DumpProfilePass.so"
OUT_CSV="$ROOT/polybench_dataset.csv"

echo "Using ROOT        = $ROOT"
echo "Using UTILITIES   = $UTILITIES_DIR"
echo "Using PASS_SO     = $PASS_SO"
echo "Output CSV        = $OUT_CSV"
echo

# Start fresh
rm -f "$OUT_CSV"

# Temp file for each kernel's pass output (with header)
TMP_CSV="$(mktemp)"

# Find all .c files except utilities/polybench.c and *orig.c
find "$ROOT" -type f -name '*.c' \
    ! -path "$UTILITIES_DIR/*" \
    ! -name '*orig.c' \
    -print0 | \
while IFS= read -r -d '' CFILE; do
    BASENAME="$(basename "$CFILE" .c)"
    DIRNAME="$(dirname "$CFILE")"

    echo "=== Processing: $CFILE ==="

    (
        cd "$DIRNAME"

        # Clean old artifacts in this kernel dir
        rm -f default.profraw default.profdata \
              *.bc linked.bc profiled.ll \
              *_pgo 2>/dev/null || true

        # 1) Build instrumented binary with PGO (O2, normal opts)
        clang -O2 -fprofile-generate \
              -I "$UTILITIES_DIR" \
              -DDEFAULT_DATASET \
              "$CFILE" "$UTILITIES_DIR/polybench.c" \
              -lm \
              -o "${BASENAME}_pgo"

        # 2) Run to produce default.profraw
        LLVM_PROFILE_FILE=default.profraw "./${BASENAME}_pgo"

        # 3) Merge to default.profdata
        llvm-profdata merge -o default.profdata default.profraw

        # 4) Build bitcode with profile-use for kernel + polybench (O2)
        clang -O2 -fprofile-use=default.profdata \
              -I "$UTILITIES_DIR" \
              -DDEFAULT_DATASET \
              -emit-llvm -c "$CFILE" -o "${BASENAME}.bc"

        clang -O2 -fprofile-use=default.profdata \
              -I "$UTILITIES_DIR" \
              -DDEFAULT_DATASET \
              -emit-llvm -c "$UTILITIES_DIR/polybench.c" -o polybench.bc

        # 5) Link into one module and disassemble to .ll
        llvm-link "${BASENAME}.bc" polybench.bc -o linked.bc
        llvm-dis linked.bc -o profiled.ll

        # 6) Run your pass. It prints CSV to stderr, so capture to TMP_CSV
        opt -load-pass-plugin="$PASS_SO" \
            -passes="function(dump-profile)" \
            profiled.ll -disable-output 2> "$TMP_CSV"
    )

    # Append TMP_CSV to OUT_CSV
    if [ -s "$TMP_CSV" ]; then
        # If OUT_CSV is empty, write header once (prefix with kernel,)
        if [ ! -s "$OUT_CSV" ]; then
            head -n 1 "$TMP_CSV" | sed 's/^/kernel,/' > "$OUT_CSV"
        fi

        # Append data lines (skip header from pass) and prefix with kernel name
        tail -n +2 "$TMP_CSV" | sed "s/^/$BASENAME,/" >> "$OUT_CSV"
    fi

    echo
done

rm -f "$TMP_CSV"

echo "Done. Combined dataset at: $OUT_CSV"
