#!/usr/bin/env bash
set -euo pipefail

PASS_SO="/mnt/e/课业/EECS583/583final/llvm-pgo-dump/build/PathProfPass.so"
RUNTIME_A="/mnt/e/课业/EECS583/583final/llvm-pgo-dump/build/libbl_runtime.a"

ROOT="/mnt/e/课业/EECS583/583final/PolyBenchC-4.2.1"
UTIL="$ROOT/utilities"

CFILE="${1:-$ROOT/linear-algebra/kernels/2mm/2mm.c}"
BASENAME="$(basename "$CFILE" .c)"
KDIR="$(dirname "$CFILE")"

CLANG="$(command -v clang)"
OPT="$(command -v opt)"
LLVMLINK="$(command -v llvm-link)"
LLVM_NM="$(command -v llvm-nm)"

echo "CLANG    = $CLANG"
echo "OPT      = $OPT"
echo "LLVMLINK = $LLVMLINK"
echo "LLVM_NM  = $LLVM_NM"
echo "PASS_SO  = $PASS_SO"
echo "RUNTIME  = $RUNTIME_A"
echo "KDIR     = $KDIR"
echo

cd "$KDIR"
rm -f ./*.bc linked.bc instrumented.bc instrumented.o run.out \
      bl_profile.txt run_stdout_stderr.log static.csv 2>/dev/null || true

# 1) Compile bitcode (disable optnone at O0 so passes run)
"$CLANG" -O0 -g -Xclang -disable-O0-optnone \
  -I "$UTIL" -I "$KDIR" -DDEFAULT_DATASET \
  -emit-llvm -c "$BASENAME.c" -o "$BASENAME.bc"

"$CLANG" -O0 -g -Xclang -disable-O0-optnone \
  -I "$UTIL" -DDEFAULT_DATASET \
  -emit-llvm -c "$UTIL/polybench.c" -o polybench.bc

# 2) Link
"$LLVMLINK" "$BASENAME.bc" polybench.bc -o linked.bc

echo "=== Dump static BL-path features ==="
set +e
"$OPT" -load-pass-plugin="$PASS_SO" \
  -passes="function(dump-bl-static)" \
  linked.bc -disable-output 2> static.csv
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "ERROR: dump-bl-static failed, first lines of static.csv:"
  head -n 30 static.csv
  exit 1
fi
echo "Wrote: $KDIR/static.csv"
echo

# 4) Instrument + build __bl_table
echo "=== Instrumenting + building table ==="
"$OPT" -load-pass-plugin="$PASS_SO" \
  -passes="function(ball-larus-prof),bl-build-table" \
  linked.bc -o instrumented.bc
echo

echo "=== Check symbols in instrumented.bc ==="
"$LLVM_NM" instrumented.bc | grep -E "__bl_table$|__bl_table_size$" || true
echo

# 5) Build executable (disable PIE)
"$CLANG" -O0 -c instrumented.bc -o instrumented.o
"$CLANG" -O0 instrumented.o \
  -no-pie \
  -Wl,--whole-archive "$RUNTIME_A" -Wl,--no-whole-archive \
  -lm -o run.out

# 6) Run -> bl_profile.txt
export BL_PROFILE_OUT="$KDIR/bl_profile.txt"
export BL_PROFILE_PATH="$KDIR/bl_profile.txt"

echo "=== Running ==="
./run.out |& tee run_stdout_stderr.log
echo

echo "=== bl_profile.txt preview ==="
head -n 30 bl_profile.txt || true
echo
echo "DONE: static.csv + bl_profile.txt"
