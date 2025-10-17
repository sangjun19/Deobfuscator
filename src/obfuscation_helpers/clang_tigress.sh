#!/usr/bin/env bash
set -euo pipefail

############################################
# 공통 설정
############################################
JOBS_MAX=2
MEMFREE_GB=12
TIMEOUT_SEC=5

SRC_ORG="./org"
OUT_VM="./ObfusData_clang/VM"
OUT_DIR="./ObfusData_clang/DIR"
OUT_INDIR="./ObfusData_clang/INDIR"
OUT_VM_DIR="./ObfusData_clang/VM_DIR"
OUT_VM_INDIR="./ObfusData_clang/VM_INDIR"
OUT_INDIR_DIR="./ObfusData_clang/INDIR_DIR"
OUT_DIR_INDIR="./ObfusData_clang/DIR_INDIR"

mkdir -p "$OUT_VM" "$OUT_DIR" "$OUT_INDIR" "$OUT_VM_DIR" "$OUT_VM_INDIR" "$OUT_INDIR_DIR" "$OUT_DIR_INDIR"

export TIGRESS_CMD="tigress"
export TIGRESS_ENV="--Environment=x86_64:Linux:Clang:18.0"
export VERBOSITY="--Verbosity=0"
export FUNCTIONS="--Functions=main"
export TIMEOUT_SEC

############################################
# 사전 점검
############################################
command -v "$TIGRESS_CMD" >/dev/null || { echo "❌ tigress not found in PATH"; exit 1; }
command -v timeout >/dev/null || { echo "❌ timeout not found"; exit 1; }
command -v parallel >/dev/null || { echo "❌ GNU parallel not installed"; exit 1; }

############################################
# 실행 함수
############################################
run_stage () {
  local name="$1"        # 예: ORG_VM
  local src="$2"         # 입력 디렉터리
  local out="$3"         # 출력 디렉터리
  local dispatch="$4"    # switch | indirect | direct
  local suffix="$5"      # _vm.c | _indir.c | _dir.c

  [[ -d "$src" ]] || { echo "⚠️  skip '$name' (src not found: $src)"; return 0; }
  mkdir -p "$out"

  echo "▶ $name: src=$src, out=$out, dispatch=$dispatch, suffix=$suffix"

  find "$src" -type f -name '*.c' | sort | \
  parallel --env TIGRESS_CMD,TIGRESS_ENV,VERBOSITY,FUNCTIONS,TIMEOUT_SEC \
    -j "$JOBS_MAX" \
    --memfree "${MEMFREE_GB}G" \
    --bar \
    " base=\$(basename {} .c); \
      out_file=\"$out/\${base}${suffix}\"; \
      if [[ -f \"\$out_file\" ]]; then echo \"⏩ skip: {}\"; exit 0; fi; \
      timeout --signal=KILL \${TIMEOUT_SEC}s \
      nice -n 10 ionice -c2 -n7 \
      \"\$TIGRESS_CMD\" \"\$VERBOSITY\" \"\$TIGRESS_ENV\" \
         --Transform=Virtualize --VirtualizeDispatch=${dispatch} \
         \"\$FUNCTIONS\" --out=\"\$out_file\" \"{}\"; \
      rc=\$?; \
      if [[ \$rc -eq 0 ]]; then \
        echo \"✅ done: {}\"; \
      else \
        echo \"❌ fail(rc=\$rc): {}\"; \
      fi"
}

############################################
# 7단계 실행
############################################
# org → VM, DIR, INDIR
#run_stage "ORG_VM"    "$SRC_ORG" "$OUT_VM"       "switch"   "_vm.c"
#run_stage "ORG_DIR"   "$SRC_ORG" "$OUT_DIR"      "direct"   "_dir.c"
#run_stage "ORG_INDIR" "$SRC_ORG" "$OUT_INDIR"    "indirect" "_indir.c"

# VM → DIR, INDIR
run_stage "VM_DIR"    "$OUT_VM"  "$OUT_VM_DIR"   "direct"   "_dir.c"
run_stage "VM_INDIR"  "$OUT_VM"  "$OUT_VM_INDIR" "indirect" "_indir.c"

# INDIR → DIR
run_stage "INDIR_DIR" "$OUT_INDIR" "$OUT_INDIR_DIR" "direct" "_dir.c"

# DIR → INDIR
run_stage "DIR_INDIR" "$OUT_DIR" "$OUT_DIR_INDIR" "indirect" "_indir.c"

