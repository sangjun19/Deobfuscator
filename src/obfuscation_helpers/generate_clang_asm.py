import os
import subprocess
import time

# 변환할 폴더 및 출력 경로 설정
TARGETS = {
    "org" : "clang_asm/org",
    "ObfusData_clang/VIR": "clang_asm/VIR_asm",
    "ObfusData_clang/DIR": "clang_asm/DIR_asm",
    "ObfusData_clang/INDIR": "clang_asm/INDIR_asm",
    "ObfusData_clang/VIR_DIR": "clang_asm/VIR_DIR_asm",
    "ObfusData_clang/VIR_INDIR": "clang_asm/VIR_INDIR_asm",
    "ObfusData_clang/DIR_INDIR": "clang_asm/DIR_INDIR_asm",
    "ObfusData_clang/INDIR_DIR": "clang_asm/INDIR_DIR_asm",
}

TIMEOUT_SECONDS = 5  # ⏱️ 5초 제한

def compile_to_assembly(src_path, out_path):
    """gcc를 이용해 C 파일을 어셈블리로 컴파일 (시간 제한 포함)"""
    try:
        start = time.time()
        subprocess.run(["clang", "-S", src_path, "-o", out_path], check=True, timeout=TIMEOUT_SECONDS)
        duration = time.time() - start
        print(f"✅ 어셈블리 생성 성공: {src_path} (⏱️ {duration:.2f}s)")
    except subprocess.TimeoutExpired:
        print(f"⚠️ 시간 초과 (>{TIMEOUT_SECONDS}s): {src_path}")
    except subprocess.CalledProcessError:
        print(f"❌ 어셈블리 생성 실패: {src_path}")

for src_dir, asm_dir in TARGETS.items():
    os.makedirs(asm_dir, exist_ok=True)

    # .c 파일 리스트
    c_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".c"))

    for c_file in c_files:
        src_path = os.path.join(src_dir, c_file)
        base_name = os.path.splitext(c_file)[0]
        out_path = os.path.join(asm_dir, f"{base_name}.s")

        compile_to_assembly(src_path, out_path)

print("\n🎯 모든 어셈블리 파일 변환 완료")
