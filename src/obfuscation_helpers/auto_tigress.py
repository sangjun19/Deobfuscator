import os
import subprocess
import logging
import sys

SOURCE_DIR = "./org"  # 원본 C 코드 폴더

# ===== 설정값 =====
OUTPUT_DIR_FLATTENING = "./ObfusData/Fla"
OUTPUT_DIR_OPAQUE     = "./ObfusData/Opa"
OUTPUT_DIR_VM         = "./ObfusData/VM"

TIGRESS_CMD = "tigress"
TIGRESS_ENV = "--Environment=x86_64:Linux:Gcc:5.1"
VERBOSITY   = "--Verbosity=0"

TRANSFORM_FLATTEN = "--Transform=Flatten"
TRANSFORM_OPAQUE  = "--Transform=InitOpaque"
TRANSFORM_VM      = "--Transform=Virtualize"

FUNCTIONS = "--Functions=main"

TIMEOUT_SEC = 5  # 각 변환 최대 실행 시간(초)

# ===== 로깅 설정 (콘솔 + 파일 동시 기록) =====
LOG_PATH = "run.log"
logger = logging.getLogger("obfus")
logger.setLevel(logging.INFO)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)

_file = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
_file.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_console.setFormatter(_fmt)
_file.setFormatter(_fmt)

if not logger.handlers:
    logger.addHandler(_console)
    logger.addHandler(_file)

# ===== 출력 폴더 생성 =====
os.makedirs(OUTPUT_DIR_FLATTENING, exist_ok=True)
os.makedirs(OUTPUT_DIR_OPAQUE,     exist_ok=True)
os.makedirs(OUTPUT_DIR_VM,         exist_ok=True)

# ===== C 파일 목록 읽기 =====
try:
    c_files = sorted(f for f in os.listdir(SOURCE_DIR) if f.endswith(".c"))
except FileNotFoundError:
    logger.warning(f"❌ 디렉토리 '{SOURCE_DIR}'가 존재하지 않습니다.")
    raise SystemExit(1)

skip_mode = True  # 이미 처리된 파일까지는 건너뛰기

for idx, filename in enumerate(c_files, 1):
    src_path  = os.path.join(SOURCE_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    out_flatten = os.path.join(OUTPUT_DIR_FLATTENING, f"{base_name}_fla.c")
    out_opaque  = os.path.join(OUTPUT_DIR_OPAQUE,     f"{base_name}_opa.c")
    out_vm      = os.path.join(OUTPUT_DIR_VM,         f"{base_name}_vm.c")

    # 이미 가상화된 결과 파일이 있으면 skip_mode 유지
    if os.path.exists(out_vm):
        logger.info(f"[{idx}/{len(c_files)}] ⏩ {filename} → 이미 존재 (스킵)")
        continue
    else:
        # 처음으로 없는 파일 발견 → 여기서부터 실행
        skip_mode = False

    if skip_mode:
        logger.info(f"[{idx}/{len(c_files)}] ⏩ {filename} → 건너뜀")
        continue

    try:
        # Virtualize
        subprocess.run([
            TIGRESS_CMD, VERBOSITY, TIGRESS_ENV, TRANSFORM_VM,
            "--VirtualizeDispatch=switch",
            FUNCTIONS, f"--out={out_vm}", src_path
        ], check=True, timeout=TIMEOUT_SEC,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logger.info(f"[{idx}/{len(c_files)}] ✅ {filename} 변환 완료")

    except subprocess.TimeoutExpired:
        logger.info(f"[{idx}/{len(c_files)}] ⏱️ {filename} → 변환 시간 초과(>{TIMEOUT_SEC}s)")
    except subprocess.CalledProcessError:
        logger.info(f"[{idx}/{len(c_files)}] ❌ {filename} → Tigress 실행 실패")

logger.info("🎉 모든 파일에 대한 난독화 작업이 완료되었습니다.")
