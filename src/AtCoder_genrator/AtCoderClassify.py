import os
import shutil

SOURCE_ROOT = "data/raw_code/temp/AtCoder"
DEST_SWITCH = "data/raw_code/switch/AtCoder"
DEST_NON_SWITCH = "data/raw_code/non-switch/AtCoder"

os.makedirs(DEST_SWITCH, exist_ok=True)
os.makedirs(DEST_NON_SWITCH, exist_ok=True)

for contest_dir in os.listdir(SOURCE_ROOT):
    contest_path = os.path.join(SOURCE_ROOT, contest_dir)
    if not os.path.isdir(contest_path):
        continue
    print(f"📁 {contest_dir} 디렉토리 처리 중...")
    
    for filename in os.listdir(contest_path):
        if not filename.endswith(".c"):
            continue

        filepath = os.path.join(contest_path, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

                if "switch" in content:
                    dest_path = os.path.join(DEST_SWITCH, f"{contest_dir}_{filename}")
                else:
                    dest_path = os.path.join(DEST_NON_SWITCH, f"{contest_dir}_{filename}")

                shutil.copy(filepath, dest_path)

        except Exception as e:
            print(f"⚠️ 오류 발생: {filepath} → {e}")

print("✅ 분류 완료")
