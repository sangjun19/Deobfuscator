import os

BASE_DIR = "data/raw_code/temp/AtCoder"
total_count = 0

for contest in os.listdir(BASE_DIR):
    contest_dir = os.path.join(BASE_DIR, contest)
    if not os.path.isdir(contest_dir):
        continue

    c_files = [f for f in os.listdir(contest_dir) if f.endswith(".c")]
    count = len(c_files)
    total_count += count
    print(f"{contest}: {count}개")

print(f"\n전체 .c 파일 개수: {total_count}개")
