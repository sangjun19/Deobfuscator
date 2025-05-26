import os

def count_c_files(base_dir):
    total_c_files = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".c"):
                total_c_files += 1

    print(f"📁 {base_dir} 디렉토리에서 생성된 .c 파일 총 개수: {total_c_files}")

if __name__ == "__main__":
    BASE_DIR = "data/raw_code/switch/AtCoder"
    # BASE_DIR = "data/temp/AtCoder"
    count_c_files(BASE_DIR)
