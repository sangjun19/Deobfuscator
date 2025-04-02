import os

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

# 경로 배열 설정
paths = [
    './data/3d_switch',
    './data/computational_switch',
    './data/nested_switch',
    './data/github_api/github_switch_codes_Ada',
    './data/github_api/github_switch_codes_c',
    './data/github_api/github_switch_codes_c++',
    './data/github_api/github_switch_codes_cobol',
    './data/github_api/github_switch_codes_cuda',
    './data/github_api/github_switch_codes_d',
    './data/github_api/github_switch_codes_Fortran',
    './data/github_api/github_switch_codes_go',
    './data/github_api/github_switch_codes_modula-2',
    './data/github_api/github_switch_codes_Objective-C',
    './data/github_api/github_switch_codes_Objective-C++',
    './data/github_api/github_switch_codes_opencl',
    './data/github_api/github_switch_codes_rust',
]

# 각 경로의 파일 개수 세기
total_file_count = 0
for path in paths:
    try:
        file_count = count_files(path)
        print(f"Number of files in {path}: {file_count}")
        total_file_count += file_count
    except FileNotFoundError:
        print(f"Error: Directory '{path}' not found.")
    except Exception as e:
        print(f"An error occurred with {path}: {e}")

print(f"\nTotal number of files across all directories: {total_file_count}")
