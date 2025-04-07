import os

def ensure_include_in_all_c_files(folder_path, include_line='#include <stdlib.h>'):
    for filename in os.listdir(folder_path):
        if filename.endswith('.c'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            has_include = any(line.strip().startswith('#include <stdlib.h>') for line in lines[:10])

            if not has_include:
                print(f"{filename}: #include <stdlib.h> 문이 없어 {include_line}을 추가합니다.")
                lines.insert(0, include_line + '\n')
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.writelines(lines)
            else:
                print(f"{filename}: 이미 #include <stdlib.h> 문이 포함되어 있어 건너뜁니다.")

# 사용 예시
ensure_include_in_all_c_files('./')  # parameter = path of directory
