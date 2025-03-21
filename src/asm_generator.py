import subprocess
import os

file_name = "3d_switch"

def compile_to_assembly(filename):
    # C 파일이 있는 디렉토리로 이동
    # cwd = os.path.dirname(filename)
    # os.chdir(cwd)

    # C 파일 이름에서 확장자 제거
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # 어셈블리 코드를 저장할 디렉토리
    output_dir = f"./data/assembly_code/{file_name}"

    # 디렉토리가 없으면 생성
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # GCC로 어셈블리 코드 생성
    compile_command = ["gcc", "-S", filename, "-o", f"{output_dir}/{base_filename}.s"]
    try:
        subprocess.run(compile_command, check=True)
        print(f"{filename} 어셈블리 코드 생성 성공")
    except subprocess.CalledProcessError as e:
        print(f"{filename} 어셈블리 코드 생성 실패: {e}")

# C 파일 목록
c_files = [f"./data/3d_switch/{file_name}_{i}.c" for i in range(1, 1001)]

# 각 C 파일을 어셈블리 코드로 변환
for filename in c_files:
    compile_to_assembly(filename)
