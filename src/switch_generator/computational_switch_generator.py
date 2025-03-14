import os

def generate_computational_switch(num_cases=5):
    """ 연산을 수행하는 switch-case 구조 생성 """
    switch_code = "    switch (value) {\n"
    for i in range(1, num_cases + 1):
        switch_code += f"        case {i}: {{\n"
        switch_code += f"            int result;\n"
        if i % 5 == 1:
            switch_code += f"            result = value * 10;\n"
            switch_code += f"            printf(\"{i} 선택됨, 결과: %d\\n\", result);\n"
        elif i % 5 == 2:
            switch_code += f"            result = value + 5;\n"
            switch_code += f"            printf(\"{i} 선택됨, 결과: %d\\n\", result);\n"
        elif i % 5 == 3:
            switch_code += f"            result = value - 3;\n"
            switch_code += f"            printf(\"{i} 선택됨, 결과: %d\\n\", result);\n"
        elif i % 5 == 4:
            switch_code += f"            result = value / 2;\n"
            switch_code += f"            printf(\"{i} 선택됨, 결과: %d\\n\", result);\n"
        else:
            switch_code += f"            result = value * value;\n"
            switch_code += f"            printf(\"{i} 선택됨, 제곱 값: %d\\n\", result);\n"
        switch_code += "            break;\n        }\n"
    switch_code += "        default:\n"
    switch_code += "            printf(\"기본값 실행됨\\n\");\n"
    switch_code += "            break;\n"
    switch_code += "    }\n"
    return switch_code

# 여러 개의 computational switch 파일 생성
num_files = 10  # 생성할 파일 개수
switch_cases = 10  # 각 switch 문의 case 개수

# 저장할 폴더 생성
os.makedirs("./data/computational_switch", exist_ok=True)

for i in range(1, num_files + 1):
    for j in range(1, switch_cases + 1):
        filename = f"./data/switch/computational_switch_{i}_{j}.c"
        with open(filename, "w") as f:
            f.write("#include <stdio.h>\n\n")
            f.write("int main() {\n")
            f.write(f"    int value = {i * j};\n")
            f.write(generate_computational_switch(j))
            f.write("    return 0;\n}\n")

# 생성된 파일 목록 출력
[f"computational_switch_{i}_{j}.c" for i in range(1, num_files + 1) for j in range(1, switch_cases + 1)]
