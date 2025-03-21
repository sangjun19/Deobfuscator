def generate_3d_switch(num_cases):
    """ 3 nested switch-case generator """
    switch_code = "    switch (value_1) {\n"
    for i in range(1, num_cases + 1):
        switch_code += f"        case {i}:\n"
        switch_code += f"            switch (value_2) {{\n"
        for j in range(1, num_cases + 1):
            switch_code += f"                case {j}:\n"
            switch_code += f"                    switch (value_3) {{\n"
            for k in range(1, num_cases + 1):
                switch_code += f"                        case {k}:\n"
                switch_code += f"                            printf(\"{i}-{j}-{k}\\n\");\n"
                switch_code += f"                            break;\n"
            switch_code += "                    }\n"
            switch_code += "                    break;\n"
        switch_code += "            }\n"
        switch_code += "            break;\n"
    switch_code += "    }\n"
    return switch_code

# 여러 개의 3중 중첩 switch문 C 코드 파일 생성
num_files = 1000  # 생성할 파일 개수 (10*10*10)
num_cases = 3    # 각 switch 문의 case 개수

for i in range(1, num_files + 1):
    # value_1, value_2, value_3 계산
    value_1 = (i - 1) // (num_cases * num_cases) + 1
    value_2 = ((i - 1) % (num_cases * num_cases)) // num_cases + 1
    value_3 = ((i - 1) % (num_cases * num_cases)) % num_cases + 1

    # C 코드 생성
    c_code = f"#include <stdio.h>\n\nint main() {{\n"
    c_code += f"    int value_1 = {value_1};\n"
    c_code += f"    int value_2 = {value_2};\n"
    c_code += f"    int value_3 = {value_3};\n"
    c_code += generate_3d_switch(num_cases)
    c_code += "    return 0;\n}\n"

    # 파일에 저장
    filename = f"./data/3d_switch/3d_switch_{i}.c"
    with open(filename, "w") as f:
        f.write(c_code)

    print(f"파일 {filename} 생성됨")

# 생성된 파일 목록 출력
[f"3d_switch_{i}.c" for i in range(1, num_files + 1)]
