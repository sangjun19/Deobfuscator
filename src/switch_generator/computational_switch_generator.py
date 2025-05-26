import os

def generate_computational_switch(num_cases=5):
    """ computational switch-case generator """
    switch_code = "    switch (value) {\n"
    for i in range(1, num_cases + 1):
        switch_code += f"        case {i}: {{\n"
        switch_code += f"            int result;\n"
        if i % 5 == 1:
            switch_code += f"            result = value * 10;\n"
            switch_code += f"            printf(\"{i} selected, result: %d\\n\", result);\n"
        elif i % 5 == 2:
            switch_code += f"            result = value + 5;\n"
            switch_code += f"            printf(\"{i} selected, result: %d\\n\", result);\n"
        elif i % 5 == 3:
            switch_code += f"            result = value - 3;\n"
            switch_code += f"            printf(\"{i} selected, result: %d\\n\", result);\n"
        elif i % 5 == 4:
            switch_code += f"            result = value / 2;\n"
            switch_code += f"            printf(\"{i} selected, result: %d\\n\", result);\n"
        else:
            switch_code += f"            result = value * value;\n"
            switch_code += f"            printf(\"{i} selected, square: %d\\n\", result);\n"
        switch_code += "            break;\n        }\n"
    switch_code += "        default:\n"
    switch_code += "            printf(\"default\\n\");\n"
    switch_code += "            break;\n"
    switch_code += "    }\n"
    return switch_code

# 여러 개의 computational switch 파일 생성
num_files = 100  # 생성할 파일 개수
switch_cases = 100  # 각 switch 문의 case 개수

for i in range(1, num_files + 1):
    for j in range(1, switch_cases + 1):
        filename = f"./data/raw_code/switch/computational_switch/computational_switch_{i}_{j}.c"
        with open(filename, "w") as f:
            f.write("#include <stdio.h>\n\n")
            f.write("int main() {\n")
            f.write(f"    int value = {i * j};\n")
            f.write(generate_computational_switch(j))
            f.write("    return 0;\n}\n")

# 생성된 파일 목록 출력
[f"computational_switch_{i}_{j}.c" for i in range(1, num_files + 1) for j in range(1, switch_cases + 1)]
