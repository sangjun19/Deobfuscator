def generate_nested_switch(depth, num_cases):
    switch_code = "    switch (value) {\n"
    for i in range(1, num_cases + 1):
        switch_code += f"        case {i}:\n"
        if depth > 1:
            switch_code += f"            switch (sub_value) {{\n"
            for j in range(1, num_cases + 1):
                switch_code += f"                case {j}:\n"
                switch_code += f"                    printf(\"{i}-{j} executed\\n\");\n"
                switch_code += f"                    break;\n"
            switch_code += "            }\n"
        else:
            switch_code += f"            printf(\"Case {i} executed\\n\");\n"
        switch_code += f"            break;\n"
    switch_code += "        default:\n"
    switch_code += "            printf(\"default executed\\n\");\n"
    switch_code += "            break;\n"
    switch_code += "    }\n"
    return switch_code

# 여러 개의 switch 문을 생성하여 저장
num_files = 100  # 생성할 파일 개수
switch_depth = 100

for i in range(1, num_files+1):
    for j in range(1, switch_depth + 1):
        filename = f"./data/raw_code/switch/nested_switch/nested_switch_{i}_{j}.c"
        with open(filename, "w") as f:
            f.write("#include <stdio.h>\n\n")
            f.write("int main() {\n")
            f.write(f"    int value = {i}, sub_value = {j};\n")
            f.write(generate_nested_switch(i, j))
            f.write("    return 0;\n}\n")

# 생성된 파일 목록 출력
[f"nested_switch_{i}.c" for i in range(1, num_files + 1)]
