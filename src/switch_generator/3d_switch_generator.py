def generate_3d_switch(x, y, z):
    """ 3 nested switch-case generator """
    switch_code = "    switch (value_1) {\n"
    for i in range(1, x + 1):
        switch_code += f"        case {i}:\n"
        switch_code += f"            switch (value_2) {{\n"
        for j in range(1, y + 1):
            switch_code += f"                case {j}:\n"
            switch_code += f"                    switch (value_3) {{\n"
            for k in range(1, z + 1):
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
num_files = 10  # 생성할 파일 개수 (10*10*10)
num_cnt = 0;

for i in range(1, num_files + 1):
    for j in range(1, num_files + 1):
        for k in range(1, num_files + 1):
            num_cnt += 1

            # C 코드 생성
            c_code = f"#include <stdio.h>\n\nint main() {{\n"
            c_code += f"    int value_1 = {i};\n"
            c_code += f"    int value_2 = {j};\n"
            c_code += f"    int value_3 = {k};\n"
            c_code += generate_3d_switch(i, j, k)
            c_code += "    return 0;\n}\n"

            # 파일에 저장
            filename = f"./data/raw_code/switch/3d_switch/3d_switch_{num_cnt}.c"
            with open(filename, "w") as f:
                f.write(c_code)

            print(f"파일 {filename} 생성됨")
    

# 생성된 파일 목록 출력
[f"3d_switch_{i}.c" for i in range(1, num_files + 1)]
