import os
import subprocess
import tempfile
import re

def verify_c_files(source_path):
    # 소스 폴더가 존재하는지 확인
    if not os.path.exists(source_path):
        print(f"경로 '{source_path}'가 존재하지 않습니다.")
        return []
    
    # C 파일만 가져오기
    c_files = [file for file in os.listdir(source_path) if file.endswith('.c')]
    return c_files

def compile_c_file(file_path):
    """C 파일을 컴파일하고 성공 여부를 반환합니다."""
    # 임시 출력 파일 경로 생성
    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as temp_out:
        output_path = temp_out.name
    
    try:
        # GCC로 컴파일 시도
        result = subprocess.run(
            ['gcc', '-o', output_path, file_path, '-Wall'],
            capture_output=True,
            text=True,
            timeout=30  # 컴파일 시간 제한 (30초)
        )
        
        # 컴파일 성공 여부 확인
        success = result.returncode == 0
        
        # 결과 반환
        return {
            'success': success,
            'stderr': result.stderr,
            'stdout': result.stdout
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stderr': '컴파일 시간 초과',
            'stdout': ''
        }
    except Exception as e:
        return {
            'success': False,
            'stderr': str(e),
            'stdout': ''
        }
    finally:
        # 임시 파일 삭제
        if os.path.exists(output_path):
            os.remove(output_path)

def has_stdlib_include(content):
    """content에 stdlib.h 포함 여부를 확인합니다."""
    # 정규식을 사용하여 #include <stdlib.h> 또는 #include<stdlib.h> 형태 모두 확인
    pattern = r'#\s*include\s*[<"]stdlib\.h[>"]'
    return bool(re.search(pattern, content))

def save_c_files_to_folder(file_list, source_path, destination_path, target_success=50):
    # 폴더가 없으면 생성
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"저장 경로 '{destination_path}'를 생성했습니다.")

    total_processed = 0
    compile_success = 0
    compile_fail = 0
    
    for file in file_list:
        # 컴파일 성공 파일이 목표치에 도달하면 종료
        if compile_success >= target_success:
            break

        source_file_path = os.path.join(source_path, file)
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.c', delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                with open(source_file_path, 'r', encoding='utf-8', errors='replace') as src:
                    original_content = src.read()
                
                # stdlib.h가 이미 포함되어 있는지 확인
                has_stdlib = has_stdlib_include(original_content)
                
                # 컴파일용 임시 파일 내용 준비
                if has_stdlib:
                    compile_content = original_content
                    print(f"🔍 {file}: 이미 stdlib.h가 포함되어 있습니다")
                else:
                    compile_content = f"#include <stdlib.h>\n{original_content}"
                
                # 임시 파일 작성
                with open(temp_path, 'w', encoding='utf-8') as tmp:
                    tmp.write(compile_content)
                
                # 임시 파일 컴파일 시도
                compile_result = compile_c_file(temp_path)
                
                # 컴파일 성공한 경우에만 저장
                if compile_result['success']:
                    destination_file_path = os.path.join(destination_path, file)
                    
                    # 성공한 파일을 저장할 때도 같은 로직 적용
                    with open(destination_file_path, 'w', encoding='utf-8') as dest:
                        if has_stdlib:
                            dest.write(original_content)  # 이미 헤더가 있으면 원본 그대로 저장
                        else:
                            dest.write(f"#include <stdlib.h>\n{original_content}")  # 헤더 추가
                    
                    compile_success += 1
                    print(f"✅ 컴파일 성공 ({compile_success}/{target_success}): {file}")
                else:
                    compile_fail += 1
                    print(f"❌ 컴파일 실패: {file} - {compile_result['stderr'][:100]}...")
                
                total_processed += 1
            except Exception as e:
                print(f"파일 '{file}' 처리 중 오류 발생: {e}")
                total_processed += 1
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    return {
        'total': total_processed,
        'success': compile_success,
        'fail': compile_fail,
        'target_reached': compile_success >= target_success
    }

# 경로 설정
# source_path = 'data/github_api/switch/github_switch_codes_c'
# destination_path = 'data/compile/switch_c'
source_path = 'data/github_api/non-switch/github_non_switch_codes_c'
destination_path = 'data/compile/non-switch_c'

# C 파일 가져오기
c_files = verify_c_files(source_path)

if not c_files:
    print(f"경로 '{source_path}'에서 C 파일을 찾을 수 없습니다.")
else:
    target_success = 50  # 목표로 하는 컴파일 성공 파일 수
    
    # C 파일 컴파일 및 저장
    result = save_c_files_to_folder(c_files, source_path, destination_path, target_success)
    
    print(f"\n===== 처리 결과 =====")
    print(f"처리된 C 파일: {result['total']}개")
    print(f"컴파일 성공: {result['success']}개")
    print(f"컴파일 실패: {result['fail']}개")
    
    if result['target_reached']:
        print(f"✅ 목표 달성: 컴파일 성공한 파일 {target_success}개를 모두 저장했습니다.")
    else:
        print(f"⚠️ 주의: 전체 파일을 모두 처리했지만, 컴파일 성공한 파일이 {result['success']}개로 목표({target_success}개)에 도달하지 못했습니다.")
    
    print(f"컴파일 성공한 파일들이 '{destination_path}' 폴더에 저장되었습니다.")
