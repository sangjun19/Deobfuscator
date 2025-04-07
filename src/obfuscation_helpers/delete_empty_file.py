import os

def delete_empty_c_files(folder_path):
    deleted = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.c'):
            filepath = os.path.join(folder_path, filename)
            if os.path.getsize(filepath) == 0:
                os.remove(filepath)
                print(f"{filename} → 삭제됨 (빈 파일)")
                deleted += 1
    if deleted == 0:
        print("삭제할 빈 .c 파일이 없습니다.")

# 사용 예시
delete_empty_c_files('./')  # parameter = path of directory