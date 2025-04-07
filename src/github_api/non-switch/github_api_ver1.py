import requests
import base64
import os
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

load_dotenv()
api_url = "https://api.github.com/search/code"
github_token = os.getenv('GITHUB_TOKEN')
LANGUAGE = "c"

# GitHub 코드 검색 API에서 사용 가능한 필터들을 활용한 하위 쿼리
SUB_QUERIES = [
    # 다양한 파일 확장자로 분할
    "+extension:c",
    
    # 파일 이름에 다양한 패턴 포함
    "+filename:main",
    "+filename:util",
    "+filename:core",
    "+filename:lib",
    
    # 일반적인 C 프로그래밍 패턴 (switch와 관련 없는)
    "+if+else",
    "+for+loop",
    "+while+loop",
    "+struct",
    "+malloc",
    "+typedef",
    "+enum",
    
    # 인기 있는 저장소와 그렇지 않은 저장소 분리
    "+repo:>10000",
    "+repo:<1000"
]

# 기본 쿼리 - C 언어 파일만 검색
base_query = f"language:{LANGUAGE}"

headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/vnd.github.v3+json"
}

params = {
    "q": base_query,
    "per_page": 100
}

def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def search_github_code():
    results = []
    total_count = 0
    session = requests_retry_session()
    
    # 각 하위 쿼리에 대해 검색 수행
    for subquery_idx, filter_param in enumerate(SUB_QUERIES):
        print(f"처리 중인 하위 쿼리 {subquery_idx + 1}/{len(SUB_QUERIES)}: {filter_param}")
        
        # 현재 하위 쿼리에 필터 추가
        params["q"] = base_query + " " + filter_param.replace("+", " ")
        
        page = 1
        subquery_count = 0
        
        # 현재 하위 쿼리의 총 페이지 수 계산
        try:
            initial_response = session.get(api_url, headers=headers, params=params)
            initial_response.raise_for_status()
            initial_data = initial_response.json()
            total_results = initial_data.get("total_count", 0)
            max_pages = min(10, (total_results + 99) // 100)  # 최대 10페이지(1000개)까지만 처리
            print(f"현재 하위 쿼리의 총 결과: {total_results}, 처리할 페이지 수: {max_pages}")
            
            # 검색 결과가 없으면 다음 쿼리로 넘어감
            if total_results == 0:
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"초기 쿼리 오류: {e}")
            continue
        
        # 현재 하위 쿼리의 모든 페이지 처리
        while page <= max_pages:
            params["page"] = page
            try:
                print(f"페이지 {page}/{max_pages} 처리 중...")
                response = session.get(api_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "items" not in data or not data["items"]:
                    print("더 이상 항목이 없습니다.")
                    break
                
                for item in data["items"]:
                    file_url = item["url"]
                    try:
                        file_content = get_file_content(session, file_url)
                        # switch가 포함되지 않은 파일만 저장
                        if file_content and "switch" not in file_content:
                            result = {
                                "repo": item["repository"]["full_name"],
                                "file": item["path"],
                                "code": file_content
                            }
                            results.append(result)
                            save_to_file(result)
                            subquery_count += 1
                            total_count += 1
                            print(f"{total_count} : Repository: {result['repo']}")
                    except Exception as e:
                        print(f"파일 내용 처리 오류: {e}")
                        continue
                
                page += 1
                
                # API 속도 제한에 걸리지 않도록 잠시 대기
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                print(f"페이지 {page} 처리 중 오류 발생: {e}")
                if "rate limit" in str(e).lower():
                    print("API 속도 제한에 도달했습니다. 1분 대기 후 재시도합니다...")
                    time.sleep(60)
                    continue
                break
        
        print(f"하위 쿼리 {subquery_idx + 1} 완료: {subquery_count}개 파일 처리됨")
        
        # API 속도 제한 방지를 위한 대기
        if subquery_idx < len(SUB_QUERIES) - 1:
            wait_time = 10  # 10초 대기
            print(f"다음 하위 쿼리 전 {wait_time}초 대기 중...")
            time.sleep(wait_time)
    
    return results

def get_file_content(session, file_url):
    response = session.get(file_url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if "content" in data:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return None

def save_to_file(result):
    directory = f"./data/github_api/non-switch/github_non_switch_codes_{LANGUAGE}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    repo_name = result["repo"].replace("/", "_")
    file_name = os.path.basename(result["file"])
    save_name = f"{repo_name}_{file_name}"

    with open(os.path.join(directory, save_name), "w", encoding="utf-8", errors="replace") as f:
        f.write(result['code'])

if __name__ == "__main__":
    results = search_github_code()
    print(f"총 {len(results)}개 파일이 'github_non_switch_codes_{LANGUAGE}' 디렉토리에 저장되었습니다.")
