import requests
import base64
import os
import random
import time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
api_url = "https://api.github.com/search/code"
github_token = os.getenv('GITHUB_TOKEN')
LANGUAGE = "c++"

# GitHub API 속도 제한 추적을 위한 글로벌 변수
rate_limit_remaining = 60  # 초기값: 가정(최대 분당 60개 요청)
rate_limit_reset = 0       # 다음 재설정 시간(Unix 타임스탬프)

# 파일 크기나 다른 기준으로 하위 쿼리 분할
SUB_QUERIES = [
    "+extension:c++",
    "+filename:main",
    "+filename:util",
    "+filename:core",
    "+case",
    "+default",
    "+break",
    "+path:src/",
    "+path:include/",
    "+path:lib/",
    "+created:2015-01-01..2015-12-31",
    "+created:2016-01-01..2016-12-31",
    "+created:2017-01-01..2017-12-31",
    "+created:2018-01-01..2018-12-31",
    "+created:2019-01-01..2019-12-31",
    "+created:2020-01-01..2020-12-31",
    "+created:2021-01-01..2021-12-31",
    "+created:2022-01-01..2022-12-31",
    "+created:2023-01-01..2023-12-31",
    "+created:2024-01-01..2024-12-31",
    "+created:>2025-01-01",
    "+size:<1000"
    "+size:>=1000..<2000",
    "+size:>=2000..<3000",
    "+size:>=3000..<4000",
    "+size:>=4000..<5000",
    "+stars:>100",
    "+stars:<=100",
]

# 기본 쿼리
base_query = f"switch language:{LANGUAGE}"

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

def check_rate_limit(response):
    """GitHub API 응답 헤더에서 속도 제한 정보를 추출합니다"""
    global rate_limit_remaining, rate_limit_reset
    
    # X-RateLimit-Remaining: 남은 요청 수
    if 'X-RateLimit-Remaining' in response.headers:
        rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
    
    # X-RateLimit-Reset: 속도 제한 재설정 시간(Unix 타임스탬프)
    if 'X-RateLimit-Reset' in response.headers:
        rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
    
    print(f"남은 API 요청 수: {rate_limit_remaining}, 재설정 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rate_limit_reset))}")
    
    # 남은 요청 수가 10 이하면 경고
    if rate_limit_remaining <= 10:
        print(f"⚠️ 경고: API 요청 제한에 근접했습니다! 남은 요청 수: {rate_limit_remaining}")

def wait_if_rate_limited():
    """속도 제한에 근접하면 대기합니다"""
    global rate_limit_remaining, rate_limit_reset
    
    # 남은 요청 수가 5 이하면 재설정 시간까지 대기
    if rate_limit_remaining <= 5:
        current_time = time.time()
        wait_time = max(0, rate_limit_reset - current_time) + 5  # 5초 추가 여유
        
        if wait_time > 0:
            print(f"🕒 속도 제한에 도달했습니다. {wait_time:.1f}초 대기 중...")
            time.sleep(wait_time)
            # 대기 후 속도 제한 재설정
            rate_limit_remaining = 60
            print("속도 제한이 재설정되었습니다. 계속 진행합니다.")

def exponential_backoff(attempt, base_delay=2, max_delay=120, jitter=True):
    """지수 백오프 대기 시간을 계산합니다"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    # 지터 추가 (무작위성으로 서버 부하 분산)
    if jitter:
        delay *= random.uniform(0.8, 1.2)
    
    return delay

def search_github_code():
    results = []
    total_count = 0
    session = requests_retry_session()
    
    # 각 하위 쿼리에 대해 검색 수행
    for subquery_idx, filter_param in enumerate(SUB_QUERIES):
        print(f"처리 중인 하위 쿼리 {subquery_idx + 1}/{len(SUB_QUERIES)}: {filter_param}")
        
        # 속도 제한 확인 및 필요 시 대기
        wait_if_rate_limited()
        
        # 현재 하위 쿼리에 필터 추가
        params["q"] = base_query + " " + filter_param.replace("+", " ")
        
        page = 1
        subquery_count = 0
        
        # 현재 하위 쿼리의 총 페이지 수 계산 (최대 3번 재시도)
        for attempt in range(3):
            try:
                initial_response = session.get(api_url, headers=headers, params=params)
                initial_response.raise_for_status()
                check_rate_limit(initial_response)  # 속도 제한 확인
                
                initial_data = initial_response.json()
                total_results = initial_data.get("total_count", 0)
                max_pages = min(10, (total_results + 99) // 100)  # 최대 10페이지(1000개)까지만 처리
                print(f"현재 하위 쿼리의 총 결과: {total_results}, 처리할 페이지 수: {max_pages}")
                
                # 검색 결과가 없으면 다음 쿼리로 넘어감
                if total_results == 0:
                    break
                
                # 첫 페이지에 있는 항목 처리
                if "items" in initial_data and initial_data["items"]:
                    process_items(session, initial_data["items"], results, subquery_count, total_count)
                    subquery_count = len(results) - total_count
                    total_count = len(results)
                
                # 페이지 처리 반복문 시작
                page = 2  # 첫 페이지는 이미 처리했으므로 2부터 시작
                
                break  # 성공하면 재시도 중단
                
            except requests.exceptions.RequestException as e:
                wait_time = exponential_backoff(attempt)
                
                if attempt < 2:  # 마지막 시도가 아닐 경우
                    print(f"초기 쿼리 오류: {e} - {wait_time:.1f}초 후 재시도 ({attempt+1}/3)...")
                    time.sleep(wait_time)
                else:
                    print(f"초기 쿼리 오류: {e} - 최대 재시도 횟수 초과. 다음 쿼리로 넘어갑니다.")
                    break
        
        # 2페이지부터 max_pages까지 처리
        while page <= max_pages:
            # 속도 제한 확인 및 필요 시 대기
            wait_if_rate_limited()
            
            params["page"] = page
            retry_count = 0
            max_retries = 5
            
            while retry_count < max_retries:
                try:
                    print(f"페이지 {page}/{max_pages} 처리 중...")
                    response = session.get(api_url, headers=headers, params=params)
                    response.raise_for_status()
                    check_rate_limit(response)  # 속도 제한 확인
                    
                    data = response.json()
                    
                    if "items" not in data or not data["items"]:
                        print("더 이상 항목이 없습니다.")
                        break
                    
                    # 항목 처리
                    new_items = process_items(session, data["items"], results, subquery_count, total_count)
                    subquery_count += new_items
                    total_count += new_items
                    
                    page += 1
                    retry_count = 0  # 성공하면 재시도 카운터 리셋
                    
                    # 각 페이지 요청 사이에 대기 (속도 제한 예방)
                    time.sleep(3)  # 2초에서 3초로 증가
                    break  # 성공했으므로 재시도 루프 종료
                    
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    wait_time = exponential_backoff(retry_count, base_delay=5)
                    
                    if "rate limit" in str(e).lower() or response.status_code == 403:
                        print(f"API 속도 제한에 도달했습니다. {wait_time:.1f}초 대기 후 재시도 ({retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                    elif retry_count < max_retries:
                        print(f"페이지 {page} 처리 중 오류 발생: {e} - {wait_time:.1f}초 후 재시도 ({retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"페이지 {page} 처리 중 최대 재시도 횟수 초과: {e}")
                        break
        
        print(f"하위 쿼리 {subquery_idx + 1} 완료: {subquery_count}개 파일 처리됨")
        
        # API 속도 제한 방지를 위한 대기
        if subquery_idx < len(SUB_QUERIES) - 1:
            wait_time = 15  # 10초에서 15초로 증가
            print(f"다음 하위 쿼리 전 {wait_time}초 대기 중...")
            time.sleep(wait_time)
    
    return results

def process_items(session, items, results, subquery_count, total_count):
    """항목 목록을 처리하고 새로 추가된 항목 수를 반환합니다"""
    added_count = 0
    
    for item in items:
        file_url = item["url"]
        
        # 파일 내용 가져오기 (최대 3번 재시도)
        for attempt in range(3):
            try:
                # 속도 제한 확인 및 필요 시 대기
                wait_if_rate_limited()
                
                file_content = get_file_content(session, file_url)
                if file_content and "switch" in file_content:
                    result = {
                        "repo": item["repository"]["full_name"],
                        "file": item["path"],
                        "code": file_content
                    }
                    results.append(result)
                    save_to_file(result)
                    added_count += 1
                    print(f"{total_count + added_count} : Repository: {result['repo']}")
                
                # 각 파일 요청 사이에 짧은 대기
                time.sleep(1)
                break  # 성공했으므로 재시도 루프 종료
                
            except Exception as e:
                wait_time = exponential_backoff(attempt)
                
                if "403" in str(e) and attempt < 2:  # 403 에러인 경우 더 길게 대기
                    print(f"파일 내용 처리 중 권한 오류: {e} - {wait_time:.1f}초 후 재시도 ({attempt+1}/3)...")
                    time.sleep(wait_time * 2)  # 403 에러는 두 배 길게 대기
                elif attempt < 2:  # 마지막 시도가 아닐 경우
                    print(f"파일 내용 처리 오류: {e} - {wait_time:.1f}초 후 재시도 ({attempt+1}/3)...")
                    time.sleep(wait_time)
                else:
                    print(f"파일 내용 처리 오류: {e} - 최대 재시도 횟수 초과. 다음 항목으로 넘어갑니다.")
    
    return added_count

def get_file_content(session, file_url):
    """파일 내용을 가져오고 속도 제한 정보를 확인합니다"""
    response = session.get(file_url, headers=headers)
    response.raise_for_status()
    
    # 속도 제한 확인
    check_rate_limit(response)
    
    data = response.json()
    if "content" in data:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return None

def save_to_file(result):
    directory = f"./data/github_api/github_switch_codes_{LANGUAGE}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    repo_name = result["repo"].replace("/", "_")
    file_name = os.path.basename(result["file"])
    save_name = f"{repo_name}_{file_name}"

    with open(os.path.join(directory, save_name), "w", encoding="utf-8", errors="replace") as f:
        f.write(result['code'])

if __name__ == "__main__":
    results = search_github_code()
    print(f"총 {len(results)}개 파일이 'github_switch_codes_{LANGUAGE}' 디렉토리에 저장되었습니다.")
