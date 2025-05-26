import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import time
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

HEADERS = {
    "User-Agent": random.choice(USER_AGENTS)
}

load_dotenv()
COOKIES = {
    "REVEL_SESSION": os.getenv("ATCODER_COOKIE")
}

BASE_SAVE_DIR = "data/raw_code/temp/AtCoder"
right_cnt = 0

contest_403_list = []

def fetch_contests_from_archive(page=1, type=1, name="abc"):
    base_url = f"https://atcoder.jp/contests/archive?ratedType={type}&category=0&keyword="
    contests = []

    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(COOKIES)
    
    url = f"{base_url}&page={page}"
    res = session.get(url)
    print(f"📄 Contest Archive 페이지 {page}: 상태 코드 {res.status_code}")

    soup = BeautifulSoup(res.text, 'html.parser')
    tag = f"table tbody tr td a[href^='/contests/{name}']"
    links = soup.select(tag)
    for link in links:
        contest_id = link['href'].split("/")[-1]
        contest_name = link.text.strip()
        contests.append((contest_id, contest_name))

    return contests

def fetch_submissions(session, contest_id, max_pages=100):
    global contest_403_list
    submission_url = f"https://atcoder.jp/contests/{contest_id}/submissions?f.Task=&f.LanguageName=C&f.Status=AC&f.User="
    all_submissions = []

    for page in range(1, max_pages + 1):
        url = f"{submission_url}&page={page}"
        res = session.get(url)

        print(f"📄 {contest_id} - 제출 페이지 {page}: 상태 코드 {res.status_code}")
        if res.status_code != 200:
            contest_403_list.append(contest_id)
            break
        if "Sign In" in res.text:
            print("⚠️ 로그인 상태 아님. 쿠키 확인 필요.")
            break

        soup = BeautifulSoup(res.text, 'html.parser')
        rows = soup.select("table tbody tr")
        if not rows:
            print("❕ 더 이상 데이터 없음. 종료.")
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue

            user_cell = cols[2]
            user_link = user_cell.find("a", href=True)
            username = user_link['href'].split("/")[-1] if user_link else "unknown"

            detail_cell = cols[-1]
            detail_link = detail_cell.find("a", href=True)
            if detail_link and "/submissions/" in detail_link['href']:
                submission_id = detail_link['href'].split("/")[-1]
                all_submissions.append((username, submission_id))

    return all_submissions

def download_submission_code(session, contest_id, username, submission_id):
    global right_cnt
    detail_url = f"https://atcoder.jp/contests/{contest_id}/submissions/{submission_id}"
    res = session.get(detail_url)
    if res.status_code != 200:
        print(f"❌ 제출 {submission_id} 코드 요청 실패")
        return

    soup = BeautifulSoup(res.text, 'html.parser')
    code_block = soup.find("pre", id="submission-code")
    if not code_block:
        print(f"⚠️ 코드 블록 없음: 제출 {submission_id}")
        
        return

    save_dir = os.path.join(BASE_SAVE_DIR, contest_id)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{username}_{submission_id}.c"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code_block.text)
    print(f"✅ 저장됨: {contest_id}/{filename}")
    right_cnt += 1
    time.sleep(1)

if __name__ == "__main__":
    # 이 곳 수정 ABC, ARC, AGC, AHC
    # page, type, name
    contests = fetch_contests_from_archive(3, 2, "arc") 
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(COOKIES)

    for contest_id, contest_name in contests:
        save_dir = os.path.join(BASE_SAVE_DIR, contest_id)
        if os.path.exists(save_dir):
            print(f"⏩ 이미 수집된 대회입니다. 건너뜀: {contest_id}")
            continue

        print(f"\n🔍 {contest_name} ({contest_id}) 수집 시작")
        submissions = fetch_submissions(session, contest_id)

        if not submissions:
            print(f"📭 수집 가능한 제출 없음 또는 오류 발생: {contest_id}")
            continue

        for username, submission_id in submissions:
            download_submission_code(session, contest_id, username, submission_id)

    print(f"\n📦 총 저장된 코드 수: {right_cnt}")
    
    print("🚫 403 에러 발생한 대회 목록:")
    for contest_id in contest_403_list:
        print(f"- {contest_id}")

