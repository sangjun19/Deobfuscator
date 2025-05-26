import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

Contest_List = [
    "abc403",
    "abc402",
    "abc401",
    "abc400"
]

CONTEST_ID = "abc398"
SUBMISSION_URL = f"https://atcoder.jp/contests/{CONTEST_ID}/submissions?f.Task=&f.LanguageName=C&f.Status=AC&f.User="

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

load_dotenv()
COOKIES = {
    "REVEL_SESSION": os.getenv("ATCODER_COOKIE")
}

right_cnt = 0

SAVE_DIR = f"data/raw_code/temp/AtCoder/{CONTEST_ID}"
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_submissions(max_pages=100):
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(COOKIES)

    all_submissions = []

    for page in range(1, max_pages + 1):
        url = f"{SUBMISSION_URL}&page={page}"
        res = session.get(url)

        print(f"📄 페이지 {page}: 상태 코드 {res.status_code}")
        if res.status_code != 200:
            break
        if "Sign In" in res.text:
            print("⚠️ 로그인 상태 아님. 쿠키 확인 필요.")
            break

        soup = BeautifulSoup(res.text, 'html.parser')
        rows = soup.select("table tbody tr")
        if not rows:
            print("✅ 더 이상 데이터 없음. 종료.")
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

def download_submission_code(session, username, submission_id):
    global right_cnt
    detail_url = f"https://atcoder.jp/contests/{CONTEST_ID}/submissions/{submission_id}"
    res = session.get(detail_url)
    if res.status_code != 200:
        print(f"❌ 제출 {submission_id} 코드 요청 실패")
        return

    soup = BeautifulSoup(res.text, 'html.parser')
    code_block = soup.find("pre", id="submission-code")
    if not code_block:
        print(f"⚠️ 코드 블록 없음: 제출 {submission_id}")
        return

    filename = f"{username}_{submission_id}.c"
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code_block.text)
    print(f"✅ 저장됨: {filename}")
    right_cnt += 1

if __name__ == "__main__":
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(COOKIES)

    submissions = fetch_submissions()

    for username, submission_id in submissions:
        download_submission_code(session, username, submission_id)
    
    print(f"📦 총 제출 수: {len(submissions)}")
    print(f"✅ 성공적으로 다운로드된 제출 수: {right_cnt}")