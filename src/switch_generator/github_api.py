import requests
import base64
import os
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
api_url = "https://api.github.com/search/code"
github_token = os.getenv('GITHUB_TOKEN')  # 실제 토큰으로 교체하세요
LANGUAGE = "c"  # 여기서 언어를 변경할 수 있습니다
query = f"switch language:{LANGUAGE}"

headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/vnd.github.v3+json"
}

params = {
    "q": query,
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
    page = 1
    cnt = 0

    while True:
        params["page"] = page
        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            if "items" not in data:
                break

            for item in data["items"]:
                cnt += 1
                file_url = item["url"]
                file_content = get_file_content(file_url)
                if file_content and "switch" in file_content:
                    result = {
                        "repo": item["repository"]["full_name"],
                        "file": item["path"],
                        "code": file_content
                    }
                    results.append(result)
                    save_to_file(result)
                    print(f"{cnt} : Repository: {result['repo']}")

            # if len(data["items"]) < 100:
            #     break

            page += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            if page == 100:
                break
            page += 1
            continue
        
        if page == 100: break

    return results

def get_file_content(file_url):
    response = requests.get(file_url, headers=headers)
    data = response.json()
    if "content" in data:
        return base64.b64decode(data["content"]).decode("utf-8")
    return None

def save_to_file(result):
    directory = f"./data/github_switch_codes_{LANGUAGE}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    repo_name = result["repo"].replace("/", "_")
    file_name = os.path.basename(result["file"])
    save_name = f"{repo_name}_{file_name}"
    
    # counter = 1
    # while os.path.exists(os.path.join(directory, save_name)):
    #     save_name = f"{repo_name}_{file_name}_{counter}"
    #     counter += 1

    with open(os.path.join(directory, save_name), "w") as f:
        f.write(f"// Repository: {result['repo']}\n")
        f.write(f"// File: {result['file']}\n\n")
        f.write(result['code'])

if __name__ == "__main__":
    results = search_github_code()
    # for result in results:
    #     print(f"Repository: {result['repo']}")
    #     print(f"File: {result['file']}")
    #     print("Code saved to file")
    #     print("-" * 50)
    
    print(f"Total {len(results)} files saved in 'github_switch_codes_{LANGUAGE}' directory.")
