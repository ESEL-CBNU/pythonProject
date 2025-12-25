import requests
from bs4 import BeautifulSoup

# 1. 데이터를 가져올 웹사이트 주소 (Target URL)
url = "https://finance.naver.com/marketindex/"

# 2. 웹사이트에 접속 요청 (브라우저인 척 하기)
# 어떤 사이트는 봇을 차단하므로 '나는 봇이 아닙니다'라는 명찰(User-Agent)을 답니다.
headers = {'User-Agent': 'Mozilla/5.0'} 
response = requests.get(url, headers=headers)

# 접속 성공 여부 확인 (200이면 성공, 404나 500이면 실패)
if response.status_code == 200:
    print(">>> 접속 성공! 데이터를 분석합니다.")
    
    # 3. HTML 코드를 파이썬이 이해할 수 있게 변환 (Soup 객체 생성)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 4. 원하는 데이터 찾기 (CSS Selector)
    # 크롬 개발자 도구(F12)에서 Copy Selector로 가져온 주소입니다.
    # 해석: id가 exchangeList인 곳 밑에 -> class가 on인 li 밑에 -> ... -> class가 value인 span
    exchange_rate = soup.select_one('#exchangeList > li.on > a.head.usd > div > span.value')

    # 5. 결과 출력
    print(f"현재 미국 USD 환율: {exchange_rate.text} 원")

else:
    print(f"접속 실패. 에러 코드: {response.status_code}")