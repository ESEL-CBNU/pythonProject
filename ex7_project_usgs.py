import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # X축 날짜 예쁘게 꾸미기용

print(">>> USGS 데이터 가져오기 프로젝트를 시작합니다...")

# --- [1. 설정] ---
site_id = '09380000' # 콜로라도 강 Lees Ferry 지점
param_cd = '00065'   # 수위 (Gage Height, feet)
period = 'P30D'      # 최근 30일

# USGS 순간 데이터(Instantaneous Values Service) API URL 조합
url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site_id}&period={period}&parameterCd={param_cd}"
print(f"요청 URL: {url}")


# --- [2. 데이터 요청 (Web Scraping 기술 응용)] ---
try:
    response = requests.get(url)
    response.raise_for_status() # 에러가 있으면 즉시 중단
    data_json = response.json() # 받아온 데이터를 파이썬 딕셔너리로 변환
    print(">>> 데이터 서버 접속 성공! 파싱을 시작합니다.")

except requests.exceptions.RequestException as e:
    print(f"에러 발생: {e}")
    exit()


# --- [3. 데이터 파싱 및 정리 (Pandas 기술 응용)] ---
# USGS JSON 구조는 매우 깊고 복잡합니다. 알맹이만 쏙 빼내야 합니다.
# 구조: value -> timeSeries[0] -> values[0] -> value 리스트 안에 실제 데이터가 있습니다.
try:
    # 3-1. 복잡한 구조 안으로 파고 들어가기
    raw_data_list = data_json['value']['timeSeries'][0]['values'][0]['value']
    
    # 3-2. 빈 리스트를 만들고 필요한 정보(시간, 값)만 반복문으로 담습니다.
    dates = []
    levels = []
    for item in raw_data_list:
        dates.append(item['dateTime'])
        levels.append(item['value'])
    
    # 3-3. Pandas 데이터프레임으로 만들기
    df = pd.DataFrame({
        'DateTime_Str': dates,
        'WaterLevel_ft': levels
    })

    # 3-4. 데이터 타입 변환 (가장 중요! 문자열 -> 시간/숫자)
    # USGS 시간 포맷은 '2024-05-20T10:15:00.000-07:00' 처럼 복잡해서 'utc=True' 옵션을 줍니다.
    df['DateTime'] = pd.to_datetime(df['DateTime_Str'], utc=True) 
    df['WaterLevel_ft'] = pd.to_numeric(df['WaterLevel_ft']) # 숫자로 변환

    print("--- 데이터 정리 완료 (상위 5개) ---")
    print(df[['DateTime', 'WaterLevel_ft']].head())

except (KeyError, IndexError) as e:
    print("JSON 데이터 구조가 예상과 다릅니다. API 응답을 확인해주세요.")
    exit()


# --- [4. 시각화 (Matplotlib 기술 응용)] ---
print(">>> 그래프를 그립니다...")
plt.figure(figsize=(12, 6))

# 메인 데이터 플롯
plt.plot(df['DateTime'], df['WaterLevel_ft'], color='blue', linewidth=1.5, label='Gage Height (ft)')

# 그래프 꾸미기
site_name = data_json['value']['timeSeries'][0]['sourceInfo']['siteName'] # JSON에서 지점명 가져오기
plt.title(f'USGS 30-Day Water Level: {site_name} ({site_id})', fontsize=14)
plt.ylabel('Gage Height (feet)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# X축 날짜 포맷 예쁘게 다듬기 (이 부분은 심화 내용이지만 결과물이 좋아집니다)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate() # 날짜가 겹치지 않게 비스듬히 기울이기

plt.tight_layout()
plt.show()
print(">>> 이제 완료!")