import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 엉망인 데이터 생성 (중간중간 값이 비어있음)
data = {
    'Date_Str': ['2024-01-01 09:00', '2024-01-01 10:00', '2024-01-01 11:00', 
                 '2024-01-01 12:00', '2024-01-01 13:00', '2024-01-02 09:00', 
                 '2024-01-02 10:00', '2024-01-02 11:00'],
    'Sensor_A': [20.5, np.nan, 22.1, 21.8, np.nan, 25.4, 26.1, 25.8], # np.nan = 결측치
    'Location': ['Zone1', 'Zone1', 'Zone1', 'Zone1', 'Zone1', 'Zone2', 'Zone2', 'Zone2']
}

df = pd.DataFrame(data)

print("--- [1] 원본 데이터 (NaN 확인) ---")
print(df)

# 2. 날짜 변환 (문자열 -> 시간 객체)
# R의 as.Date와 유사하지만 훨씬 똑똑합니다. 포맷을 알아서 유추합니다.
df['DateTime'] = pd.to_datetime(df['Date_Str'])

# 3. 결측치(NaN) 채우기 (Engineering Tip!)
# 단순 삭제(dropna)보다는 직전 값으로 채우는(ffill) 것이 시계열 센서 데이터에 유리합니다.
df['Sensor_A_Filled'] = df['Sensor_A'].ffill() 

print("\n--- [2] 결측치 보정 완료 ---")
print(df[['DateTime', 'Sensor_A', 'Sensor_A_Filled']])

# 4. 데이터 집계 (Pivot / GroupBy)
# 날짜별(Date)로 묶어서 평균 구하기
# dt.date는 시간(시:분:초)을 떼고 날짜(년-월-일)만 추출하는 기능입니다.
daily_avg = df.groupby(df['DateTime'].dt.date)['Sensor_A_Filled'].mean()

print("\n--- [3] 일별 평균 온도 ---")
print(daily_avg)

# 5. 시각화 (보정 전 vs 보정 후)
plt.figure(figsize=(10, 6))
plt.plot(df['DateTime'], df['Sensor_A'], 'o', label='Original (Missing)', color='red')
plt.plot(df['DateTime'], df['Sensor_A_Filled'], '--', label='Filled (Interpolated)', color='blue')
plt.title('Sensor Data Cleaning & Interpolation')
plt.xticks(rotation=45) # X축 날짜 글씨가 겹치지 않게 45도 회전
plt.legend()
plt.grid(True)
plt.tight_layout() # 그래프 여백 자동 조정
plt.show()