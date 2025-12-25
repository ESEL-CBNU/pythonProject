import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # 수학 계산용 라이브러리(난수 생성 등)

# --- [Step 1] 가상의 센서 데이터 파일 만들기 (원래는 이미 있는 파일을 씁니다) ---
# 100개의 시간 데이터 생성
time = np.arange(0, 100, 1) 
# 랜덤한 노이즈가 섞인 사인파 온도 데이터 생성
temp = 20 + 5 * np.sin(time / 10) + np.random.normal(0, 0.5, 100)

# 데이터 프레임으로 만들기
raw_data = pd.DataFrame({
    'Time': time,
    'Temperature': temp
})

# csv 파일로 저장하기 (현재 폴더에 'sensor_log.csv'가 생깁니다)
raw_data.to_csv('sensor_log.csv', index=False)
print(">>> 'sensor_log.csv' 파일이 생성되었습니다.")


# --- [Step 2] 저장된 파일을 불러와서 분석하기 ---
print(">>> 파일을 읽어옵니다...")
df = pd.read_csv('sensor_log.csv')

# --- [Step 3] 데이터 필터링 (R의 subset과 유사) ---
# 조건: 온도가 24도 이상인 위험 구간만 찾기
# 문법해석: df[ 조건식 ] -> 조건식이 True인 행만 뽑아라
high_temp_df = df[df['Temperature'] > 24]

print(f"총 데이터 개수: {len(df)}개")
print(f"24도 이상 데이터 개수: {len(high_temp_df)}개")

# --- [Step 4] 시각화 (원본 vs 위험구간) ---
plt.figure(figsize=(12, 6))

# 전체 데이터 (파란 실선)
plt.plot(df['Time'], df['Temperature'], label='Normal Data', color='blue', alpha=0.5)

# 위험 구간 (빨간 점) - 필터링된 데이터만 찍기
plt.scatter(high_temp_df['Time'], high_temp_df['Temperature'], color='red', label='High Temp (>24)', zorder=5)

# 기준선 그리기 (24도)
plt.axhline(y=24, color='green', linestyle='--', label='Threshold (24C)')

plt.title('Sensor Data Analysis: High Temperature Detection')
plt.legend()
plt.grid(True)
plt.show()