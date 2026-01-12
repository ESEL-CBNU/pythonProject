import pandas as pd              # 데이터 처리는 pd라는 별명으로 부름
import matplotlib.pyplot as plt  # 그래프 그리는 도구는 plt라고 부름

# 1. 가상의 실험 데이터 생성 (R의 Vector/List와 유사)
data = {
    'Time_sec': [0, 10, 20, 30, 40, 50, 60],
    'Temp_C': [20.5, 22.1, 24.3, 28.5, 30.2, 31.5, 29.8],
    'Pressure_Bar': [1.0, 1.1, 1.2, 1.5, 1.6, 1.7, 1.5]
}

# 2. 데이터 프레임(엑셀 시트 같은 표) 만들기
df = pd.read_json(pd.DataFrame(data).to_json()) # 딕셔너리를 데이터프레임으로 변환
# 보통은 df = pd.DataFrame(data) 라고 씁니다. (위 방식은 호환성을 위해 작성)
df = pd.DataFrame(data)

# 3. 기초 통계 확인 (R의 summary 함수와 비슷)
print("--- 데이터 요약 ---")
print(df.describe()) 

# 4. 그래프 그리기 (Matplotlib)
plt.figure(figsize=(10, 5))  # 도화지 크기 설정 (가로 10, 세로 5)

# X축: 시간, Y축: 온도
plt.plot(df['Time_sec'], df['Temp_C'], label='Temperature (C)', color='red', marker='o')

plt.title('Experiment Result: Time vs Temp') # 제목
plt.xlabel('Time (sec)')                     # X축 라벨
plt.ylabel('Temperature (C)')                # Y축 라벨
plt.grid(True)                               # 모눈종이 격자 표시
plt.legend()                                 # 범례 표시

plt.show() # 그래프 창 띄우기 (이게 있어야 창이 뜹니다!)