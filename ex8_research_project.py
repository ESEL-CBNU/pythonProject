import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns          # 논문용 고품질 시각화 라이브러리
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 1년간의 날짜 데이터 생성
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
n = len(dates)

# 2. 기온(Air Temp) 데이터 생성: 여름에 덥고 겨울에 추운 사인파 형태 + 노이즈
x = np.linspace(0, 2 * np.pi, n)
air_temp = 15 + 15 * np.sin(x - np.pi / 2) + np.random.normal(0, 2, n)

# 3. 수온(Water Temp) 데이터 생성: [중요] 물의 비열 때문에 기온보다 늦게 변함 (Time Lag)
# 위상(Phase)을 조금 뒤로 밀고(-0.5), 변화 폭(Amplitude)을 줄입니다(15 -> 10).
water_temp = 12 + 10 * np.sin(x - np.pi / 2 - 0.5) + np.random.normal(0, 1, n)

# 4. 데이터 프레임 생성
df = pd.DataFrame({'Date': dates, 'Air_Temp': air_temp, 'Water_Temp': water_temp})

print("--- [1] 데이터 생성 완료 (상위 5행) ---")
print(df.head())

# 5. 시계열 그래프 확인 (수온이 기온을 뒤따라가는지 확인)
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Air_Temp'], label='Air Temp (C)', alpha=0.6, color='gray')
plt.plot(df['Date'], df['Water_Temp'], label='Water Temp (C)', color='blue', linewidth=2)
plt.title('Time Series: Air vs Water Temperature (Hysteresis Check)')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

# --- [2] Feature Engineering (변수 창출) ---

# Shift 기능을 이용해 '어제 기온(Lag_1)'과 '3일 전 기온(Lag_3)' 데이터를 만듭니다.
df['Air_Temp_Lag1'] = df['Air_Temp'].shift(1) # 하루 밀기
df['Air_Temp_Lag3'] = df['Air_Temp'].shift(3) # 3일 밀기
df['Air_Temp_Lag7'] = df['Air_Temp'].shift(7) # 7일 밀기

# shift를 하면 앞부분에 데이터가 비게 되므로(NaN), 그 행들은 삭제합니다.
df = df.dropna()

print("\n--- [2] 파생 변수(Lag) 생성 완료 ---")
print(df[['Date', 'Air_Temp', 'Air_Temp_Lag1', 'Water_Temp']].head())

# 상관계수 분석 (Correlation Matrix)
corr_matrix = df[['Water_Temp', 'Air_Temp', 'Air_Temp_Lag1', 'Air_Temp_Lag3', 'Air_Temp_Lag7']].corr()

# 히트맵 그리기 (논문 Figure 스타일)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1)
plt.title('Correlation Matrix: Which lag is most important?')
plt.show()

# --- [3] 모델 학습 및 검증 ---

# 입력 변수(X): 오늘 기온 + 지연된 기온들
X = df[['Air_Temp', 'Air_Temp_Lag1', 'Air_Temp_Lag3', 'Air_Temp_Lag7']]
# 목표 변수(y): 수온
y = df['Water_Temp']

# 학습용/테스트용 데이터 분리 (전체의 80%로 공부하고, 20%로 시험봄)
# 시계열 데이터이므로 랜덤으로 섞지 않고(shuffle=False) 순서대로 자릅니다.
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 성능 평가 (R-squared)
score = r2_score(y_test, y_pred)

print(f"\n--- [3] 모델 성능 평가 ---")
print(f"결정 계수 (R^2 Score): {score:.4f}")
print(f"회귀 식: Water = {model.intercept_:.2f} + {model.coef_[0]:.2f}*Air + {model.coef_[1]:.2f}*Lag1...")

# 최종 결과 시각화
plt.figure(figsize=(12, 6))
# 테스트 구간의 실제 값
plt.plot(df['Date'].iloc[train_size:], y_test, label='Observed (Actual)', color='black', linewidth=2)
# 테스트 구간의 예측 값
plt.plot(df['Date'].iloc[train_size:], y_pred, label='Predicted (AI Model)', color='red', linestyle='--')

plt.title(f'Water Temperature Prediction (R2 = {score:.3f})')
plt.xlabel('Date')
plt.ylabel('Water Temp (C)')
plt.legend()
plt.grid()
plt.show()