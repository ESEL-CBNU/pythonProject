import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression # 선형회귀 모델

# 1. 학습 데이터 생성 (공학 실험 데이터 가정)
# y = 3x + 10 이라는 물리 법칙을 따른다고 가정하고 노이즈를 섞습니다.
x = np.arange(0, 50).reshape(-1, 1)  # [중요] 입력값은 2차원 행렬이어야 함 (50행 1열)
noise = np.random.normal(0, 10, 50).reshape(-1, 1) # 잡음
y = 3 * x + 10 + noise # 실제 관측값

# 2. 모델 생성 및 학습 (Model Fitting)
# R의 lm(y ~ x) 와 같습니다.
model = LinearRegression()
model.fit(x, y) # 기계에게 "x일 때 y가 나온다"고 학습시킴

# 3. 학습 결과 확인 (기울기와 절편)
slope = model.coef_[0][0]      # 기울기 (Weight)
intercept = model.intercept_[0] # 절편 (Bias)
print(f"--- 인공지능이 찾은 공식 ---")
print(f"예측 공식: y = {slope:.2f}x + {intercept:.2f}")
print(f"실제 공식: y = 3.00x + 10.00 (비슷한가요?)")

# 4. 미래 예측 (Prediction)
# x가 50부터 60까지일 때 값 예측해보기
x_future = np.arange(50, 60).reshape(-1, 1)
y_predict = model.predict(x_future)

# 5. 시각화 (과거 데이터 + 회귀선 + 미래 예측)
plt.figure(figsize=(10, 6))

# 과거 데이터 (파란 점)
plt.scatter(x, y, label='Observation Data', color='blue', alpha=0.5)

# 학습한 회귀선 (빨간 선) - 현재 구간
plt.plot(x, model.predict(x), label='Regression Line', color='red', linewidth=2)

# 미래 예측 (초록색 별)
plt.scatter(x_future, y_predict, label='Future Prediction', color='green', marker='*', s=150)

plt.title('Linear Regression Example')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.grid(True)
plt.show()