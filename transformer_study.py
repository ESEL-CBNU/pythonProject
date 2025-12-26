import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# [1] 데이터 생성 (Engineering Data)
# ---------------------------------------------------------
# 0~100 구간의 사인파 생성 + 약간의 노이즈
t = np.linspace(0, 100, 1000)
data = np.sin(t) + np.random.normal(0, 0.05, 1000)

# 시계열 데이터를 윈도우 단위로 자르는 함수
# 예: [1,2,3,4,5] -> 입력:[1,2,3,4], 정답:[5]
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 20 # 과거 20개를 보고 미래를 예측
X, y = create_sequences(data, SEQ_LENGTH)

# PyTorch가 이해하는 텐서(Tensor) 형태로 변환 (행렬 연산용)
# 형태: (배치 크기, 시퀀스 길이, 특징 개수) -> (979, 20, 1)
X_tensor = torch.from_numpy(X).float().unsqueeze(-1) 
y_tensor = torch.from_numpy(y).float().unsqueeze(-1)

# ---------------------------------------------------------
# [2] Transformer 모델 정의 (핵심 부분)
# ---------------------------------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # 1. 입력을 고차원 벡터로 변환 (1차원 -> 64차원)
        self.input_linear = nn.Linear(input_size, d_model)
        
        # 2. Transformer Encoder (기계의 두뇌)
        # d_model: 내부 처리 차원 수, nhead: 몇 개의 관점으로 볼 것인가(멀티 헤드)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 최종 출력층 (64차원 -> 1차원 예측값)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        x = self.input_linear(x) # [batch, seq_len, 64]
        x = self.transformer_encoder(x) # [batch, seq_len, 64] (여기서 Attention이 일어남)
        
        # 시퀀스의 마지막 타임스텝의 정보만 가져와서 예측
        x = x[:, -1, :] 
        x = self.output_linear(x)
        return x

# 모델 초기화
model = TimeSeriesTransformer()
criterion = nn.MSELoss() # 손실함수 (오차 계산)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 최적화 도구

# ---------------------------------------------------------
# [3] 학습 (Training)
# ---------------------------------------------------------
print(">>> 학습을 시작합니다 (총 100 Epoch)...")
losses = []

for epoch in range(100):
    optimizer.zero_grad()       # 1. 기울기 초기화
    outputs = model(X_tensor)   # 2. 모델 예측
    loss = criterion(outputs, y_tensor) # 3. 오차 계산
    loss.backward()             # 4. 역전파 (오차를 줄이는 방향 찾기)
    optimizer.step()            # 5. 가중치 수정
    
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")

# ---------------------------------------------------------
# [4] 결과 시각화
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

# 학습된 모델로 전체 예측 수행
model.eval() # 평가 모드로 전환
with torch.no_grad():
    predicted = model(X_tensor).numpy()

# 원본 데이터 그리기
plt.plot(y, label='Actual Data (Sine + Noise)', color='gray', alpha=0.5)
# 예측 데이터 그리기
plt.plot(predicted, label='Transformer Prediction', color='red', linestyle='--')

plt.title('Transformer Time Series Prediction')
plt.legend()
plt.grid(True)
plt.show()