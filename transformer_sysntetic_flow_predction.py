import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------
# [1] 설정 (서버 연결 없이 로컬에서 생성)
# -----------------------------------------------------------
SITE_ID = 'Mock_Colorado_River'
INPUT_WINDOW = 30     # 과거 30일
OUTPUT_WINDOW = 3     # 미래 3일
BATCH_SIZE = 32
EPOCHS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- [대체 함수] USGS 서버 대신 가상 데이터 생성 ---
def get_mock_daily_data(period_years=10):
    print(f">>> USGS 서버 점검으로 인해 '가상 유량 데이터'를 생성합니다. (기간: {period_years}년)")
    
    # 1. 날짜 생성
    dates = pd.date_range(start='2014-01-01', periods=period_years*365, freq='D')
    t = np.linspace(0, period_years * 2 * np.pi, len(dates))
    
    # 2. 유량 데이터 생성 (수문학적 특성 모방)
    # 기본 유량 15,000 cfs + 계절 변동(여름 홍수) 5,000 cfs + 랜덤 노이즈
    seasonal = 15000 + 5000 * np.sin(t - np.pi/2) 
    noise = np.random.normal(0, 1000, len(dates)) # 불규칙한 변동
    trend = np.linspace(0, 2000, len(dates))      # 10년간 약간의 유량 증가 추세
    
    values = seasonal + noise + trend
    
    # 음수 유량 방지 (물리적으로 불가능하므로)
    values = np.maximum(values, 100)
    
    df = pd.DataFrame({'Date': dates, 'Value': values})
    df = df.set_index('Date')
    
    print(f">>> 데이터 생성 완료! 총 {len(df)}일 데이터")
    return df

# -----------------------------------------------------------
# [2] 데이터 전처리
# -----------------------------------------------------------
def create_sequences(data, input_len, output_len):
    xs, ys = [], []
    for i in range(len(data) - input_len - output_len + 1):
        x = data[i : i+input_len]
        y = data[i+input_len : i+input_len+output_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# [변경] 가상 데이터 로드
df = get_mock_daily_data(period_years=10)
raw_data = df['Value'].values.reshape(-1, 1)

# 정규화 (MinMax Scaling)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(raw_data)

# 학습 데이터셋 생성
X, y = create_sequences(scaled_data, INPUT_WINDOW, OUTPUT_WINDOW)

train_size = int(len(X) * 0.8)
X_train = torch.FloatTensor(X[:train_size]).to(device)
y_train = torch.FloatTensor(y[:train_size]).squeeze(-1).to(device)
X_test = torch.FloatTensor(X[train_size:]).to(device)
y_test = torch.FloatTensor(y[train_size:]).squeeze(-1).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------------------
# [3] Transformer 모델 설계
# -----------------------------------------------------------
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, output_dim=3):
        super(TransformerPredictor, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        # batch_first=True가 중요합니다 (입력 차원 순서 관련)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim) 

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer(x)
        last_hidden = x[:, -1, :] # 마지막 시점의 특징만 사용
        output = self.fc_out(last_hidden)
        return output

model = TransformerPredictor(output_dim=OUTPUT_WINDOW).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# -----------------------------------------------------------
# [4] 모델 학습
# -----------------------------------------------------------
print(f">>> Transformer 학습 시작 ({device} 모드)...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

# -----------------------------------------------------------
# [5] 예측 및 시각화
# -----------------------------------------------------------
print(">>> 향후 3일 유량 예측 수행 중...")
model.eval()

# 가장 최근 30일 데이터 가져오기
last_30_days = scaled_data[-INPUT_WINDOW:]
last_30_days_tensor = torch.FloatTensor(last_30_days).unsqueeze(0).to(device)

with torch.no_grad():
    predicted_scaled = model(last_30_days_tensor).cpu().numpy()

# 스케일 역변환 (cfs 단위 복구)
predicted_cfs = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=3)

print("\n================ [예측 결과] ================")
print(f"기준일: {last_date.date()} (유량: {df['Value'].iloc[-1]:.2f} cfs)")
for date, val in zip(future_dates, predicted_cfs):
    print(f"예측일: {date.date()} -> 예측 유량: {val[0]:.2f} cfs")
print("=============================================")

# 그래프
plt.figure(figsize=(10, 6))
recent_df = df.iloc[-365:] # 최근 1년치만 표시
plt.plot(recent_df.index, recent_df['Value'], label='History (Mock Data)', color='blue', alpha=0.6)

plot_dates = [last_date] + list(future_dates)
plot_values = [df['Value'].iloc[-1]] + list(predicted_cfs.flatten())

plt.plot(plot_dates, plot_values, label='Transformer Forecast', 
         color='red', marker='*', linestyle='--', markersize=10)

plt.title(f"River Discharge Forecast (Transformer Model)")
plt.ylabel("Discharge (cfs)")
plt.grid(True)
plt.legend()
plt.show()