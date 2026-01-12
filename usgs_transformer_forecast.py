import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------
# [1] 설정 및 데이터 수집 (USGS Daily Value API)
# -----------------------------------------------------------
SITE_ID = '09380000'  # Colorado River at Lees Ferry
PARAM_CD = '00060'    # [수정됨] 00060: 유량 (Discharge, cubic feet per second)
STAT_CD = '00003'     # Daily Mean (일 평균값)
PERIOD = 'P10Y'       # 최근 10년

INPUT_WINDOW = 30     # AI가 참고할 과거 기간 (30일)
OUTPUT_WINDOW = 3     # AI가 예측할 미래 기간 (3일)
BATCH_SIZE = 32
EPOCHS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_usgs_daily_data(site_id, period):
    # 1차 시도: 사용자가 요청한 기간(P10Y) 시도
    periods_to_try = [period, 'P1Y', 'P1M'] # 10년 -> 1년 -> 1개월 순으로 재시도
    
    for p in periods_to_try:
        print(f">>> 데이터 다운로드 시도 중... (기간: {p})")
        url = f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_id}&period={p}&parameterCd={PARAM_CD}&statCd={STAT_CD}"
        
        try:
            resp = requests.get(url)
            
            # 서버 응답 코드가 200(성공)이 아니면 에러 내용 출력
            if resp.status_code != 200:
                print(f"Warning: 서버 응답 코드 {resp.status_code}")
                # 서버가 보낸 에러 메시지 일부를 출력해봄
                print(f"서버 메시지: {resp.text[:100]}...") 
                continue # 다음 기간으로 재시도
                
            data = resp.json() # 여기서 에러가 났던 것임
            
            # 데이터가 비어있는지 확인
            if not data['value']['timeSeries']:
                print(f"알림: 기간 {p}에 데이터가 없습니다.")
                continue

            ts_data = data['value']['timeSeries'][0]['values'][0]['value']
            
            dates = [item['dateTime'] for item in ts_data]
            values = [float(item['value']) for item in ts_data]
            
            df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Value': values})
            df = df.set_index('Date')
            
            # 결측치 보간
            df = df.resample('D').mean().interpolate()
            print(f">>> 성공! 총 {len(df)}일 데이터 수집 완료.")
            return df

        except Exception as e:
            print(f"기간 {p} 시도 중 에러 발생: {e}")
            # 다음 짧은 기간으로 재시도
            continue
            
    print("!!! 모든 시도가 실패했습니다.")
    return None
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

# 데이터 로드
df = get_usgs_daily_data(SITE_ID, PERIOD)

# [안전장치] 데이터가 없으면 종료
if df is None:
    exit()

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim) 

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer(x)
        last_hidden = x[:, -1, :]
        output = self.fc_out(last_hidden)
        return output

model = TransformerPredictor(output_dim=OUTPUT_WINDOW).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# -----------------------------------------------------------
# [4] 모델 학습
# -----------------------------------------------------------
print(f">>> 학습 시작 ({device} 모드)...")
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

last_30_days = scaled_data[-INPUT_WINDOW:]
last_30_days_tensor = torch.FloatTensor(last_30_days).unsqueeze(0).to(device)

with torch.no_grad():
    predicted_scaled = model(last_30_days_tensor).cpu().numpy()

# 스케일 역변환 (원래 단위 cfs로)
predicted_cfs = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=3)

print("\n================ [예측 결과] ================")
print(f"기준일: {last_date.date()} (유량: {df['Value'].iloc[-1]:.2f} cfs)")
for date, val in zip(future_dates, predicted_cfs):
    print(f"예측일: {date.date()} -> 예측 유량: {val[0]:.2f} cfs")
print("=============================================")

# 그래프 그리기
plt.figure(figsize=(10, 6))
recent_df = df.iloc[-90:] # 최근 90일만 표시
plt.plot(recent_df.index, recent_df['Value'], label='History (Discharge)', color='blue')

plot_dates = [last_date] + list(future_dates)
plot_values = [df['Value'].iloc[-1]] + list(predicted_cfs.flatten())

plt.plot(plot_dates, plot_values, label='Transformer Forecast', 
         color='red', marker='o', linestyle='--', markersize=6)

plt.title(f"USGS Discharge Forecast: {SITE_ID} (Lees Ferry)")
plt.ylabel("Discharge (cfs)")
plt.grid(True)
plt.legend()
plt.show()