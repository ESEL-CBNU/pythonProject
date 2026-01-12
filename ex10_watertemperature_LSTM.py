import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import matplotlib.dates as mdates

# 경고 메시지 무시 및 스타일 설정
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 데이터 생성 (대한민국 기후 모사)
# ---------------------------------------------------------
def generate_climate_data(end_date_str='2024-12-23', years=5):
    """
    대한민국의 4계절 특성을 반영한 기온 및 수온 데이터 생성
    [수정] 종료일을 '2024-12-23'으로 고정하여 시나리오와 일치시킴
    """
    np.random.seed(42)
    
    # 날짜 범위 생성 (종료일 기준 역산)
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.DateOffset(years=years) + pd.DateOffset(days=1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    days = len(dates)
    t = np.arange(days)
    
    # 기온 생성 (사인파 + 노이즈)
    air_temp = 12.5 + 17.5 * np.sin(2 * np.pi * t / 365 - np.pi/2) + np.random.normal(0, 2, days)
    
    # 수온 생성 (기온 기반 + 지연 효과 + 열관성)
    lag = 7
    water_temp = 12.5 + 14 * np.sin(2 * np.pi * (t - lag) / 365 - np.pi/2) + np.random.normal(0, 0.8, days)
    
    return pd.DataFrame({'Date': dates, 'Air_Temp': air_temp, 'Water_Temp': water_temp})

print(">>> [1] 대한민국 기후 데이터 생성 중 (2024-12-23 종료)...")
df = generate_climate_data(end_date_str='2024-12-23', years=5)

# ---------------------------------------------------------
# 2. 데이터 전처리
# ---------------------------------------------------------
PREDICT_DAYS = 7  
LOOK_BACK = 30    

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size].reset_index(drop=True)
test_df = df.iloc[train_size:].reset_index(drop=True)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X_scaled = scaler_X.fit_transform(train_df[['Air_Temp', 'Water_Temp']])
train_y_scaled = scaler_y.fit_transform(train_df[['Water_Temp']])
test_X_scaled = scaler_X.transform(test_df[['Air_Temp', 'Water_Temp']])
test_y_scaled = scaler_y.transform(test_df[['Water_Temp']])

def create_sequences(data_X, data_y, look_back, predict_days):
    Xs, ys = [], []
    for i in range(len(data_X) - look_back - predict_days + 1):
        Xs.append(data_X[i:(i + look_back)])
        ys.append(data_y[i + look_back : i + look_back + predict_days].flatten())
    return np.array(Xs), np.array(ys)

print(">>> [2] 시퀀스 데이터 변환 중...")
X_train, y_train = create_sequences(train_X_scaled, train_y_scaled, LOOK_BACK, PREDICT_DAYS)
X_test, y_test = create_sequences(test_X_scaled, test_y_scaled, LOOK_BACK, PREDICT_DAYS)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# ---------------------------------------------------------
# 3. 모델 정의 및 학습 (LSTM)
# ---------------------------------------------------------
class WaterTempLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(WaterTempLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_dim = 2
hidden_dim = 64
output_dim = PREDICT_DAYS
num_layers = 2
learning_rate = 0.01
epochs = 100

model = WaterTempLSTM(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(">>> [3] LSTM 모델 학습 시작...")
dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}")

# ---------------------------------------------------------
# 4. 비교 모델 (선형회귀) 학습
# ---------------------------------------------------------
print(">>> [4] 비교 모델(Linear Regression) 학습 중...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train)

# ---------------------------------------------------------
# 5. [수정] 일별 예측 성능 평가 (Day+1 ~ Day+7)
# ---------------------------------------------------------
print(">>> [5] 예측 성능 정밀 평가 (By Forecast Horizon)...")

# 전체 Test 셋에 대한 예측 수행
model.eval()
with torch.no_grad():
    lstm_pred_scaled = model(X_test_t).numpy()
    
lr_pred_scaled = lr_model.predict(X_test_flat)

# 역정규화 (실제 수온 단위로 변환)
lstm_pred_full = scaler_y.inverse_transform(lstm_pred_scaled)
lr_pred_full = scaler_y.inverse_transform(lr_pred_scaled)
actual_full = scaler_y.inverse_transform(y_test)

# 일별(Horizon별) 성능 계산
metrics_by_day = []

for day in range(PREDICT_DAYS):
    # 각 모델의 day번째 컬럼(예: Day+1, Day+2...) 추출
    y_true = actual_full[:, day]
    y_pred_lstm = lstm_pred_full[:, day]
    y_pred_lr = lr_pred_full[:, day]
    
    # LSTM 성능
    rmse_lstm = np.sqrt(mean_squared_error(y_true, y_pred_lstm))
    r2_lstm = r2_score(y_true, y_pred_lstm)
    
    # LR 성능
    rmse_lr = np.sqrt(mean_squared_error(y_true, y_pred_lr))
    r2_lr = r2_score(y_true, y_pred_lr)
    
    metrics_by_day.append([f"Day+{day+1}", rmse_lstm, r2_lstm, rmse_lr, r2_lr])

# 결과 출력
metrics_df = pd.DataFrame(metrics_by_day, columns=['Horizon', 'LSTM_RMSE', 'LSTM_R2', 'LR_RMSE', 'LR_R2'])
print("\n>>> [일별 예측 성능 비교 (Performance by Forecast Horizon)]")
print(metrics_df.round(4))

# ---------------------------------------------------------
# 6. 미래 예측 및 시각화 (개선된 버전)
# ---------------------------------------------------------
# 미래 7일 예측 (가장 최근 데이터 기준)
last_sequence_scaled = test_X_scaled[-LOOK_BACK:] 
last_sequence_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0)

model.eval()
with torch.no_grad():
    future_pred_scaled = model(last_sequence_tensor).numpy()
    future_pred_lstm = scaler_y.inverse_transform(future_pred_scaled).flatten()

# [수정] 미래 예측 기간 설정: 마지막 날짜 바로 다음 날부터 7일
last_date = test_df.iloc[-1]['Date']
# last_date가 12월 23일이면, future_days는 12월 24일 ~ 12월 30일
future_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICT_DAYS)

# --- 결과표 출력 (Console) ---
forecast_df = pd.DataFrame({
    '날짜': future_days.strftime('%Y-%m-%d'),
    '예측 수온(°C)': np.round(future_pred_lstm, 2)
})
print("\n" + "="*40)
print(f"   [향후 {PREDICT_DAYS}일 하천 수온 예측 결과]")
print("="*40)
print(forecast_df.to_string(index=False))
print("="*40 + "\n")

# --- 그래프 그리기 (2개의 Subplot) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# [Graph 1] 일별 RMSE 변화 (성능 비교)
ax1 = axes[0]
x_days = range(1, PREDICT_DAYS + 1)
ax1.plot(x_days, metrics_df['LSTM_RMSE'], marker='o', label='LSTM RMSE', linewidth=2, color='blue')
ax1.plot(x_days, metrics_df['LR_RMSE'], marker='s', label='Linear Regression RMSE', linewidth=2, color='green', linestyle='--')
ax1.set_title('선행 예측 기간별 모델 오차 비교 (Lower is Better)', fontsize=14, fontweight='bold')
ax1.set_xlabel('선행 예측 일수 (Lead Time)', fontsize=12)
ax1.set_ylabel('RMSE (°C)', fontsize=12)
ax1.set_xticks(x_days)
ax1.set_xticklabels([f"Day+{d}" for d in x_days])
ax1.legend()
ax1.grid(True, alpha=0.3)

# [Graph 2] 시계열 예측 결과 (최근 100일 + 미래 7일)
ax2 = axes[1]
plot_len = 100

# [중요 수정] 실제 관측 이력을 test_df 전체에서 가져옴 (끊김 방지)
# 이전 방식은 평가(eval)용 배열 길이에 의존하여 데이터 끝부분 7일이 잘려나갔음.
history_dates_full = test_df['Date'].values[-plot_len:]
history_y_full = test_df['Water_Temp'].values[-plot_len:]

# 모델 평가 라인은 평가 데이터가 존재하는 구간만 표시 (날짜 매핑)
# y_test는 LOOK_BACK 시점부터 시작함
eval_start_idx = LOOK_BACK
eval_end_idx = eval_start_idx + len(lstm_pred_full)
eval_dates = test_df['Date'].iloc[eval_start_idx : eval_end_idx].values
# plot_len 구간에 해당하는 eval 데이터만 필터링
mask = np.isin(eval_dates, history_dates_full)

ax2.plot(history_dates_full, history_y_full, label='실제 수온', color='black', alpha=0.6, linewidth=1.5)
ax2.plot(eval_dates[mask], lstm_pred_full[mask, 0], label='LSTM (Test Eval)', color='blue', linestyle='--', alpha=0.6)
ax2.plot(eval_dates[mask], lr_pred_full[mask, 0], label='Linear Reg (Test Eval)', color='green', linestyle=':', alpha=0.8)

# [개선] 현재 시점(기준일) 표시 - 데이터의 진짜 마지막 날짜 사용 (Dec 23)
current_date = history_dates_full[-1]
current_temp = history_y_full[-1]
ax2.plot(current_date, current_temp, marker='D', markersize=9, color='purple', label='현재 시점 (Baseline)', zorder=15)

# 현재 날짜 텍스트 강조
# [FIX] numpy.datetime64 객체는 strftime이 없으므로 pd.to_datetime으로 변환
current_date_pd = pd.to_datetime(current_date)
ax2.text(current_date, current_temp + 1.5, 
         f"Current: {current_date_pd.strftime('%Y-%m-%d')}\n{current_temp:.1f}°C", 
         ha='center', va='bottom', fontsize=10, fontweight='bold', 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.9))

# [개선] 미래 예측 (심볼과 선)
ax2.plot(future_days, future_pred_lstm, label='LSTM Future (Next 7 Days)', 
         color='red', marker='o', markersize=7, linewidth=2.5, zorder=10)
# 연결선
ax2.plot([current_date, future_days[0]], [current_temp, future_pred_lstm[0]], 
         color='red', linestyle='-', linewidth=2.5, zorder=10)

# 미래 예측값 텍스트 표시
for x, y in zip(future_days, future_pred_lstm):
    ax2.text(x, y + 0.5, f"{y:.1f}", ha='center', color='red', fontweight='bold', fontsize=9)

ax2.set_title('하천 수온 예측 결과 및 향후 7일 전망', fontsize=14, fontweight='bold')
ax2.set_xlabel('날짜', fontsize=12)
ax2.set_ylabel('수온 (°C)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) # 월-일 포맷으로 변경하여 가독성 확보

fig.autofmt_xdate()
plt.tight_layout()
plt.show()