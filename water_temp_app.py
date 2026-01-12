import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import datetime

# ---------------------------------------------------------
# 1. 페이지 설정 및 스타일
# ---------------------------------------------------------
st.set_page_config(
    page_title="하천 수온 예측 AI",
    page_icon="🌊",
    layout="wide"
)

# 한글 폰트 설정 (Windows/Linux/Mac 대응)
import platform
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# ---------------------------------------------------------
# 2. 모델 및 데이터 설정
# ---------------------------------------------------------
# 전역 상수
PREDICT_DAYS = 7
LOOK_BACK = 30
HIDDEN_DIM = 64
NUM_LAYERS = 2
EPOCHS = 50 # 웹 앱 구동 속도를 위해 Epoch 조정

class WaterTempLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(WaterTempLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_DIM).to(x.device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_DIM).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

@st.cache_resource # 이 함수는 앱 실행 시 한 번만 실행되어 모델을 메모리에 캐싱합니다.
def train_and_get_model():
    # 1. 데이터 생성
    np.random.seed(42)
    end_date_str = '2024-12-31'
    years = 5
    
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.DateOffset(years=years) + pd.DateOffset(days=1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    days = len(dates)
    t = np.arange(days)
    
    air_temp = 12.5 + 17.5 * np.sin(2 * np.pi * t / 365 - np.pi/2) + np.random.normal(0, 2, days)
    lag = 7
    water_temp = 12.5 + 14 * np.sin(2 * np.pi * (t - lag) / 365 - np.pi/2) + np.random.normal(0, 0.8, days)
    
    df = pd.DataFrame({'Date': dates, 'Air_Temp': air_temp, 'Water_Temp': water_temp})
    
    # 2. 전처리
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 전체 데이터를 학습에 사용하여 예측 범위를 넓힘
    X_scaled = scaler_X.fit_transform(df[['Air_Temp', 'Water_Temp']])
    y_scaled = scaler_y.fit_transform(df[['Water_Temp']])
    
    def create_sequences(data_X, data_y, look_back, predict_days):
        Xs, ys = [], []
        for i in range(len(data_X) - look_back - predict_days + 1):
            Xs.append(data_X[i:(i + look_back)])
            ys.append(data_y[i + look_back : i + look_back + predict_days].flatten())
        return np.array(Xs), np.array(ys)
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOK_BACK, PREDICT_DAYS)
    
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq)
    
    # 3. 모델 학습
    model = WaterTempLSTM(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=PREDICT_DAYS, num_layers=NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # 진행률 표시 (첫 로딩 시에만 보임)
        if (epoch + 1) % 10 == 0:
            progress = (epoch + 1) / EPOCHS
            progress_bar.progress(progress)
            status_text.text(f"AI 모델 학습 중... Epoch {epoch+1}/{EPOCHS}")
            
    progress_bar.empty()
    status_text.empty()
    
    return model, df, scaler_X, scaler_y

# ---------------------------------------------------------
# 3. 메인 UI 및 로직
# ---------------------------------------------------------
st.title("🌊 하천 수온 예측 AI 시스템")
st.markdown("""
이 시스템은 **LSTM 딥러닝 모델**을 사용하여 기온 데이터를 기반으로 향후 7일간의 하천 수온을 예측합니다.
좌측 메뉴에서 기준 날짜를 선택해주세요.
""")

# 모델 로드 (캐싱됨)
with st.spinner("모델을 불러오고 있습니다... (최초 실행 시 1~2분 소요될 수 있습니다)"):
    model, df, scaler_X, scaler_y = train_and_get_model()

# 사이드바 설정
st.sidebar.header("📅 예측 설정")
min_date = df['Date'].iloc[LOOK_BACK].date()
max_date = df['Date'].iloc[-PREDICT_DAYS-1].date()

# 날짜 선택 (기본값: 데이터의 마지막 가능한 날짜)
selected_date = st.sidebar.date_input(
    "기준 날짜 선택",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    help="선택한 날짜를 기준으로 과거 30일 데이터를 분석하여, 다음 날부터 7일간의 수온을 예측합니다."
)

if st.sidebar.button("수온 예측하기", type="primary"):
    # 1. 선택된 날짜의 인덱스 찾기
    selected_date_pd = pd.to_datetime(selected_date)
    base_idx = df[df['Date'] == selected_date_pd].index[0]
    
    # 2. 입력 데이터 추출 (과거 30일)
    start_idx = base_idx - LOOK_BACK + 1
    end_idx = base_idx + 1 # slicing은 끝 인덱스 포함 안하므로 +1
    
    input_data = df.iloc[start_idx:end_idx]
    
    # 3. 전처리 & 텐서 변환
    input_scaled = scaler_X.transform(input_data[['Air_Temp', 'Water_Temp']])
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0) # (1, 30, 2)
    
    # 4. 예측 수행
    model.eval()
    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()
        pred_values = scaler_y.inverse_transform(pred_scaled).flatten()
        
    # 5. 결과 정리
    future_dates = pd.date_range(start=selected_date_pd + pd.Timedelta(days=1), periods=PREDICT_DAYS)
    
    result_df = pd.DataFrame({
        '날짜': future_dates,
        '예측 수온(°C)': np.round(pred_values, 2)
    })
    
    # ---------------------------------------------------------
    # 4. 결과 시각화
    # ---------------------------------------------------------
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 예측 결과표")
        # 날짜 포맷 예쁘게 변경하여 표시
        display_df = result_df.copy()
        display_df['날짜'] = display_df['날짜'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        avg_temp = pred_values.mean()
        st.info(f"향후 7일 평균 수온: **{avg_temp:.1f}°C**")

    with col2:
        st.subheader("📈 시계열 분석 그래프")
        
        # 그래프 데이터 준비 (과거 60일 + 미래 7일)
        history_start = base_idx - 60
        history_data = df.iloc[history_start : base_idx + 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 과거 데이터 (기온 & 수온)
        ax.plot(history_data['Date'], history_data['Water_Temp'], 
                label='실제 수온 (과거)', color='black', alpha=0.7)
        ax.plot(history_data['Date'], history_data['Air_Temp'], 
                label='기온 (참조)', color='gray', linestyle=':', alpha=0.5)
        
        # 현재 시점 표시
        current_temp = df.iloc[base_idx]['Water_Temp']
        ax.plot(selected_date_pd, current_temp, marker='D', markersize=8, color='purple', zorder=10)
        ax.text(selected_date_pd, current_temp + 1, "기준일", ha='center', color='purple', fontweight='bold')
        
        # 미래 예측
        ax.plot(future_dates, pred_values, label='AI 예측 수온', 
                color='red', marker='o', linestyle='-', linewidth=2)
        
        # 연결선
        ax.plot([selected_date_pd, future_dates[0]], [current_temp, pred_values[0]], 
                color='red', linestyle='-')

        ax.set_title(f"수온 예측 ({selected_date_pd.strftime('%Y-%m-%d')} 기준)", fontweight='bold')
        ax.set_ylabel("온도 (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        st.pyplot(fig)

else:
    st.info("👈 좌측 사이드바에서 날짜를 선택하고 '수온 예측하기' 버튼을 눌러주세요.")

# ---------------------------------------------------------
# 5. 앱 정보
# ---------------------------------------------------------
st.markdown("---")
st.caption("Developed by ESEL of CBNU | Powered by PyTorch & Streamlit")