import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataretrieval.nwis as nwis
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# PART 1: MCP Tool Definition (USGS Data)
# ==========================================
class USGS_MCP_Tool:
    """
    MCP(Model Context Protocol) 스타일로 정의된 USGS 데이터 도구입니다.
    외부 시스템(AI 모델 등)이 호출할 수 있는 인터페이스를 제공합니다.
    """
    
    @staticmethod
    def get_hydrology_data(site_id: str, days: int = 60):
        """
        USGS 사이트 ID를 받아 유량(00060)과 강우량(00045) 데이터를 함께 수집합니다.
        """
        print(f"📡 [MCP Tool] USGS NWIS에 연결 중... Site: {site_id}, 기간: {days}일")
        
        end_date = datetime.date.today().strftime('%Y-%m-%d')
        start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        # 00060: Discharge (유량), 00045: Precipitation (강우량)
        parameter_codes = ['00060', '00045']
        
        try:
            # 다중 파라미터 요청
            df, md = nwis.get_iv(sites=site_id, start=start_date, end=end_date, parameterCd=parameter_codes)
            
            if df.empty:
                raise ValueError("데이터가 없습니다.")

            # 컬럼 정리 및 매핑
            rename_map = {}
            for col in df.columns:
                if '00060' in col and not col.endswith('_cd'):
                    rename_map[col] = 'flow'
                elif '00045' in col and not col.endswith('_cd'):
                    rename_map[col] = 'precip'
            
            df = df.rename(columns=rename_map)
            
            # 필요한 컬럼만 선택
            cols_to_keep = ['flow']
            if 'precip' in df.columns:
                cols_to_keep.append('precip')
            else:
                print("⚠️ 해당 사이트에 강우 데이터(00045)가 없습니다. 유량 데이터만 사용합니다.")
                df['precip'] = 0.0
                cols_to_keep.append('precip')
                
            df = df[cols_to_keep]
            
            # [수정] 데이터 정제: 음수 값 처리 (로그 변환 오류 방지)
            df[df < 0] = np.nan
            
            # 결측치 처리 (유량: 보간, 강우: 0으로 채움)
            df['flow'] = df['flow'].interpolate(method='time')
            df['precip'] = df['precip'].fillna(0)
            
            # 보간 후에도 남아있는 NaN 제거 (데이터 앞쪽 등)
            df = df.dropna()
            
            print(f"✅ [MCP Tool] 데이터 수신 완료: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ [MCP Tool] 데이터 수집 실패: {e}")
            return None

# ==========================================
# PART 2: LSTM Model (PyTorch)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        
        # 마지막 타임스텝의 히든 스테이트 사용
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class WaterLevelPredictor:
    def __init__(self, data_df, look_back=24, forecast_horizon=3):
        """
        Args:
            data_df: 시계열 데이터 (flow, precip 컬럼 포함)
            look_back: 과거 24시간 데이터를 봄
            forecast_horizon: 향후 3시간을 예측함
        """
        self.raw_df = data_df
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        
        # 스케일러 분리
        self.flow_scaler = MinMaxScaler(feature_range=(0, 1))
        self.precip_scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess(self):
        # [수정] 데이터 안정성 확보
        raw_flow = np.maximum(self.raw_df['flow'].values.reshape(-1, 1), 0)
        raw_precip = np.maximum(self.raw_df['precip'].values.reshape(-1, 1), 0)
        
        # 로그 변환 적용
        log_flow = np.log1p(raw_flow) 
        log_precip = np.log1p(raw_precip)
        
        # 스케일링
        scaled_flow = self.flow_scaler.fit_transform(log_flow)
        scaled_precip = self.precip_scaler.fit_transform(log_precip)
        
        # 특성 결합
        combined_data = np.hstack((scaled_flow, scaled_precip))
        
        # NaN 체크
        if np.isnan(combined_data).any():
            print("⚠️ 전처리 중 NaN 발생! 0으로 대체합니다.")
            combined_data = np.nan_to_num(combined_data)
        
        X, y = [], []
        limit = len(combined_data) - self.look_back - self.forecast_horizon + 1
        
        for i in range(limit):
            X.append(combined_data[i : i + self.look_back, :])
            y.append(scaled_flow[i + self.look_back : i + self.look_back + self.forecast_horizon, 0])
            
        X, y = np.array(X), np.array(y)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        return X_tensor, y_tensor

    def build_model(self, input_size=2, hidden_size=64):
        print(f"🧠 [LSTM] PyTorch 모델 구축 중 (Input: {input_size}, Output: {self.forecast_horizon})...")
        self.model = LSTMModel(input_size, hidden_size, output_size=self.forecast_horizon).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X, y, epochs=30, batch_size=32):
        print(f"🏋️ [LSTM] 학습 시작 (Device: {self.device})...")
        
        if torch.isnan(X).any() or torch.isnan(y).any():
            raise ValueError("학습 데이터에 NaN이 포함되어 있습니다. 전처리를 확인하세요.")

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.5f}')

    def predict_future(self):
        self.model.eval()
        with torch.no_grad():
            last_flow = self.raw_df['flow'].values[-self.look_back:].reshape(-1, 1)
            last_precip = self.raw_df['precip'].values[-self.look_back:].reshape(-1, 1)
            
            # 음수 방지 및 로그 변환
            last_flow = np.maximum(last_flow, 0)
            last_precip = np.maximum(last_precip, 0)
            
            log_last_flow = np.log1p(last_flow)
            log_last_precip = np.log1p(last_precip)
            
            scaled_flow = self.flow_scaler.transform(log_last_flow)
            scaled_precip = self.precip_scaler.transform(log_last_precip)
            
            last_combined = np.hstack((scaled_flow, scaled_precip))
            
            X_input = torch.tensor(last_combined, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            predicted_scaled = self.model(X_input).cpu().numpy()
            
            predicted_log = self.flow_scaler.inverse_transform(predicted_scaled)
            predicted_values = np.expm1(predicted_log)
            
            predicted_values = np.maximum(predicted_values, 0.0)
        
        return predicted_values[0]

# ==========================================
# PART 3: Main Execution Flow
# ==========================================
if __name__ == "__main__":
    # Potomac River near Washington, DC
    TARGET_SITE = '01646500' 
    
    print("--- 1. Data Collection via MCP (Flow & Precip) ---")
    df = USGS_MCP_Tool.get_hydrology_data(TARGET_SITE, days=60)
    
    if df is not None:
        print("⚙️ [Processing] 1시간 간격으로 데이터 리샘플링 중...")
        df_resampled = df.resample('1h').agg({'flow': 'mean', 'precip': 'sum'})
        
        df_resampled['flow'] = df_resampled['flow'].interpolate(method='time')
        df_resampled['precip'] = df_resampled['precip'].fillna(0)
        df_resampled = df_resampled.dropna()
        
        # 리샘플링 후에도 음수가 생길 수 있으므로(보간법 등) 0으로 클리핑
        df_resampled[df_resampled < 0] = 0
        
        print(f"✅ 리샘플링 완료: {len(df_resampled)} records")

        print("\n--- 2. Data Preprocessing ---")
        predictor = WaterLevelPredictor(df_resampled, look_back=24, forecast_horizon=3)
        X, y = predictor.preprocess()
        
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        
        print("\n--- 3. Model Training ---")
        predictor.build_model(input_size=2)
        predictor.train(X_train, y_train, epochs=50)
        
        print("\n--- 4. Future Prediction (Next 3 Hours) ---")
        future_vals = predictor.predict_future()
        current_val = df_resampled['flow'].iloc[-1]
        
        print(f"\n🌊 [Result]")
        print(f"현재 유량: {current_val:.2f} ft³/s")
        for i, val in enumerate(future_vals):
            print(f"➡️ {i+1}시간 후 예측: {val:.2f} ft³/s")
        
        print("\n--- 5. Visualization (Interactive) ---")
        # [수정] 인터랙티브 그래프를 위해 subplots 사용
        fig, ax = plt.subplots(figsize=(12, 6))
        
        display_days = 7
        display_data = df_resampled.iloc[-(display_days*24):]
        
        # 유량 데이터 플롯 (변수에 할당하여 이벤트 핸들링에 사용)
        line_obs, = ax.plot(display_data.index, display_data['flow'], label='Observed Flow', color='blue')
        
        # 강우 데이터 (보조축)
        if display_data['precip'].sum() > 0:
            ax2 = ax.twinx()
            ax2.bar(display_data.index, display_data['precip'], color='gray', alpha=0.3, label='Precipitation', width=0.04)
            ax2.set_ylabel('Precipitation', color='gray')
        
        last_time = display_data.index[-1]
        future_times = [last_time + datetime.timedelta(hours=i+1) for i in range(3)]
        
        ax.plot(future_times, future_vals, 'r--', label='Predicted Flow')
        # 예측 포인트 (변수에 할당)
        scat_pred = ax.scatter(future_times, future_vals, color='red', s=100, zorder=5)
        
        for i, (t, v) in enumerate(zip(future_times, future_vals)):
            ax.text(t, v, f'{i+1}h', fontsize=10, verticalalignment='bottom', fontweight='bold', color='darkred')

        ax.set_title(f"USGS Flow Forecast w/ Rainfall (Site: {TARGET_SITE})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Discharge (ft³/s)")
        ax.legend(loc='upper left')
        ax.grid(True)
        
        # ==========================================
        # [기존] Hover Tooltip (커서 올리면 값 표시)
        # ==========================================
        annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind, artist):
            if isinstance(artist, plt.Line2D):
                x, y = artist.get_data()
                idx = ind["ind"][0]
                annot.xy = (x[idx], y[idx])
                text = f"{y[idx]:.2f} ft³/s"
            elif isinstance(artist, type(scat_pred)):
                offsets = artist.get_offsets()
                idx = ind["ind"][0]
                annot.xy = (offsets[idx][0], offsets[idx][1])
                text = f"{offsets[idx][1]:.2f} ft³/s"
            annot.set_text(text)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont_scat, ind_scat = scat_pred.contains(event)
                if cont_scat:
                    update_annot(ind_scat, scat_pred)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return

                cont_line, ind_line = line_obs.contains(event)
                if cont_line:
                    update_annot(ind_line, line_obs)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return

            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        # ==========================================
        # [추가] Zoom Functionality (Mouse Wheel)
        # ==========================================
        def zoom(event):
            base_scale = 1.2 # 확대/축소 비율
            # 현재 x축 범위 가져오기
            cur_xlim = ax.get_xlim()
            cur_xrange = (cur_xlim[1] - cur_xlim[0])
            xdata = event.xdata # 마우스 포인터의 x 좌표
            
            if event.button == 'up':
                # 휠 올리기: 확대 (Zoom In)
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # 휠 내리기: 축소 (Zoom Out)
                scale_factor = base_scale
            else:
                scale_factor = 1
                
            if xdata is None: # 마우스가 그래프 영역 밖에 있으면 무시
                return

            # 새로운 x축 범위 계산 (마우스 위치 중심)
            new_width = cur_xrange * scale_factor
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            
            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
            fig.canvas.draw_idle() # 그래프 갱신

        fig.canvas.mpl_connect('scroll_event', zoom)
        
        print("💡 팁: 그래프 위에서 마우스 휠을 굴려 확대/축소할 수 있습니다.")
        # ==========================================
        
        plt.savefig('prediction_graph_improved.png')
        print("📊 그래프가 'prediction_graph_improved.png' 파일로 저장되었습니다.")
        
        plt.show()

    else:
        print("데이터를 가져오지 못해 프로세스를 종료합니다.")