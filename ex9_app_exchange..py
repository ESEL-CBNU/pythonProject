import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# -----------------------------------------------------------
# [1] 백엔드 로직: 네이버 금융에서 데이터 가져오기
# -----------------------------------------------------------
import requests # 상단에 이 줄이 없으면 추가해주세요
import io 

def get_exchange_data(currency_code, pages=10):
    """
    [수정] 네이버 봇 차단을 피하기 위해 requests와 User-Agent 헤더를 사용합니다.
    """
    df_list = []
    base_url = "https://finance.naver.com/marketindex/exchangeDailyQuote.naver"
    
    # [중요] 브라우저인 척 위장하는 헤더 정보
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }
    
    try:
        for page in range(1, pages + 1):
            url = f"{base_url}?marketindexCd=FX_{currency_code}&page={page}"
            
            # 1. requests로 먼저 접속해서 HTML을 가져옵니다 (헤더 포함)
            response = requests.get(url, headers=headers)
            response.raise_for_status() # 접속 에러 시 즉시 예외 발생
            
            # 2. 가져온 HTML 텍스트를 pandas에게 넘겨줍니다.
            # (StringIO는 문자열을 파일처럼 다루게 해주는 도구입니다)
            dfs = pd.read_html(io.StringIO(response.text), header=1)
            
            if len(dfs) > 0:
                df_list.append(dfs[0])
        
        if not df_list:
            return None
            
        # 여러 페이지 합치기
        df_total = pd.concat(df_list)
        
        # 데이터 정리
        df_total['날짜'] = pd.to_datetime(df_total['날짜'], format='%Y.%m.%d')
        df_total = df_total.sort_values('날짜')
        
        return df_total
        
    except Exception as e:
        print(f"상세 에러 내용: {e}") # 터미널에서 구체적인 에러를 확인하기 위함
        return None
# -----------------------------------------------------------
# [2] 프론트엔드 로직: Tkinter GUI 앱 클래스
# -----------------------------------------------------------
class ExchangeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("실시간 환율 대시보드")
        self.root.geometry("800x600")

        # --- 상단 컨트롤 패널 (검색 옵션) ---
        control_frame = ttk.LabelFrame(root, text="조회 옵션")
        control_frame.pack(side="top", fill="x", padx=10, pady=5)

        # 1. 통화 선택 콤보박스
        ttk.Label(control_frame, text="국가 선택:").pack(side="left", padx=5)
        self.currency_var = tk.StringVar(value="미국 USD")
        self.combo_currency = ttk.Combobox(control_frame, textvariable=self.currency_var, state="readonly")
        self.combo_currency['values'] = ("미국 USD", "유럽연합 EUR", "일본 JPY (100엔)", "중국 CNY")
        self.combo_currency.pack(side="left", padx=5)

        # 2. 기간 선택 라디오 버튼
        ttk.Label(control_frame, text=" |  기간 선택:").pack(side="left", padx=5)
        self.period_var = tk.StringVar(value="3M") # 기본값 3개월
        
        periods = [("1주일", "1W"), ("1개월", "1M"), ("3개월", "3M"), ("1년", "1Y")]
        for text, value in periods:
            ttk.Radiobutton(control_frame, text=text, variable=self.period_var, value=value).pack(side="left", padx=2)

        # 3. 조회 버튼
        self.btn_search = ttk.Button(control_frame, text="조회하기", command=self.search_data)
        self.btn_search.pack(side="left", padx=15)

        # 4. 현재 환율 표시 라벨
        self.lbl_current = ttk.Label(root, text="현재 환율: 조회 대기 중...", font=("맑은 고딕", 14, "bold"), foreground="blue")
        self.lbl_current.pack(pady=10)

        # --- 그래프 영역 (Matplotlib를 Tkinter에 심기) ---
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # 캔버스 생성 (이게 없으면 그래프가 창에 안 뜹니다)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def search_data(self):
        # 1. 사용자 입력 값 가져오기
        selected_name = self.currency_var.get()
        period_mode = self.period_var.get()
        
        # 2. 매핑 (한글 이름 -> 네이버 코드)
        code_map = {
            "미국 USD": "USDKRW",
            "유럽연합 EUR": "EURKRW",
            "일본 JPY (100엔)": "JPYKRW",
            "중국 CNY": "CNYKRW"
        }
        code = code_map.get(selected_name, "USDKRW")
        
        # 3. 데이터 가져오기 (기간에 따라 가져올 페이지 수 조절)
        # 대략 한 페이지에 10일치 데이터가 있음 (주말 제외)
        page_map = {"1W": 2, "1M": 4, "3M": 10, "1Y": 30}
        req_pages = page_map.get(period_mode, 10)
        
        self.lbl_current.config(text="데이터를 가져오는 중입니다...")
        self.root.update() # UI 갱신 강제 수행
        
        df = get_exchange_data(code, pages=req_pages)
        
        if df is None or df.empty:
            messagebox.showerror("에러", "데이터를 가져오지 못했습니다.")
            return

        # 4. 최신 환율 업데이트
        latest_rate = df.iloc[-1]['매매기준율']
        latest_date = df.iloc[-1]['날짜'].strftime('%Y-%m-%d')
        self.lbl_current.config(text=f"[{selected_name}] {latest_date} 기준: {latest_rate:,.2f} 원")

        # 5. 그래프 그리기
        self.ax.clear() # 기존 그림 지우기
        self.ax.plot(df['날짜'], df['매매기준율'], color='red', marker='o', markersize=3)
        
        self.ax.set_title(f"{selected_name} Exchange Rate Trend ({period_mode})")
        self.ax.grid(True, linestyle='--')
        
        # 날짜 포맷 예쁘게
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        self.fig.autofmt_xdate() # 날짜 기울이기

        self.canvas.draw() # 캔버스에 다시 그리기

# -----------------------------------------------------------
# [3] 앱 실행
# -----------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ExchangeApp(root)
    root.mainloop()