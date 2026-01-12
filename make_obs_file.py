import pandas as pd
import numpy as np
import os

# 1. 파일 설정
input_file = 'SS_obs_data.csv'       # 원본 파일 (0.25일 간격)
output_file = 'SS_obs_data_7day.csv' # 생성될 파일 (7일 간격 + 오차)

# 노이즈 설정 (표준편차 비율, 예: 10%)
NOISE_LEVEL = 0.10 

def create_calibration_data():
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' 파일을 찾을 수 없습니다.")
        return

    print(f">>> '{input_file}' 읽기 중...")
    
    # CSV 읽기
    df = pd.read_csv(input_file)
    
    # 컬럼 공백 제거 (안전장치)
    df.columns = df.columns.str.strip()
    
    # 필수 컬럼 확인
    if 'JDAY' not in df.columns or 'SS' not in df.columns:
        print("Error: 파일에 'JDAY' 또는 'SS' 컬럼이 없습니다.")
        return

    # 2. 7일 간격 평균 계산 (Binning)
    print(">>> 7일 간격 평균 계산 중...")
    
    # JDAY의 최소값부터 최대값까지 7일 간격 구간 생성
    start_day = np.floor(df['JDAY'].min())
    end_day = np.ceil(df['JDAY'].max())
    bins = np.arange(start_day, end_day + 7, 7)
    
    # 각 데이터를 7일 구간에 할당
    df['bin'] = pd.cut(df['JDAY'], bins=bins, labels=bins[:-1])
    
    # 구간별 평균 계산
    # observed=True는 범주형 데이터 경고 방지용
    grouped = df.groupby('bin', observed=True)['SS'].mean().reset_index()
    
    # 3. 데이터 가공
    # 대표 시간(JDAY) 설정: 구간의 시작점 + 3.5일 (중간값)
    grouped['JDAY'] = grouped['bin'].astype(float) + 3.5
    
    # 컬럼명 변경 (W2 보정 코드와 맞춤: SS -> SS_obs)
    grouped.rename(columns={'SS': 'SS_obs'}, inplace=True)
    
    # 데이터가 없는 구간(NaN) 제거
    grouped.dropna(subset=['SS_obs'], inplace=True)
    
    # 4. 난수 오차(Noise) 추가
    print(f">>> 랜덤 오차 추가 중 (Noise Level: {NOISE_LEVEL*100}%)")
    np.random.seed(42) # 결과 재현을 위해 시드 고정 (원치 않으면 삭제)
    
    # 정규분포 노이즈 생성 (평균 0, 표준편차 10%)
    noise = np.random.normal(0, NOISE_LEVEL, len(grouped))
    
    # 원본 값에 (1 + noise)를 곱하여 변동성 부여
    grouped['SS_obs'] = grouped['SS_obs'] * (1 + noise)
    
    # 음수 방지 (수질 농도는 음수가 될 수 없음)
    grouped['SS_obs'] = grouped['SS_obs'].apply(lambda x: max(x, 0.0))

    # 5. 저장
    final_df = grouped[['JDAY', 'SS_obs']].round(5) # 소수점 5자리 반올림
    final_df.to_csv(output_file, index=False)
    
    print("="*40)
    print(f"처리 완료! '{output_file}' 파일이 생성되었습니다.")
    print(f"데이터 개수: {len(df)}개 (원본) -> {len(final_df)}개 (7일 간격)")
    print("="*40)
    print(final_df.head())

if __name__ == "__main__":
    create_calibration_data()