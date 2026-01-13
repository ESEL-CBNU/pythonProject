import os
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import shutil
import time
import matplotlib.pyplot as plt  # [추가] 시각화 라이브러리
from scipy.optimize import differential_evolution

class W2AutoCalibrator:
    def __init__(self, model_dir, exe_name="w2_v372.exe", control_file="w2_con.npt"):
        """
        초기화 함수
        """
        self.model_dir = model_dir
        self.exe_path = os.path.join(model_dir, exe_name)
        self.control_file_path = os.path.join(model_dir, control_file)
        self.backup_file_path = os.path.join(model_dir, control_file + ".bak")
        
        # 안전을 위해 원본 파일 백업
        if not os.path.exists(self.backup_file_path):
            shutil.copy(self.control_file_path, self.backup_file_path)
            print(f">>> 원본 제어 파일 백업 완료: {self.backup_file_path}")

    def update_npt_file(self, params):
        """
        제어 파일(w2_con.npt)의 매개변수 값을 업데이트합니다.
        고정 폭(Fixed-width) 형식을 유지하기 위해 숫자가 길어지면 왼쪽 공백을 소비합니다.
        """
        import re
        import os
        
        # 1. 파일 경로 확인 및 보정
        filename = "w2_con.npt"
        if hasattr(self, 'control_file'):
            filename = self.control_file
            
        if not os.path.exists(filename):
            alt_path = r"C:\Users\SEWOONG CHUNG\OneDrive\Python\W2Project\Degray\w2_con.npt"
            if os.path.exists(alt_path):
                filename = alt_path
            
        abs_path = os.path.abspath(filename)
        if not os.path.exists(abs_path):
            print(f"[Error] update_npt_file: 제어 파일을 찾을 수 없습니다 -> {abs_path}")
            return

        # 2. Helper: 고정 폭 유지 교체 (Smart Replacement)
        def replace_fixed_width(line, start, end, new_val_str):
            original_str = line[start:end]
            len_diff = len(new_val_str) - len(original_str)
            
            if len_diff <= 0:
                # 새 값이 더 짧거나 같음: 오른쪽 정렬(왼쪽에 공백 채움)하여 길이 유지
                final_str = new_val_str.rjust(len(original_str))
                return line[:start] + final_str + line[end:]
            else:
                # 새 값이 더 김: 왼쪽의 공백을 소비하여 공간 확보
                prefix = line[:start]
                spaces_available = len(prefix) - len(prefix.rstrip())
                
                if spaces_available >= len_diff:
                    # 충분한 공백이 있음 -> 시작 위치를 앞으로 당김
                    new_start = start - len_diff
                    return line[:new_start] + new_val_str + line[end:]
                else:
                    # 공백 부족 (어쩔 수 없이 밀림 - 경고 출력)
                    print(f"   [Warning] 여유 공백 부족으로 컬럼이 밀릴 수 있음: '{new_val_str}'")
                    return line[:start] + new_val_str + line[end:]

        # 3. Helper: 위치 찾기
        def find_closest_number_token(target_line, target_center):
            number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
            best_match = None
            min_dist = float('inf')
            for m in number_pattern.finditer(target_line):
                token_center = (m.start() + m.end()) / 2
                dist = abs(token_center - target_center)
                if dist < min_dist:
                    min_dist = dist
                    best_match = m
            if min_dist < 15: 
                return best_match
            return None

        try:
            with open(abs_path, 'r', encoding='latin1') as f:
                lines = f.readlines()

            new_lines = lines[:]
            
            for param, value in params.items():
                param_found = False
                new_val_str = f"{value:.5f}" # 원하는 포맷
                
                for i, line in enumerate(lines):
                    # (1) 정확한 단어 매칭
                    match = re.search(rf"\b{re.escape(param)}\b", line)
                    if match:
                        # Case A: 같은 줄
                        rest_of_line = line[match.end():]
                        val_match = re.search(r"\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", rest_of_line)
                        if val_match:
                            start_idx = match.end() + val_match.start(1)
                            end_idx = match.end() + val_match.end(1)
                            new_lines[i] = replace_fixed_width(new_lines[i], start_idx, end_idx, new_val_str)
                            param_found = True
                            break 
                        
                        # Case B: 다음 줄
                        elif i + 1 < len(lines):
                            next_line = lines[i+1]
                            param_center = (match.start() + match.end()) / 2
                            val_token = find_closest_number_token(next_line, param_center)
                            if val_token:
                                start_idx = val_token.start()
                                end_idx = val_token.end()
                                new_lines[i+1] = replace_fixed_width(new_lines[i+1], start_idx, end_idx, new_val_str)
                                param_found = True
                                break
                
                if not param_found:
                    print(f"   [Warning] '{param}' 파라미터를 찾지 못했습니다.")

            with open(abs_path, 'w', encoding='latin1') as f:
                f.writelines(new_lines)
                
        except Exception as e:
            print(f"[Error] 제어 파일 업데이트 오류: {e}")
            
    def run_model(self):
        """모델 실행 함수"""
        try:
            result = subprocess.run(
                [self.exe_path], 
                cwd=self.model_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                print("!!! 모델 실행 중 에러 발생 (Return Code != 0)")
                return False
            return True
        except Exception as e:
            print(f"!!! 실행 파일 호출 오류: {e}")
            return False

    def get_model_results(self, output_file="tsr_1_seg15.opt"):
        """
        모델 결과 파일 읽기
        """
        file_path = os.path.join(self.model_dir, output_file)
        if not os.path.exists(file_path):
             print(f"!!! 결과 파일을 찾을 수 없습니다: {output_file}")
             return None

        try:
            # 헤더 위치 찾기
            header_row_index = -1
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if 'JDAY' in line.upper():
                        header_row_index = i
                        break
            
            if header_row_index == -1:
                header_row_index = 3 # 기본값

            # 데이터 읽기
            df = pd.read_csv(
                file_path, 
                sep=r'\s+', 
                header=0, 
                skiprows=header_row_index,
                on_bad_lines='skip'
            )
            df.columns = df.columns.astype(str).str.upper().str.strip()
            return df

        except Exception as e:
            print(f"!!! 데이터 파싱 오류 ({output_file}): {e}")
            return None

    def _convert_units(self, df, target_variable):
        """
        [추가] 모델 결과의 단위 변환 헬퍼 함수
        PO4, NH4 등은 모델 출력(mg/m3)을 관측값 단위(g/m3)로 맞추기 위해 1000으로 나눔
        """
        if df is None or target_variable not in df.columns:
            return df
            
        # 변환 대상 변수 목록 (필요시 추가)
        conversion_targets = ['PO4', 'NH4', 'NO3', 'TIP', 'TKN']
        
        # 타겟 변수가 변환 목록에 포함되어 있으면 1000으로 나눔
        # (예: 'PO4'가 타겟이거나 컬럼명에 포함된 경우)
        if target_variable in conversion_targets:
            # print(f"Converting units for {target_variable} (mg/m3 -> g/m3)...")
            df[target_variable] = df[target_variable] / 1000.0
            
        return df

    def objective_function(self, params, param_names, obs_data, target_variable):
        """
        최적화 알고리즘에서 호출하는 목적 함수입니다.
        데이터 정제 및 RMSE Flatline(변화 없음) 감지 기능을 포함합니다.
        """
        import os
        import time

        # 1. 매개변수 리스트를 딕셔너리로 변환
        param_dict = dict(zip(param_names, params))

        # 2. 제어 파일 절대 경로 확인
        control_filename = "w2_con.npt"
        if hasattr(self, 'control_file'):
            control_filename = self.control_file
        
        if not os.path.exists(control_filename):
            alt_path = r"C:\Users\SEWOONG CHUNG\OneDrive\Python\W2Project\Degray\w2_con.npt"
            if os.path.exists(alt_path):
                control_filename = alt_path
        
        abs_control_path = os.path.abspath(control_filename)
        
        if not os.path.exists(abs_control_path):
            print(f"\n[Critical Error] 제어 파일을 찾을 수 없습니다: {abs_control_path}")
            raise FileNotFoundError(f"제어 파일 없음: {abs_control_path}")

        # 3. 제어 파일 업데이트 수행
        self.update_npt_file(param_dict)
        
        # 4. 결과 파일 삭제 (Stale Data 방지)
        output_filename = "tsr_1_seg15.opt" 
        if hasattr(self, 'output_file'):
            output_filename = self.output_file
        
        if not os.path.exists(output_filename):
             parent_dir = os.path.dirname(abs_control_path)
             alt_output = os.path.join(parent_dir, os.path.basename(output_filename))
             output_filename = alt_output

        abs_output_path = os.path.abspath(output_filename)

        if os.path.exists(abs_output_path):
            try:
                os.remove(abs_output_path)
            except PermissionError:
                pass 

        # 5. 모델 실행
        self.run_model()

        # 6. 모델 결과 읽기
        for i in range(5): 
            if hasattr(self, 'get_model_results'):
                try:
                    model_df = self.get_model_results()
                    break
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    time.sleep(1)
            else:
                 if hasattr(self, 'read_w2_output'):
                     model_df = self.read_w2_output()
                 else:
                     model_df = self.read_output()
                 break
        else:
             print(f"[Warning] 모델 실행 실패 또는 결과 파일 미생성 (Penalty 부여)")
             return 9999.0

        if target_variable not in model_df.columns:
            return 9999.0

        # 7. 데이터 병합 및 정제
        sim_data = model_df[['JDAY', target_variable]].copy()
        sim_data.rename(columns={target_variable: 'VAL_MODEL'}, inplace=True)

        obs_candidates = [c for c in obs_data.columns if c != 'JDAY']
        if len(obs_candidates) == 1:
            obs_col_name = obs_candidates[0]
        elif f'{target_variable}_obs' in obs_data.columns:
            obs_col_name = f'{target_variable}_obs'
        elif target_variable in obs_data.columns:
            obs_col_name = target_variable
        else:
             raise KeyError(f"실측 데이터 컬럼을 찾을 수 없습니다.")

        real_data = obs_data[['JDAY', obs_col_name]].copy()
        real_data.rename(columns={obs_col_name: 'VAL_OBS'}, inplace=True)

        merged_df = pd.merge(sim_data, real_data, on='JDAY')
        if merged_df.empty:
            return 9999.0

        merged_df['VAL_MODEL'] = pd.to_numeric(merged_df['VAL_MODEL'], errors='coerce')
        merged_df['VAL_OBS'] = pd.to_numeric(merged_df['VAL_OBS'], errors='coerce')
        merged_df.dropna(inplace=True)

        if merged_df.empty or len(merged_df) < 5:
            return 9999.0

        # 8. RMSE 계산
        rmse = np.sqrt(mean_squared_error(merged_df['VAL_OBS'], merged_df['VAL_MODEL']))
        
        # ---------------------------------------------------------
        # [New] RMSE Flatline (변화 없음) 감지
        # ---------------------------------------------------------
        if not hasattr(self, 'last_rmse'):
            self.last_rmse = -1.0
            self.flatline_count = 0
        
        # 소수점 6자리까지 같으면 변화 없음으로 간주
        if abs(self.last_rmse - rmse) < 1e-6:
            self.flatline_count += 1
            if self.flatline_count >= 2: # 2회 연속 동일하면 경고
                print(f"   [Warning] RMSE가 변하지 않습니다 ({rmse:.6f}).")
                print(f"   -> 제어 파일에서 해당 모듈이 켜져 있는지 확인하세요 (예: SEDIMENT=ON ?)")
        else:
            self.flatline_count = 0
            
        self.last_rmse = rmse
        # ---------------------------------------------------------
        
        print(f"RMSE: {rmse:.6f} | Params: {param_dict}")
        return rmse
    
    def calibrate(self, target_params, obs_data, target_variable):
        # 1. Bounds(범위) 및 이름 추출
        param_names = list(target_params.keys())
        bounds = []
        for key in param_names:
            # target_params 값이 [initial, min, max] 라고 가정 -> (min, max) 추출
            bounds.append((target_params[key][1], target_params[key][2]))

        # 2. Differential Evolution 실행
        print(">>> Starting Global Optimization (Differential Evolution)...")
        result = differential_evolution(
            self.objective_function,
            bounds,
            # [핵심 수정] objective_function에 param_names를 전달해야 함
            args=(param_names, obs_data, target_variable),
            strategy='best1bin', # 기본 전략
            maxiter=10,          # 반복 횟수 (너무 크면 오래 걸림)
            popsize=10,          # 개체군 크기 (변수 개수의 10배 권장)
            disp=True,
            workers=1            # 병렬 처리
        )

        return result.x

    # [추가] 그래프 작성을 위한 단독 시뮬레이션 실행 헬퍼 함수
    def run_simulation_for_plot(self, params_values, param_names, target_variable, output_file="tsr_1_seg15.opt"):
        params_dict = dict(zip(param_names, params_values))
        self.update_npt_file(params_dict)
        if self.run_model():
            df = self.get_model_results(output_file)
            if df is not None:
                if 'JDAY' not in df.columns:
                    df.rename(columns={df.columns[0]: 'JDAY'}, inplace=True)
                # [수정] 그래프 작성용 시뮬레이션에서도 단위 변환 적용
                df = self._convert_units(df, target_variable)
            return df
        return None

    # [추가] 결과 비교 그래프 그리기 함수
    def plot_comparison(self, initial_values, best_params, param_names, obs_data, target_variable):
        """
        최적화 전후의 모델 결과와 실측 데이터를 비교하는 그래프를 그립니다.
        """
        import matplotlib.pyplot as plt
        import os

        # --- Helper: 파라미터 딕셔너리 변환 ---
        def create_param_dict(values, names):
            if isinstance(values, dict):
                return {k: (v[0] if isinstance(v, list) else v) for k, v in values.items()}
            return dict(zip(names, values))

        # --- Helper: 안전하게 결과 읽기 ---
        def safe_read_output(calibrator):
            if hasattr(calibrator, 'get_model_results'):
                return calibrator.get_model_results()
            elif hasattr(calibrator, 'read_w2_output'):
                return calibrator.read_w2_output()
            else:
                return calibrator.read_output()
        
        # --- Helper: 파일 강제 삭제 (Stale Data 방지) ---
        def remove_output_file(calibrator):
            filename = "tsr_1_seg15.opt" 
            if hasattr(calibrator, 'output_file'):
                filename = calibrator.output_file
            
            # [수정] 경로 자동 보정
            if not os.path.exists(filename):
                alt_path = r"C:\Users\SEWOONG CHUNG\OneDrive\Python\W2Project\Degray\tsr_1_seg15.opt"
                if os.path.exists(alt_path):
                    filename = alt_path

            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass

        # 1. 초기 모델 실행
        print("\n>>> 결과 비교 그래프를 작성합니다...")
        print("Running Initial Model...")
        remove_output_file(self)
        
        init_param_dict = create_param_dict(initial_values, param_names)
        self.update_npt_file(init_param_dict) 
        self.run_model()
        initial_df = safe_read_output(self)

        # 2. 최적화된 모델 실행
        print("Running Optimized Model...")
        remove_output_file(self)
        
        opt_param_dict = create_param_dict(best_params, param_names)
        self.update_npt_file(opt_param_dict)
        self.run_model()
        optimized_df = safe_read_output(self)

        # 3. 실측 데이터 컬럼 스마트 감지
        obs_candidates = [c for c in obs_data.columns if c != 'JDAY']
        if len(obs_candidates) == 1:
            obs_col_name = obs_candidates[0]
        elif f'{target_variable}_obs' in obs_data.columns:
            obs_col_name = f'{target_variable}_obs'
        elif target_variable in obs_data.columns:
            obs_col_name = target_variable
        else:
            raise KeyError("Target variable not found in obs_data.")

        # 4. 그래프 그리기
        plt.figure(figsize=(12, 6))
        plt.scatter(obs_data['JDAY'], obs_data[obs_col_name], label=f'Observed', color='red', s=20, zorder=5, alpha=0.7)
        plt.plot(initial_df['JDAY'], initial_df[target_variable], label='Initial Model', linestyle='--', color='blue', alpha=0.8, linewidth=1.5)
        plt.plot(optimized_df['JDAY'], optimized_df[target_variable], label='Optimized Model', color='green', linewidth=2.5)
        plt.xlabel('JDAY')
        plt.ylabel(target_variable)
        plt.title(f'Model Calibration Result: {target_variable}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
# ---------------------------------------------------------
# [사용 예제]
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. 모델 폴더 지정
    MODEL_DIR = r"C:\Users\SEWOONG CHUNG\OneDrive\Python\W2Project\Degray"
    
    # 2. 보정할 매개변수 지정
    TARGET_PARAMS = {
        'SEDS': [0.75, 0.2, 1.5],  # [초기값, 최소, 최대]
       # 'TSED': [15.0, 10.0, 20.0]
    }

    # 3. 실측 자료 로드 (CSV 파일 사용)
    obs_file_path = os.path.join(MODEL_DIR, 'SS_obs_data_7day.csv')
    
    if os.path.exists(obs_file_path):
        print(f">>> 실측 데이터 파일 로드: {obs_file_path}")
        obs_data = pd.read_csv(obs_file_path)
        # CSV 파일 컬럼 확인: 'JDAY'와 '{target}_obs'가 있어야 함
    else:
        print("!!! 실측 데이터(SS_obs_data.csv)가 없습니다. 테스트용 데이터를 생성합니다.")
        obs_data = pd.DataFrame({
            'JDAY': [100.5, 105.5, 110.5, 115.5],
            'SS_obs': [2.0013, 2.0025, 1.0012, 1.0018] 
        })

    # 인스턴스 생성 및 보정 시작
    calibrator = W2AutoCalibrator(MODEL_DIR)
    
    # 최적화 수행
    target_var = 'ISS1'
    best_params = calibrator.calibrate(TARGET_PARAMS, obs_data, target_variable=target_var)
    
    # [추가] 결과 그래프 비교
    initial_values = [TARGET_PARAMS[k][0] for k in TARGET_PARAMS.keys()]
    param_names = list(TARGET_PARAMS.keys())
    
    calibrator.plot_comparison(initial_values, best_params, param_names, obs_data, target_variable=target_var)