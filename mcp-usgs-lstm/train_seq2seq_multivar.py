# train_seq2seq_multivar.py
from __future__ import annotations

import argparse
import json
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import anyio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import webbrowser

# -------------------------
# MCP 결과 JSON 추출
# -------------------------
def _extract_payload(result) -> dict:
    # 1) FastMCP(json_response=True)라면 보통 structuredContent로 들어옵니다. :contentReference[oaicite:1]{index=1}
    if hasattr(result, "structuredContent") and result.structuredContent:
        return result.structuredContent

    # 2) fallback: TextContent 안에 JSON 문자열로 들어오는 경우 파싱 :contentReference[oaicite:2]{index=2}
    for content in getattr(result, "content", []):
        if isinstance(content, types.TextContent) and getattr(content, "text", None):
            try:
                return json.loads(content.text)
            except Exception:
                pass

    # 3) 에러 처리
    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP tool returned error: {result}")

    raise ValueError("Could not extract structuredContent or JSON text from MCP tool result.")


def plot_30d_with_forecast(df_u: pd.DataFrame, pred_series: pd.Series, out_html: str = "usgs_30d_forecast.html"):
    """
    한 패널(dual y-axis)에서:
    - 좌측 y: Stage / Gage height (00065) 관측 + 6h 예측(라인+심볼)
    - 우측 y: Discharge (00060) 관측
    - hover로 값 확인, 드래그/휠 줌, rangeslider 제공
    """
    # 최근 30일만 자르기
    end_t = df_u.index.max()
    start_t = end_t - pd.Timedelta(days=30)
    df30 = df_u.loc[df_u.index >= start_t].copy()

    # dual y-axis subplot
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
        subplot_titles=("Stage(00065) + 6h Forecast (left y)  |  Discharge(00060) observed (right y)",)
    )

    # 수위(실측) - left y
    fig.add_trace(
        go.Scatter(
            x=df30.index, y=df30["00065"],
            mode="lines",
            name="Stage (00065) observed",
            hovertemplate="Time: %{x}<br>Stage: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False
    )

    # 수위(예측) - left y (라인+심볼로 강조; 색상은 Plotly가 자동으로 다르게 배정)
    fig.add_trace(
        go.Scatter(
            x=pred_series.index, y=pred_series.values,
            mode="lines+markers",
            name="Stage (00065) forecast (next 6h)",
            marker=dict(size=8, symbol="diamond"),
            line=dict(width=3),
            hovertemplate="Time: %{x}<br>Forecast stage: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False
    )

    # 유량(실측) - right y
    fig.add_trace(
        go.Scatter(
            x=df30.index, y=df30["00060"],
            mode="lines",
            name="Discharge (00060) observed",
            hovertemplate="Time: %{x}<br>Discharge: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=True
    )

    # 예측 시작선(현재 시각) - add_vline 대신 add_shape (pandas Timestamp 문제 회피)
    t0 = pd.Timestamp(df_u.index.max()).to_pydatetime()
    fig.add_shape(
        type="line",
        x0=t0, x1=t0,
        y0=0, y1=1,
        xref="x",
        yref="paper",
        line=dict(width=2, dash="dash"),
    )
    fig.add_annotation(
        x=t0, y=1,
        xref="x", yref="paper",
        text="forecast start",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
    )

    # 축 라벨
    fig.update_yaxes(title_text="Stage / Gage height (00065)", secondary_y=False)
    fig.update_yaxes(title_text="Discharge (00060)", secondary_y=True)

    # hover/zoom UX
    fig.update_layout(
        title="USGS 30-day observed (Stage+Discharge) + 6-hour stage forecast",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0),
        height=700,
        margin=dict(l=70, r=70, t=90, b=70),
    )

    # rangeslider + 커서 스파이크(줌/값 읽기 편하게)
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
    )
    fig.update_yaxes(showspikes=True, spikesnap="cursor")

    out_path = Path(out_html).resolve()
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"\n[Saved interactive plot] {out_path}")
    webbrowser.open(out_path.as_uri())


async def mcp_fetch_iv_multi(site_no: str, start: str, end: str, parameterCds: list[str]) -> pd.DataFrame:
    # Use absolute path to the stdio server script so it can be launched
    server_script = Path(__file__).parent.joinpath("server_usgs_stage_mcp.py").resolve()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "usgs_get_iv_multi",
                {"site_no": site_no, "start": start, "end": end, "parameterCds": parameterCds},
            )
            payload = _extract_payload(result)

    df = pd.DataFrame(payload["points"])
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).set_index("t").sort_index()
    # 컬럼은 '00060','00065' 또는 변형된 이름이 payload["columns"]에 있음
    cols = payload["columns"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols]


# -------------------------
# 관측 간격 자동 정규화
# -------------------------
def infer_target_freq(index: pd.DatetimeIndex, candidates_min=(5, 15, 60)) -> pd.Timedelta:
    idx = index.sort_values()
    diffs = pd.Series(idx[1:]) - pd.Series(idx[:-1])
    diffs = diffs.dropna()
    if len(diffs) == 0:
        return pd.Timedelta(minutes=15)

    # 초 단위로 변환 후, "대표 간격" 추정(중앙값 + 최빈 근사)
    secs = diffs.dt.total_seconds().astype(int)
    # 극단값 제거(상위/하위 2% 컷)로 튐 완화
    lo, hi = np.quantile(secs, 0.02), np.quantile(secs, 0.98)
    secs_f = secs[(secs >= lo) & (secs <= hi)]
    if len(secs_f) == 0:
        secs_f = secs

    # 빈도 후보에 가장 잘 맞는 것을 선택:
    # - 후보 간격(초) 각각에 대해 |diff - 후보|가 작은 비율이 가장 큰 후보를 선택
    cand_secs = np.array([m * 60 for m in candidates_min], dtype=int)
    tol = 30  # 30초 이내면 같은 간격으로 간주
    scores = []
    for cs in cand_secs:
        scores.append(np.mean(np.abs(secs_f - cs) <= tol))
    best = int(np.argmax(scores))

    # 점수가 너무 낮으면 중앙값 기준으로 가장 가까운 후보
    if scores[best] < 0.25:
        med = int(np.median(secs_f))
        best = int(np.argmin(np.abs(cand_secs - med)))

    return pd.Timedelta(seconds=int(cand_secs[best]))


def normalize_to_grid(
    df: pd.DataFrame,
    target_freq: Optional[pd.Timedelta] = None,
    candidates_min=(5, 15, 60),
    max_interp_gap_steps: int = 2,
) -> Tuple[pd.DataFrame, pd.Timedelta]:
    """
    df: index=DatetimeIndex(UTC 권장), columns=['00060','00065'...]
    - target_freq가 없으면 후보(5/15/60분)에서 자동 선택
    - 정확한 그리드로 reindex 후, 짧은 결측만 보간/ffill
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if target_freq is None:
        target_freq = infer_target_freq(df.index, candidates_min=candidates_min)

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=target_freq, tz=df.index.tz)
    out = df.reindex(full_idx)

    # 보간 전략:
    # - 수위(00065): 선형 보간(짧은 결측만)
    # - 유량(00060): 보간 + 필요시 ffill(짧은 결측만)
    # 긴 결측은 남겨서 학습 샘플 생성 시 제외되도록 함
    if "00065" in out.columns:
        out["00065"] = out["00065"].interpolate(limit=max_interp_gap_steps, limit_direction="both")
    if "00060" in out.columns:
        out["00060"] = out["00060"].interpolate(limit=max_interp_gap_steps, limit_direction="both")
        out["00060"] = out["00060"].ffill(limit=max_interp_gap_steps)

    return out, target_freq


# -------------------------
# Seq2Seq 데이터셋 생성
# -------------------------
def make_seq2seq_supervised(
    df: pd.DataFrame,
    freq: pd.Timedelta,
    window_hours=24.0,
    horizon_hours=6.0,
    x_cols=("00060", "00065"),
    y_col="00065",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (N, Tin, F)
    y: (N, Tout, 1)  -> 미래 수위 시퀀스
    """
    # 필요한 컬럼 존재 확인
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing input column: {c}")
    if y_col not in df.columns:
        raise ValueError(f"Missing target column: {y_col}")

    Tin = int(round(pd.Timedelta(hours=window_hours) / freq))
    Tout = int(round(pd.Timedelta(hours=horizon_hours) / freq))
    if Tin < 10 or Tout < 2:
        raise ValueError("Tin/Tout too small. Check freq/window/horizon.")

    X_list, y_list = [], []
    values_x = df.loc[:, list(x_cols)].values.astype(np.float32)
    values_y = df.loc[:, [y_col]].values.astype(np.float32)

    # 결측 포함 샘플 제거를 위해 마스크를 같이 사용
    mask = ~np.isnan(values_x).any(axis=1) & ~np.isnan(values_y).any(axis=1)

    n = len(df)
    for t in range(Tin, n - Tout):
        # 입력 구간/출력 구간 모두 결측 없어야 함
        if mask[t - Tin:t].all() and mask[t:t + Tout].all():
            X_list.append(values_x[t - Tin:t, :])
            y_list.append(values_y[t:t + Tout, :])

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, Tin, len(x_cols)), dtype=np.float32)
    y = np.stack(y_list, axis=0) if y_list else np.empty((0, Tout, 1), dtype=np.float32)
    if len(X) == 0:
        raise ValueError("No training samples created (too many NaNs or too short period).")
    return X, y


class Seq2SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, Tin, F)
        self.y = torch.from_numpy(y)  # (N, Tout, 1)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


# -------------------------
# Seq2Seq 모델(Encoder-Decoder LSTM)
# -------------------------
class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (B, Tin, F)
        _, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h, c):
        # x: (B, 1, input_size)
        out, (h, c) = self.lstm(x, (h, c))
        y = self.fc(out[:, -1, :])  # (B, 1)
        return y, h, c


class Seq2Seq(nn.Module):
    def __init__(self, x_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.encoder = Encoder(x_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size=1, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, X, y_teacher=None, teacher_forcing: float = 0.5):
        """
        X: (B, Tin, F)
        y_teacher: (B, Tout, 1) (훈련 시 정답 시퀀스)
        return: y_hat (B, Tout, 1)
        """
        B = X.size(0)
        h, c = self.encoder(X)

        # decoder 첫 입력: 마지막 관측 수위(입력 X의 마지막 타임스텝에서 수위 채널을 사용)
        # 여기서는 입력 특성 중 '수위'가 마지막 컬럼이라고 가정하지 않기 위해,
        # 학습 전에 X 구성 시 [00060,00065]로 넣는 것을 전제로 마지막 feature를 사용.
        last_stage = X[:, -1, -1].unsqueeze(1)  # (B,1)
        dec_in = last_stage.unsqueeze(1)        # (B,1,1)

        Tout = y_teacher.size(1) if y_teacher is not None else 24
        outs = []

        for t in range(Tout):
            y_pred, h, c = self.decoder(dec_in, h, c)  # (B,1)
            outs.append(y_pred.unsqueeze(1))           # (B,1,1)

            use_tf = (y_teacher is not None) and (np.random.rand() < teacher_forcing)
            next_in = y_teacher[:, t, :] if use_tf else y_pred  # (B,1)
            dec_in = next_in.unsqueeze(1)                       # (B,1,1)

        return torch.cat(outs, dim=1)  # (B,Tout,1)


# -------------------------
# 학습/평가
# -------------------------
def fit_seq2seq(X: np.ndarray, y: np.ndarray, epochs=10, batch_size=64, teacher_forcing=0.5):
    # time-order split
    n = len(X)
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    # feature-wise 정규화(X만). y(수위)는 별도 스케일(수위 채널 기준)로 하는 게 보통 안정적
    # 여기서는:
    # - X는 채널별 표준화
    # - y는 "수위 스케일"(= X의 수위 채널 통계)로 표준화
    mu_x = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    sd_x = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-6

    # 수위 채널은 X의 마지막 feature(00065)로 가정
    stage_mu = mu_x[-1]
    stage_sd = sd_x[-1]

    X_tr_n = (X_tr - mu_x) / sd_x
    X_te_n = (X_te - mu_x) / sd_x
    y_tr_n = (y_tr - stage_mu) / stage_sd
    y_te_n = (y_te - stage_mu) / stage_sd

    tr_loader = DataLoader(Seq2SeqDataset(X_tr_n, y_tr_n), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(Seq2SeqDataset(X_te_n, y_te_n), batch_size=batch_size, shuffle=False)

    model = Seq2Seq(x_size=X.shape[-1], hidden_size=64, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in tr_loader:
            yhat = model(xb, y_teacher=yb, teacher_forcing=teacher_forcing)
            loss = loss_fn(yhat, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep:02d} | train_mse={float(np.mean(losses)):.6f}")

    # 평가: 전체 horizon에 대해 MAE/RMSE(원 단위 복원)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            yhat = model(xb, y_teacher=None, teacher_forcing=0.0)  # inference
            preds.append(yhat.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0) * stage_sd + stage_mu
    trues = np.concatenate(trues, axis=0) * stage_sd + stage_mu

    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))

    stats = {
        "mae_all_steps": mae,
        "rmse_all_steps": rmse,
        "mu_x": mu_x.tolist(),
        "sd_x": sd_x.tolist(),
        "stage_mu": float(stage_mu),
        "stage_sd": float(stage_sd),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
    return model, stats


def predict_next_6h(model: Seq2Seq, X_last: np.ndarray, mu_x, sd_x, stage_mu, stage_sd) -> np.ndarray:
    """
    X_last: (Tin, F) 원 단위
    return: (Tout,) 원 단위 수위 예측 시퀀스
    """
    x = (X_last.astype(np.float32) - mu_x) / (sd_x + 1e-6)
    x = torch.from_numpy(x[None, :, :])  # (1,Tin,F)
    with torch.no_grad():
        yhat_n = model(x, y_teacher=None, teacher_forcing=0.0).cpu().numpy()[0, :, 0]
    return yhat_n * stage_sd + stage_mu


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="10109000")
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--window_hours", type=float, default=24.0)
    ap.add_argument("--horizon_hours", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--teacher_forcing", type=float, default=0.5)
    ap.add_argument("--max_interp_gap_steps", type=int, default=2)
    args = ap.parse_args()

    # 00060(유량) + 00065(수위)
    df = await mcp_fetch_iv_multi(args.site, args.start, args.end, ["00060", "00065"])
    print(f"Fetched raw points: {len(df)} | columns={list(df.columns)}")

    df_u, freq = normalize_to_grid(df, target_freq=None, candidates_min=(5,15,60), max_interp_gap_steps=args.max_interp_gap_steps)
    print(f"Normalized freq={freq} | points={len(df_u)} | NaN rows={int(df_u.isna().any(axis=1).sum())}")

    X, y = make_seq2seq_supervised(
        df_u, freq=freq,
        window_hours=args.window_hours,
        horizon_hours=args.horizon_hours,
        x_cols=("00060","00065"),
        y_col="00065",
    )
    Tout = y.shape[1]
    print(f"Samples={len(X)} | Tin={X.shape[1]} | F={X.shape[2]} | Tout={Tout} (covers {args.horizon_hours}h)")

    model, stats = fit_seq2seq(
        X, y,
        epochs=args.epochs,
        batch_size=64,
        teacher_forcing=args.teacher_forcing,
    )
    print("Metrics:", {k:v for k,v in stats.items() if k.startswith("mae") or k.startswith("rmse") or k.startswith("n_")})

    # 마지막 입력 윈도우로 다음 6시간(여러 스텝) 예측
    Tin = X.shape[1]
    X_last = df_u[["00060","00065"]].values.astype(np.float32)[-Tin:, :]
    yhat = predict_next_6h(
        model,
        X_last,
        mu_x=np.array(stats["mu_x"], dtype=np.float32),
        sd_x=np.array(stats["sd_x"], dtype=np.float32),
        stage_mu=stats["stage_mu"],
        stage_sd=stats["stage_sd"],
    )
    # 예측 타임스탬프 생성
    t0 = df_u.index[-1]
    future_idx = pd.date_range(t0 + freq, periods=len(yhat), freq=freq, tz=df_u.index.tz)
    pred_series = pd.Series(yhat, index=future_idx, name="stage_pred_00065")
    print("\nNext horizon predictions (head):")
    print(pred_series.head(10).to_string())
    plot_30d_with_forecast(df_u, pred_series, out_html="usgs_30d_forecast.html")

if __name__ == "__main__":
    anyio.run(main)