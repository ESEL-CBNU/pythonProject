from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import anyio
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

from mcp.server.fastmcp import FastMCP

# MCP server: Tools/Resources/Prompts 제공
mcp = FastMCP("USGS Stage Forecast (Seq2Seq)", json_response=True)


# -----------------------------
# USGS IV fetch (waterservices)
# -----------------------------
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


def _fetch_usgs_iv(site: str, start_dt: str, end_dt: str, parameter_cds: List[str]) -> pd.DataFrame:
    """
    USGS Instantaneous Values service(JSON)에서 site+parameterCd들을 읽어
    datetime index의 DataFrame으로 반환.
    """
    params = {
        "format": "json",
        "sites": site,
        "startDT": start_dt,
        "endDT": end_dt,
        "parameterCd": ",".join(parameter_cds),
        "siteStatus": "all",
    }
    r = requests.get(USGS_IV_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    ts_list = payload.get("value", {}).get("timeSeries", [])
    if not ts_list:
        raise ValueError(f"No timeSeries returned. site={site}, params={parameter_cds}")

    series_map: Dict[str, pd.Series] = {}
    for ts in ts_list:
        var = ts.get("variable", {})
        code = var.get("variableCode", [{}])[0].get("value")
        values = ts.get("values", [{}])[0].get("value", [])
        if not code or not values:
            continue

        dt = [v["dateTime"] for v in values]
        vv = [v.get("value", None) for v in values]
        idx = pd.to_datetime(dt, utc=True, errors="coerce")
        ser = pd.to_numeric(pd.Series(vv, index=idx), errors="coerce")
        ser.name = code
        series_map[code] = ser

    if not series_map:
        raise ValueError("Parsed empty series_map from USGS response")

    df = pd.concat(series_map.values(), axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _infer_and_normalize_freq(df: pd.DataFrame, candidates_min=(5, 15, 60)) -> Tuple[pd.DataFrame, pd.Timedelta]:
    """
    관측 간격이 5/15/60분 중 어느 것과 가장 가까운지 추정하고,
    그 간격으로 reindex/resample 해서 NaN을 시간 보간으로 메움.
    """
    if len(df) < 10:
        raise ValueError("Not enough points to infer frequency")

    diffs = df.index.to_series().diff().dropna()
    med = diffs.median()
    med_min = med.total_seconds() / 60.0

    best = min(candidates_min, key=lambda x: abs(x - med_min))
    freq = pd.Timedelta(minutes=int(best))

    full_idx = pd.date_range(df.index.min().floor("min"), df.index.max().ceil("min"), freq=freq, tz="UTC")
    df_u = df.reindex(full_idx)

    # time interpolation + forward/back fill for edges
    df_u = df_u.interpolate(method="time", limit_direction="both")
    df_u = df_u.ffill().bfill()

    return df_u, freq


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform_1d(self, y: np.ndarray, idx: int) -> np.ndarray:
        return y * self.std[idx] + self.mean[idx]


# -----------------------------
# Seq2Seq LSTM (multivar -> stage)
# -----------------------------
class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.enc = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.dec = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, teacher_forcing: float = 0.5):
        # x: (B, Tin, F), y: (B, Tout) [stage target in normalized space]
        B = x.size(0)
        _, (h, c) = self.enc(x)

        # decoder input starts with last observed stage (assume stage is last feature in x)
        prev = x[:, -1, -1].view(B, 1, 1)  # (B,1,1)
        Tout = y.size(1) if y is not None else 1

        outs = []
        for t in range(Tout):
            dec_out, (h, c) = self.dec(prev, (h, c))
            step = self.proj(dec_out[:, -1, :])  # (B,1)
            outs.append(step)

            if y is not None and np.random.rand() < teacher_forcing:
                prev = y[:, t].view(B, 1, 1)
            else:
                prev = step.view(B, 1, 1)

        return torch.cat(outs, dim=1)  # (B, Tout)


def _make_supervised_arrays(df_u: pd.DataFrame, tin: int, tout: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (N, tin, F), y: (N, tout) where y is stage(00065) only (assumed last column)
    """
    arr = df_u.values.astype("float32")
    F = arr.shape[1]
    stage_idx = F - 1

    X, Y = [], []
    for i in range(tin, len(arr) - tout):
        X.append(arr[i - tin:i, :])
        Y.append(arr[i:i + tout, stage_idx])
    return np.stack(X), np.stack(Y)


def _train_or_load_model(
    X: np.ndarray,
    Y: np.ndarray,
    model_path: str,
    epochs: int = 10,
    lr: float = 1e-3,
    hidden: int = 64,
) -> Tuple[Seq2SeqLSTM, Dict]:
    device = "cpu"
    n_features = X.shape[2]
    tout = Y.shape[1]

    model = Seq2SeqLSTM(n_features=n_features, hidden=hidden).to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        return model, {"loaded": True, "epochs": 0}

    # time-based split (last 20% test)
    n = len(X)
    n_train = int(n * 0.8)
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xte, Yte = X[n_train:], Y[n_train:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def batches(a, b, bs=64):
        for i in range(0, len(a), bs):
            yield a[i:i + bs], b[i:i + bs]

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in batches(Xtr, Ytr, bs=64):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            pred = model(xb_t, y=yb_t, teacher_forcing=0.5)
            loss = loss_fn(pred, yb_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # 간단 로그(Host에서 보기 좋게)
        print(f"epoch {ep:02d} | train_mse={float(np.mean(losses)):.6f}")

    # eval metrics
    model.eval()
    with torch.no_grad():
        yhat = model(torch.from_numpy(Xte).to(device), y=torch.from_numpy(Yte).to(device), teacher_forcing=0.0).cpu().numpy()

    mae = float(np.mean(np.abs(yhat - Yte)))
    rmse = float(np.sqrt(np.mean((yhat - Yte) ** 2)))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, model_path)

    return model, {"loaded": False, "epochs": epochs, "mae_all_steps": mae, "rmse_all_steps": rmse, "n_train": n_train, "n_test": len(Xte)}


def _plot_30d_with_forecast(df_u: pd.DataFrame, pred_stage: pd.Series, out_html: str) -> str:
    """
    Plotly: 30일 관측(수위+유량) + 6시간 수위 예측(다른 라인)
    hover(커서 숫자) + zoom 기본 지원
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df30 = df_u.last("30D").copy()
    t0 = df_u.index[-1].to_pydatetime()  # plotly timestamp 안정화

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # stage(00065) on left y
    fig.add_trace(
        go.Scatter(x=df30.index, y=df30["00065"], name="Stage (obs, 00065)", mode="lines"),
        secondary_y=False,
    )
    # discharge(00060) on right y
    fig.add_trace(
        go.Scatter(x=df30.index, y=df30["00060"], name="Discharge (obs, 00060)", mode="lines"),
        secondary_y=True,
    )
    # forecast stage
    fig.add_trace(
        go.Scatter(x=pred_stage.index, y=pred_stage.values, name="Stage forecast (+6h)", mode="lines+markers"),
        secondary_y=False,
    )

    # vertical line at forecast start
    fig.add_shape(
        type="line",
        x0=t0, x1=t0, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(dash="dot"),
    )

    fig.update_layout(
        title="USGS 30-day Obs (Stage+Discharge) with +6h Stage Forecast",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Stage (00065)", secondary_y=False)
    fig.update_yaxes(title_text="Discharge (00060)", secondary_y=True)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="inline")
    return out_html


# -----------------------------
# MCP Resources / Prompts
# -----------------------------
@mcp.resource("usgs-params://common")
def usgs_common_params() -> str:
    return (
        "Common USGS parameter codes:\n"
        "- 00060: Discharge (streamflow)\n"
        "- 00065: Gage height (stage)\n"
        "These are 5-digit parameter codes used by USGS Water Data services."
    )


@mcp.prompt()
def stage_forecast_prompt(site: str = "", days: int = 30, horizon_hours: int = 6) -> str:
    return (
        "You are a hydrology forecasting assistant.\n"
        "User wants a stage forecast using USGS IV data.\n"
        f"- site: {site or '[ask user for USGS site number]'}\n"
        f"- lookback days: {days}\n"
        f"- forecast horizon: {horizon_hours} hours (multi-step)\n"
        "Use the tool usgs_stage_forecast_report to fetch data, train/load model, forecast, and create an HTML plot.\n"
        "Then summarize results and present a small table of the forecast head plus the saved HTML path."
    )


# -----------------------------
# MCP Tools
# -----------------------------
@mcp.tool()
def usgs_search_sites(state: str, text: str = "", max_sites: int = 10) -> Dict:
    """
    간단 검색(USGS 'explore' 화면용 힌트 수준): 여기서는 API 호출 대신,
    사용자가 site_no를 이미 알고 있는 경우가 많아 최소 기능만 제공.
    실전에서는 state+text로 waterdata 검색 API를 붙이는 것을 권장.
    """
    return {
        "note": "This is a lightweight placeholder. Provide the site number directly if you know it.",
        "hint": "Try USGS Water Data 'Explore' and search by parameter codes 00060,00065.",
        "state": state,
        "text": text,
        "max_sites": max_sites,
    }


@mcp.tool()
def usgs_stage_forecast_report(
    site: str,
    start: str,
    end: str,
    window_hours: int = 24,
    horizon_hours: int = 6,
    epochs: int = 10,
    retrain: bool = False,
) -> Dict:
    """
    메인 툴:
    - USGS IV(00060+00065) fetch
    - 관측 간격 자동 정규화(5/15/60분)
    - 멀티변량 입력 -> stage(00065) 다중 step 예측(Seq2Seq LSTM)
    - 30일 관측 + 6시간 예측 Plotly HTML 생성
    """
    # 1) fetch
    df = _fetch_usgs_iv(site, start, end, ["00060", "00065"])
    cols = list(df.columns)

    # 2) normalize freq
    df_u, freq = _infer_and_normalize_freq(df)
    freq_min = int(freq.total_seconds() // 60)

    # 3) scaler
    Xraw = df_u.values.astype("float32")
    mean = Xraw.mean(axis=0)
    std = Xraw.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaler = Scaler(mean=mean, std=std)
    Xn = scaler.transform(Xraw)
    df_n = pd.DataFrame(Xn, index=df_u.index, columns=df_u.columns)

    # 4) seq lengths
    tin = int(round(window_hours * 60 / freq_min))
    tout = int(round(horizon_hours * 60 / freq_min))
    tin = max(tin, 4)
    tout = max(tout, 2)

    X, Y = _make_supervised_arrays(df_n, tin=tin, tout=tout)

    # 5) train/load
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"seq2seq_site{site}_f{freq_min}m_tin{tin}_tout{tout}.pt")
    if retrain and os.path.exists(model_path):
        os.remove(model_path)

    model, train_info = _train_or_load_model(X, Y, model_path=model_path, epochs=epochs)

    # 6) forecast next horizon using last window
    model.eval()
    last_x = df_n.values.astype("float32")[-tin:, :][None, ...]  # (1,tin,F)
    with torch.no_grad():
        yhat_n = model(torch.from_numpy(last_x), y=torch.zeros((1, tout)), teacher_forcing=0.0).cpu().numpy().reshape(-1)

    # inverse transform for stage (last column)
    yhat = scaler.inverse_transform_1d(yhat_n, idx=len(cols) - 1)

    t0 = df_u.index[-1]
    future_idx = pd.date_range(t0 + freq, periods=len(yhat), freq=freq, tz="UTC")
    pred_series = pd.Series(yhat, index=future_idx, name="stage_pred_00065")

    # 7) plot
    out_html = os.path.join("reports", f"usgs_{site}_30d_plus6h_{t0.strftime('%Y%m%dT%H%M%SZ')}.html")
    html_path = _plot_30d_with_forecast(df_u, pred_series, out_html=out_html)

    # 8) return structured output for LLM to format as table
    head_rows = []
    for ts, val in pred_series.head(24).items():
        head_rows.append({"time_utc": str(ts), "stage_pred_00065": float(val)})

    return {
        "site": site,
        "columns": cols,
        "normalized_freq_minutes": freq_min,
        "points": int(len(df_u)),
        "tin": tin,
        "tout": tout,
        "horizon_hours": horizon_hours,
        "train_info": train_info,
        "forecast_head_rows": head_rows,
        "html_report_path": html_path,
        "notes": [
            "Open the HTML report to see hover values and zoom/pan interactions.",
            "Stage forecast is plotted as a separate line with markers.",
        ],
    }


if __name__ == "__main__":
    # Cursor/VSCode는 stdio로 실행하는 구성이 가장 간단합니다.
    # MCP 표준도 stdio를 기본 지원으로 권장합니다.
    mcp.run(transport="stdio")
