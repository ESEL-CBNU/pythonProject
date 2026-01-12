# server_usgs_stage_mcp.py
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from dataretrieval import nwis
import numpy as np
import pandas as pd

mcp = FastMCP(
    "usgs-water-multivar",
    json_response=True,
    instructions="USGS NWIS에서 관측소 검색 및 IV(00060,00065) 시계열을 제공하는 MCP 서버",
)

_SENTINELS = {-999999.0, -999999}

def _clean_iv_df(df: pd.DataFrame) -> pd.DataFrame:
    # index 정리(중복/정렬)
    df = df.copy()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    # 센티넬/이상치 처리(센티넬만)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace(list(_SENTINELS), np.nan)
    return df

@mcp.tool()
def usgs_what_sites(
    stateCd: str,
    parameterCd: str = "00065",
    siteType: str = "ST",
    period: str = "P30D",
    max_sites: int = 20,
) -> dict:
    df, md = nwis.what_sites(
        stateCd=stateCd,
        parameterCd=parameterCd,
        siteType=siteType,
        period=period,
        siteOutput="basic",
    )
    keep_cols = [c for c in ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "site_tp_cd", "huc_cd"] if c in df.columns]
    df = df.reset_index(drop=True)
    if keep_cols:
        df = df[keep_cols]
    df = df.head(max_sites)

    return {
        "query": {"stateCd": stateCd, "parameterCd": parameterCd, "siteType": siteType, "period": period},
        "count": int(len(df)),
        "sites": df.to_dict(orient="records"),
        "request_url": getattr(md, "url", None),
    }

@mcp.tool()
def usgs_get_iv_multi(
    site_no: str,
    start: str,
    end: str,
    parameterCds: list[str] = ["00060", "00065"],
) -> dict:
    """
    IV(instantaneous values)에서 여러 파라미터(예: 00060 유량, 00065 수위)를 동시에 가져온다.
    """
    df, md = nwis.get_iv(
        sites=site_no,
        parameterCd=parameterCds,  # 여러 코드 리스트 지원 :contentReference[oaicite:3]{index=3}
        start=start,
        end=end,
    )

    # 컬럼 매핑: 보통 '00060','00065' 그대로 들어옴
    cols_found = []
    for p in parameterCds:
        if p in df.columns:
            cols_found.append(p)
        else:
            # 혹시 컬럼명이 변형되면 후보 탐색
            cand = [c for c in df.columns if str(c).endswith(p)]
            if cand:
                cols_found.append(cand[0])

    if not cols_found:
        raise ValueError(f"No requested parameter columns found. Columns={list(df.columns)}")

    df = df[cols_found]
    df = _clean_iv_df(df)

    # all-NaN row 제거
    df = df.dropna(how="all")

    # JSON points로 변환 (t, 00060, 00065 ...)
    points = []
    for t, row in df.iterrows():
        rec = {"t": t.isoformat()}
        for c in cols_found:
            v = row[c]
            rec[c] = None if pd.isna(v) else float(v)
        points.append(rec)

    return {
        "site_no": site_no,
        "parameterCds": parameterCds,
        "columns": cols_found,
        "n_points": int(len(points)),
        "points": points,
        "request_url": getattr(md, "url", None),
    }

if __name__ == "__main__":
    mcp.run()
