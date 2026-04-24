# 실행 명령어: python -m streamlit run NIER_data_query.py
# streamlit 버전: 1.24.1 app 사용해서 물환경정보시스템 자료 조회

import requests
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from pathlib import Path

# =========================
# App Config
# =========================
st.set_page_config(page_title="물환경수질측정망 자료 조회", layout="wide")
st.title("물환경수질측정망 자료 조회")
st.caption("지점 선택 조회 → API 조회 → 결과 테이블/시계열 그래프 → CSV 저장")

API_URL = "http://apis.data.go.kr/1480523/WaterQualityService/getWaterMeasuringList"
DEFAULT_STATIONS_PATH = str(Path(__file__).resolve().parent / "stations.csv")

PARAM_MAP = {
    "BOD": "itemBod",
    "COD": "itemCod",
    "TN": "itemTn",
    "TP": "itemTp",
    "수온": "itemTemp",
    "SS": "itemSs",
}
DEPTH_COL_CANDIDATES = ["wmdep", "itemDep", "depth", "dep", "layer"]

UPPER_TO_STD = {
    "PT_NO": "ptNo",
    "PT_NM": "ptNm",
    "ADDR": "addr",
    "ORG_NM": "orgNm",
    "WMYR": "wmyr",
    "WMOD": "wmod",
    "WMCYMD": "wmcymd",
    "WMDEP": "wmdep",
    "ITEM_DEP": "itemDep",
    "DEPTH": "depth",
    "DEP": "dep",
    "LAYER": "layer",
    "ROWNO": "rowno",
    "ITEM_BOD": "itemBod",
    "ITEM_COD": "itemCod",
    "ITEM_SS": "itemSs",
    "ITEM_TN": "itemTn",
    "ITEM_TP": "itemTp",
    "ITEM_TEMP": "itemTemp",
    "ITEM_WT": "itemTemp",
    "ITEM_WTEMP": "itemTemp",
}

# =========================
# Helpers
# =========================
@st.cache_data
def load_stations(csv_path: str) -> pd.DataFrame:
    # stations.csv: 지점코드,지점명 [Source](https://www.genspark.ai/api/files/s/70Ljcf6b)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")

    df.columns = [str(c).strip() for c in df.columns]
    df["지점코드"] = df["지점코드"].astype(str).str.strip()
    df["지점명"] = df["지점명"].astype(str).str.strip()
    return df


def month_range(start_yyyymm: str, end_yyyymm: str):
    s = datetime.strptime(start_yyyymm, "%Y-%m")
    e = datetime.strptime(end_yyyymm, "%Y-%m")
    if s > e:
        s, e = e, s

    years, months = set(), set()
    cur = s
    while cur <= e:
        years.add(cur.year)
        months.add(cur.month)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    return sorted(years), sorted(months)


def deep_get(d, keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def extract_items(api_json: dict) -> list[dict]:
    """
    지원 응답 구조:
    A) {"getWaterMeasuringList": {"header":..., "item":[...], ...}}
    B) {"response":{"body":{"items":{"item":[...]}}}}
    """
    g = api_json.get("getWaterMeasuringList")
    if isinstance(g, dict):
        items = g.get("item")
        if isinstance(items, list):
            return items

    items = deep_get(api_json, ["response", "body", "items", "item"])
    if isinstance(items, list):
        return items

    items = deep_get(api_json, ["response", "body", "items"])
    if isinstance(items, list):
        return items

    return []


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {c: UPPER_TO_STD[c] for c in df.columns if c in UPPER_TO_STD}
    return df.rename(columns=rename_map)


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    date 자동 생성/복구:
    - wmyr/wmod -> date
    - 실패 시 wmod 0패딩 후 재시도
    - 그래도 실패 시 wmcymd(yyyy.mm.dd) -> 월단위로 내림
    """
    if df.empty:
        return df

    df = normalize_columns(df)

    for c in ["wmyr", "wmod", "wmcymd"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "date" not in df.columns and "wmyr" in df.columns and "wmod" in df.columns:
        df["date"] = pd.to_datetime(df["wmyr"] + "-" + df["wmod"] + "-01", errors="coerce")

    if "date" in df.columns and df["date"].notna().sum() == 0 and "wmyr" in df.columns and "wmod" in df.columns:
        wmod2 = df["wmod"].astype(str).str.zfill(2)
        df["date"] = pd.to_datetime(df["wmyr"] + "-" + wmod2 + "-01", errors="coerce")

    if ("date" not in df.columns) or (df["date"].notna().sum() == 0):
        if "wmcymd" in df.columns:
            df["date"] = pd.to_datetime(df["wmcymd"].str.replace(".", "-", regex=False), errors="coerce")
            if df["date"].notna().any():
                df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in set(PARAM_MAP.values()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_sort(df: pd.DataFrame, preferred_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        return df
    return df.sort_values(cols)


def find_depth_column(df: pd.DataFrame) -> str | None:
    for col in DEPTH_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def build_params(service_key: str, pt_codes: list[str], years: list[int], months: list[int], num_of_rows: int, page_no: int):
    # 가이드: serviceKey(필수), ptNoList/wmyrList/wmodList/resultType 등 [Source](https://www.genspark.ai/api/files/s/Xh06BpN4)
    return {
        "serviceKey": service_key,
        "resultType": "json",
        "numOfRows": str(num_of_rows),
        "pageNo": str(page_no),
        "ptNoList": ",".join(pt_codes),
        "wmyrList": ",".join(str(y) for y in years),
        "wmodList": ",".join(f"{m:02d}" for m in months),  # 01~09 주의 [Source](https://www.genspark.ai/api/files/s/Xh06BpN4)
    }


@st.cache_data(show_spinner=False)
def fetch_data(service_key: str, pt_codes: list[str], years: list[int], months: list[int], num_of_rows: int = 2000, page_no: int = 1) -> pd.DataFrame:
    params = build_params(service_key, pt_codes, years, months, num_of_rows, page_no)
    r = requests.get(API_URL, params=params, timeout=40)
    r.raise_for_status()

    data = r.json()
    items = extract_items(data)
    df = pd.DataFrame(items)

    df = normalize_columns(df)
    df = ensure_date_column(df)
    df = coerce_numeric(df)

    return df


# =========================
# Sidebar inputs
# =========================
with st.sidebar:
    st.header("조회 조건")

    service_key = st.text_input("serviceKey (data.go.kr 발급키)", type="password")

    stations_path = st.text_input("stations.csv 경로", value=DEFAULT_STATIONS_PATH)

    st.subheader("기간(월)")
    c1, c2 = st.columns(2)
    with c1:
        start_yyyymm = st.text_input("시작 (YYYY-MM)", value="2011-01")
    with c2:
        end_yyyymm = st.text_input("끝 (YYYY-MM)", value="2011-12")

    st.subheader("지점 검색")
    query = st.text_input("지점명/코드 검색어", value="")
    max_candidates = st.slider("검색 결과 최대 표시 수", 50, 2000, 300, step=50)

# =========================
# Load stations and choose
# =========================
try:
    stations_df = load_stations(stations_path)
except Exception as e:
    st.error(f"stations.csv를 읽을 수 없습니다: {e}")
    st.stop()

if query.strip():
    mask = (
        stations_df["지점명"].str.contains(query, case=False, na=False)
        | stations_df["지점코드"].str.contains(query, case=False, na=False)
    )
    candidates = stations_df[mask].copy()
else:
    candidates = stations_df.copy()

candidates = candidates.head(max_candidates).reset_index(drop=True)
candidates["label"] = candidates["지점명"] + " (" + candidates["지점코드"] + ")"

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("지점")
    if candidates.empty:
        st.warning("검색 조건에 맞는 지점이 없습니다.")
        selected_codes = []
    else:
        options = candidates["label"].tolist()
        default_index = 0
        default_match = candidates.index[candidates["지점코드"] == "3008B30"].tolist()
        if default_match:
            default_index = int(default_match[0])
        selected_label = st.selectbox("조회 지점 선택", options=options, index=default_index)
        selected_code = candidates.loc[candidates["label"] == selected_label, "지점코드"].iloc[0]
        selected_codes = [selected_code]
        st.write(f"선택 지점: **{selected_label}**")

with right:
    st.subheader("조회 결과")

    run = st.button("조회 실행", type="primary", disabled=(not service_key or len(selected_codes) == 0))

    # ── 조회 실행: session_state에 데이터 저장 ──────────────────────────
    if run:
        try:
            years, months = month_range(start_yyyymm, end_yyyymm)
        except Exception:
            st.error("기간 형식이 올바르지 않습니다. YYYY-MM 예: 2023-01")
            st.stop()

        with st.spinner("조회 중..."):
            try:
                df = fetch_data(service_key, selected_codes, years, months, num_of_rows=2000, page_no=1)
            except Exception as e:
                st.error(f"API 호출 실패: {e}")
                st.stop()

        if df.empty:
            st.warning("조회 결과가 없습니다. (지점/기간/키를 확인하세요)")
            st.stop()

        df_sorted = safe_sort(df, ["ptNo", "date", "wmyr", "wmod", "rowno"])
        depth_col = find_depth_column(df_sorted)

        # 수심 컬럼 → 동일 날짜·지점 내 순위로 상/중/하 라벨링
        if depth_col and depth_col in df_sorted.columns:
            df_sorted["_dep_num"] = pd.to_numeric(df_sorted[depth_col], errors="coerce")
            group_keys = [c for c in ["ptNo", "date"] if c in df_sorted.columns]
            df_sorted["_dep_rank"] = (
                df_sorted.groupby(group_keys)["_dep_num"]
                .rank(method="dense", ascending=True)
            )
            df_sorted["_dep_max"] = (
                df_sorted.groupby(group_keys)["_dep_rank"]
                .transform("max")
            )

            def _to_label(row):
                r, m = row["_dep_rank"], row["_dep_max"]
                if pd.isna(r):
                    return "미기재"
                if m == 1:
                    return "상"
                elif m == 2:
                    return "상" if r == 1 else "하"
                else:  # 3개 이상
                    if r == 1:
                        return "상"
                    elif r == m:
                        return "하"
                    else:
                        return "중"

            df_sorted[depth_col] = df_sorted.apply(_to_label, axis=1)
            df_sorted.drop(columns=["_dep_num", "_dep_rank", "_dep_max"], inplace=True)

        st.session_state["df"] = df_sorted
        st.session_state["depth_col"] = depth_col
        st.session_state["query_key"] = f"{'_'.join(selected_codes)}_{start_yyyymm}_{end_yyyymm}"

    # ── 저장된 데이터가 있으면 필터 UI + 결과 표시 ───────────────────────
    if "df" in st.session_state:
        df_sorted = st.session_state["df"]
        depth_col = st.session_state["depth_col"]

        # ── 필터 선택  ────────────────────────────────────────────────────
        st.markdown("#### 표시 필터")

        # 지점명 다중 선택
        avail_pts = df_sorted[["ptNo", "ptNm"]].drop_duplicates()
        pt_options = (avail_pts["ptNm"] + " (" + avail_pts["ptNo"] + ")").tolist()
        selected_pts = st.multiselect(
            "지점 선택 (다중)", options=pt_options, default=pt_options,
        )
        selected_pt_nos = [
            avail_pts.loc[avail_pts["ptNm"] + " (" + avail_pts["ptNo"] + ")" == lbl, "ptNo"].iloc[0]
            for lbl in selected_pts
            if lbl in (avail_pts["ptNm"] + " (" + avail_pts["ptNo"] + ")").values
        ]

        c_metric, c_depth = st.columns(2)

        # 항목 다중 선택
        metric_options = {label: col for label, col in PARAM_MAP.items() if col in df_sorted.columns}
        with c_metric:
            if metric_options:
                selected_metrics = st.multiselect(
                    "항목 선택 (다중)", options=list(metric_options.keys()),
                    default=list(metric_options.keys())[:1],
                )
            else:
                st.warning("수질 항목 컬럼이 응답에 없습니다.")
                selected_metrics = []

        # 수심 다중 선택
        with c_depth:
            if depth_col and depth_col in df_sorted.columns:
                depth_values = sorted(df_sorted[depth_col].dropna().unique().tolist())
                selected_depths = st.multiselect(
                    "수심 선택 (다중)", options=depth_values,
                    default=depth_values,
                )
            else:
                selected_depths = []
                st.info("응답에 수심 컬럼이 없습니다.")

        # ── 결과 테이블 ──────────────────────────────────────────────────
        st.markdown("#### 결과 테이블")
        show_cols = [c for c in ["ptNo", "ptNm", "wmyr", "wmod", "wmcymd", depth_col,
                                  "itemBod", "itemCod", "itemTemp", "itemSs", "itemTn", "itemTp"]
                     if c and c in df_sorted.columns]
        st.dataframe(df_sorted[show_cols], use_container_width=True, height=300)

        # ── CSV 다운로드 ─────────────────────────────────────────────────
        st.markdown("#### CSV 저장")
        csv_bytes = df_sorted.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        filename = f"water_quality_{st.session_state['query_key']}.csv"
        st.download_button(
            label="조회 결과 CSV 다운로드",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
        )

        # ── 시계열 그래프 ────────────────────────────────────────────────
        st.markdown("#### 월별 시계열 그래프")

        if not selected_metrics:
            st.info("항목을 1개 이상 선택하세요.")
            st.stop()
        if "date" not in df_sorted.columns or df_sorted["date"].notna().sum() == 0:
            st.error("date 컬럼이 없어 그래프를 그릴 수 없습니다.")
            st.stop()

        # 필터 적용
        filtered = df_sorted.dropna(subset=["date"]).copy()
        if selected_pt_nos:
            filtered = filtered[filtered["ptNo"].isin(selected_pt_nos)]
        if depth_col and selected_depths:
            filtered = filtered[filtered[depth_col].isin(selected_depths)]

        # 선택된 항목들을 long 형태로 변환
        value_cols = [metric_options[m] for m in selected_metrics if m in metric_options]
        group_by = ["date"]
        if depth_col and selected_depths and depth_col in filtered.columns:
            group_by.append(depth_col)

        g = filtered.groupby(group_by, as_index=False)[value_cols].mean()
        g_long = (
            g.melt(id_vars=group_by, value_vars=value_cols, var_name="item_col", value_name="value")
            .dropna(subset=["value"])
        )
        # 컬럼명 → 라벨명 역매핑
        col_to_label = {v: k for k, v in PARAM_MAP.items()}
        g_long["항목"] = g_long["item_col"].map(col_to_label).fillna(g_long["item_col"])

        # 범례 컬럼: 수심이 있으면 "항목 @ 수심", 없으면 "항목"
        if depth_col and depth_col in g_long.columns and len(selected_depths) > 1:
            g_long["legend"] = g_long["항목"] + " @ " + g_long[depth_col].astype(str)
        else:
            g_long["legend"] = g_long["항목"]

        tooltip_extra = ([alt.Tooltip(f"{depth_col}:N", title="수심")]
                         if depth_col and depth_col in g_long.columns else [])

        if g_long.empty:
            st.warning("선택한 조건에 해당하는 값이 없습니다.")
            st.stop()

        chart = (
            alt.Chart(g_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="월"),
                y=alt.Y("value:Q", title="수질값"),
                color=alt.Color("legend:N", title="항목"),
                tooltip=[alt.Tooltip("date:T", title="월"),
                         alt.Tooltip("항목:N", title="항목"),
                         *tooltip_extra,
                         alt.Tooltip("value:Q", title="값", format=".3f")],
            )
            .properties(height=400)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption(
    "API는 WaterQualityService/getWaterMeasuringList를 사용하며 serviceKey/ptNoList/wmyrList/wmodList/resultType 등의 파라미터를 사용합니다. "
    "[Source](https://www.genspark.ai/api/files/s/Xh06BpN4)"
)
st.caption("지점 목록은 stations.csv의 지점코드/지점명으로 구성됩니다. [Source](https://www.genspark.ai/api/files/s/70Ljcf6b)")
