# 실행 명령어: python -m streamlit run NIER_data_query.py
# streamlit 버전: 1.24.1 app 사용해서 물환경정보시스템 자료 조회

import requests
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

# =========================
# App Config
# =========================
st.set_page_config(page_title="물환경수질측정망 자료 조회", layout="wide")
st.title("물환경수질측정망 자료 조회")
st.caption("지점 검색/다중선택 → API 조회 → 결과 테이블/그래프 → CSV 저장")

API_URL = "http://apis.data.go.kr/1480523/WaterQualityService/getWaterMeasuringList"

PARAM_MAP = {"BOD": "itemBod", "COD": "itemCod", "TN": "itemTn", "TP": "itemTp"}

UPPER_TO_STD = {
    "PT_NO": "ptNo",
    "PT_NM": "ptNm",
    "ADDR": "addr",
    "ORG_NM": "orgNm",
    "WMYR": "wmyr",
    "WMOD": "wmod",
    "WMCYMD": "wmcymd",
    "ROWNO": "rowno",
    "ITEM_BOD": "itemBod",
    "ITEM_COD": "itemCod",
    "ITEM_TN": "itemTn",
    "ITEM_TP": "itemTp",
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

    stations_path = st.text_input("stations.csv 경로", value="stations.csv")

    st.subheader("기간(월)")
    c1, c2 = st.columns(2)
    with c1:
        start_yyyymm = st.text_input("시작 (YYYY-MM)", value="2011-01")
    with c2:
        end_yyyymm = st.text_input("끝 (YYYY-MM)", value="2011-12")

    st.subheader("표출 항목")
    metric = st.selectbox("BOD/COD/TN/TP", list(PARAM_MAP.keys()), index=0)

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
candidates["label"] = candidates["지점명"] + "  (" + candidates["지점코드"] + ")"

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("지점 선택(다중 선택 가능)")
    selected_labels = st.multiselect("지점", options=candidates["label"].tolist(), default=[])
    label_to_code = dict(zip(candidates["label"], candidates["지점코드"]))
    selected_codes = [label_to_code[x] for x in selected_labels] if selected_labels else []

    st.write(f"선택된 지점 수: **{len(selected_codes)}**")
    if selected_codes:
        st.code(", ".join(selected_codes), language="text")

with right:
    st.subheader("조회 결과")

    metric_col = PARAM_MAP[metric]

    run = st.button("조회 실행", type="primary", disabled=(not service_key or len(selected_codes) == 0))

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

        # 안전 정렬: ptNo -> date -> wmyr/wmod -> rowno
        df_sorted = safe_sort(df, ["ptNo", "date", "wmyr", "wmod", "rowno"])

        # 테이블 표시
        st.markdown("#### 결과 테이블")
        show_cols = [c for c in ["ptNo", "ptNm", "addr", "wmyr", "wmod", "wmcymd", "itemBod", "itemCod", "itemTn", "itemTp"] if c in df_sorted.columns]
        st.dataframe(df_sorted[show_cols], use_container_width=True, height=340)

        # CSV 저장(다운로드)
        st.markdown("#### CSV 저장")
        # 엑셀에서 한글 깨짐을 줄이기 위해 utf-8-sig 사용
        csv_bytes = df_sorted.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        filename = f"water_quality_{'_'.join(selected_codes[:3])}{'_etc' if len(selected_codes) > 3 else ''}_{start_yyyymm}_to_{end_yyyymm}.csv"
        st.download_button(
            label="조회 결과 CSV 다운로드",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv"
        )

        # 그래프 표시
        st.markdown("#### 월별 시계열 그래프")
        if "date" not in df_sorted.columns or df_sorted["date"].notna().sum() == 0:
            st.error("date 컬럼 생성에 실패하여 그래프를 그릴 수 없습니다. (기간/데이터를 확인하세요)")
            st.stop()

        if metric_col not in df_sorted.columns:
            st.error(f"선택한 항목 컬럼({metric_col})이 응답에 없습니다.")
            st.stop()

        g = (
            df_sorted.dropna(subset=["date"])
                    .groupby(["ptNo", "ptNm", "date"], as_index=False)[metric_col]
                    .mean()
        )

        chart = (
            alt.Chart(g)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="월"),
                y=alt.Y(f"{metric_col}:Q", title=f"{metric}"),
                color=alt.Color("ptNm:N", title="지점"),
                tooltip=[
                    alt.Tooltip("ptNm:N", title="지점"),
                    alt.Tooltip("ptNo:N", title="코드"),
                    alt.Tooltip("date:T", title="월"),
                    alt.Tooltip(f"{metric_col}:Q", title=metric),
                ],
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption(
    "API는 WaterQualityService/getWaterMeasuringList를 사용하며 serviceKey/ptNoList/wmyrList/wmodList/resultType 등의 파라미터를 사용합니다. "
    "[Source](https://www.genspark.ai/api/files/s/Xh06BpN4)"
)
st.caption("지점 목록은 stations.csv의 지점코드/지점명으로 구성됩니다. [Source](https://www.genspark.ai/api/files/s/70Ljcf6b)")
