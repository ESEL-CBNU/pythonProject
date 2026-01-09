import os
import re
import zipfile
import math
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from netCDF4 import Dataset
from datetime import datetime

# =========================================================
# 0) 사용자 설정
# =========================================================
nc_file   = r"c:\dev\pythonProject\aem3d\sheet_top_cyano.nc"
bath_file = r"c:\dev\pythonProject\aem3d\bath100_edm1_geo.xyz"
var_name  = "CYANO"

output_dir = "kml_output"
os.makedirs(output_dir, exist_ok=True)

# AEM3D (I,J) 인덱스가 1부터 시작하면 1, 0부터 시작하면 0
# (자동 선택하도록 아래 AUTO_INDEX_BASE=True 권장)
INDEX_BASE = 1
AUTO_INDEX_BASE = True

# (I, J, lon, lat)  — 교수님 제공 4개 기준점
GCP = [
    [45, 18, 127.480833, 36.477569],   # Point 1 (대청댐 본체)
    [193, 68, 127.650222, 36.349806],  # Point 2 (상류 옥천)
    [99, 81, 127.550906, 36.426133],   # Point 3 (중류 안내면)
    [160, 12, 127.477294, 36.370797]   # Point 4 (문의/현도)
]

# 결측/무효값 코드(파일에 따라 다를 수 있음)
INVALID_LEQ = -999

# percentiles 계산 시 샘플 최대 개수(메모리 절약)
PERCENTILE_SAMPLE_MAX = 400_000

# =========================================================
# 1) bath 파일 헤더에서 격자 정보 읽기 (x_rows, y_columns, x_grid, y_grid)
# =========================================================
def parse_bath_header(path: str) -> dict:
    vals = {}
    if (path is None) or (not os.path.exists(path)):
        return vals

    keys = {"x_rows", "y_columns", "x_grid", "y_grid"}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            m = re.match(r'^([+-]?\d+(\.\d+)?)\s+([A-Za-z_]+)', s)
            if m:
                num = float(m.group(1))
                key = m.group(3)
                if key in keys:
                    vals[key] = num
    return vals

bath = parse_bath_header(bath_file)

# =========================================================
# 2) 거리-도(deg) 변환(국지 근사)
# =========================================================
def meters_per_deg_lat(lat_deg: float) -> float:
    # 국지 근사: 1 deg latitude ~ 111.32 km 수준
    return 111_320.0

def meters_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))

# =========================================================
# 3) (I,J) -> (lon,lat) 변환 (origin + dx/dy 가정)
#    origin(lat0, lon0)는 (I=INDEX_BASE, J=INDEX_BASE) 셀 중심 좌표로 가정
# =========================================================
def predict_lonlat(lat0, lon0, I, J, index_base, dx_m, dy_m):
    i0 = I - index_base
    j0 = J - index_base

    lat = lat0 - (i0 * dx_m) / meters_per_deg_lat(lat0)  # I south(+) => lat 감소
    # lon 변환은 위도에 따라 약간 달라서 lat를 사용
    lon = lon0 + (j0 * dy_m) / meters_per_deg_lon(lat)
    return lon, lat

# =========================================================
# 4) GCP로부터 origin(lat0, lon0) 추정 (이상치 완화)
# =========================================================
def estimate_origin_from_gcp(gcp_list, index_base, dx_m, dy_m):
    est = []
    for I, J, lon_p, lat_p in gcp_list:
        i0 = I - index_base
        j0 = J - index_base

        # (I,J)에서 origin으로 역산:
        # lat_p = lat0 - i0*dx/mperlat  => lat0 = lat_p + i0*dx/mperlat
        lat0 = lat_p + (i0 * dx_m) / meters_per_deg_lat(lat_p)

        # lon_p = lon0 + j0*dy/mperlon => lon0 = lon_p - j0*dy/mperlon
        lon0 = lon_p - (j0 * dy_m) / meters_per_deg_lon(lat_p)

        est.append((lat0, lon0))

    est = np.asarray(est, float)
    lat0s = est[:, 0]
    lon0s = est[:, 1]

    lat_med = float(np.median(lat0s))
    lon_med = float(np.median(lon0s))
    mad_lat = float(np.median(np.abs(lat0s - lat_med)))
    mad_lon = float(np.median(np.abs(lon0s - lon_med)))

    thr_lat = 5 * mad_lat if mad_lat > 0 else 1e9
    thr_lon = 5 * mad_lon if mad_lon > 0 else 1e9

    inlier = (np.abs(lat0s - lat_med) <= thr_lat) & (np.abs(lon0s - lon_med) <= thr_lon)

    lat0 = float(np.mean(lat0s[inlier]))
    lon0 = float(np.mean(lon0s[inlier]))
    return lat0, lon0, inlier

def gcp_median_error_m(lat0, lon0, gcp_list, index_base, dx_m, dy_m):
    errs = []
    for I, J, lon_p, lat_p in gcp_list:
        lon_hat, lat_hat = predict_lonlat(lat0, lon0, I, J, index_base, dx_m, dy_m)
        # 거리(m)로 환산(국지 근사)
        dlat_m = (lat_hat - lat_p) * meters_per_deg_lat(lat_p)
        dlon_m = (lon_hat - lon_p) * meters_per_deg_lon(lat_p)
        errs.append(math.hypot(dlat_m, dlon_m))
    return float(np.median(errs))

# =========================================================
# 5) NetCDF 열기 + dx/dy + 축 해석(전치 여부)
# =========================================================
ds = Dataset(nc_file)
var = ds.variables[var_name]  # (T, Y, X)
nt, ny, nx = var.shape

# dx/dy: nc에 있으면 우선, 없으면 bath 헤더(없으면 100m)
if "DX_i" in ds.variables and len(ds.variables["DX_i"][:]) > 0:
    dx = float(ds.variables["DX_i"][0])
else:
    dx = float(bath.get("x_grid", 100.0))

if "DY_j" in ds.variables and len(ds.variables["DY_j"][:]) > 0:
    dy = float(ds.variables["DY_j"][0])
else:
    dy = float(bath.get("y_grid", 100.0))

# bath의 격자 수가 있으면 그것과 var shape를 비교해 전치 판단
bath_I = int(bath["x_rows"]) if "x_rows" in bath else None
bath_J = int(bath["y_columns"]) if "y_columns" in bath else None

# 기본 가정: var[t, :, :]의 shape = (J, I) = (ny, nx)  => data.T로 (I,J)
transpose_needed = True
I_count, J_count = nx, ny

if bath_I and bath_J:
    # var slice (ny, nx)
    if (ny, nx) == (bath_J, bath_I):
        transpose_needed = True
        I_count, J_count = bath_I, bath_J
    elif (ny, nx) == (bath_I, bath_J):
        transpose_needed = False
        I_count, J_count = bath_I, bath_J
    else:
        # 불일치하면 var shape 기반으로 진행
        transpose_needed = True
        I_count, J_count = nx, ny

print(f"[INFO] var.shape = (T={nt}, Y={ny}, X={nx})")
print(f"[INFO] dx={dx} m, dy={dy} m, transpose_needed={transpose_needed}")
print(f"[INFO] I_count={I_count}, J_count={J_count}")

# =========================================================
# 6) 시간 처리 (AEM3D 방식)
# =========================================================
def read_times(ds_):
    need = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
    if not all(k in ds_.variables for k in need):
        return [None] * nt
    y = ds_.variables["Year"][:]
    m = ds_.variables["Month"][:]
    d = ds_.variables["Day"][:]
    hh = ds_.variables["Hour"][:]
    mm = ds_.variables["Minute"][:]
    ss = ds_.variables["Second"][:]
    out = []
    for yy, mo, dd, h, mi, s in zip(y, m, d, hh, mm, ss):
        out.append(datetime(int(yy), int(mo), int(dd), int(h), int(mi), int(s)))
    return out

times = read_times(ds)

# =========================================================
# 7) INDEX_BASE 자동 선택(0/1 중 GCP 오차가 작은 쪽)
# =========================================================
def choose_index_base():
    candidates = []
    for base in (0, 1):
        lat0, lon0, inlier = estimate_origin_from_gcp(GCP, base, dx, dy)
        mederr = gcp_median_error_m(lat0, lon0, GCP, base, dx, dy)
        inlier_count = int(np.sum(inlier))
        candidates.append((mederr, -inlier_count, base, lat0, lon0, inlier))
    candidates.sort()
    return candidates[0]

if AUTO_INDEX_BASE:
    mederr, neg_inlier, INDEX_BASE, lat0, lon0, inlier = choose_index_base()
    print(f"[INFO] AUTO INDEX_BASE 선택 = {INDEX_BASE} (GCP median error ≈ {mederr:.1f} m, inlier={(-neg_inlier)})")
else:
    lat0, lon0, inlier = estimate_origin_from_gcp(GCP, INDEX_BASE, dx, dy)
    mederr = gcp_median_error_m(lat0, lon0, GCP, INDEX_BASE, dx, dy)
    print(f"[INFO] INDEX_BASE = {INDEX_BASE} (GCP median error ≈ {mederr:.1f} m)")

print("GCP 기반 원점 추정(lat0, lon0) =", lat0, lon0)
print("사용된 GCP(inlier) =", inlier.tolist())

# =========================================================
# 8) LatLonBox 경계 계산 (half-cell 보정)
#    - origin(lat0,lon0)는 좌상단 셀 중심이라고 가정
# =========================================================
mperlat = meters_per_deg_lat(lat0)

north = lat0 + (dx / 2) / mperlat
south = lat0 - ((I_count - 0.5) * dx) / mperlat

lat_mid = 0.5 * (north + south)
mperlon = meters_per_deg_lon(lat_mid)

west = lon0 - (dy / 2) / mperlon
east = lon0 + ((J_count - 0.5) * dy) / mperlon

print("LatLonBox =", dict(north=north, south=south, west=west, east=east))

# =========================================================
# 9) 컬러 스케일(고정) - read-only/MaskedArray 문제 없는 방식
#    - 전체를 var[:]로 한 번에 읽지 않고 시간축으로 샘플링
# =========================================================
def robust_percentiles_from_var(ncvar, p=(5, 95), invalid_leq=-999, sample_max=400_000, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    per_frame = max(2000, sample_max // max(1, ncvar.shape[0]))

    for t in range(ncvar.shape[0]):
        a = ncvar[t, :, :]
        if isinstance(a, ma.MaskedArray):
            a = a.filled(np.nan)
        else:
            a = np.asarray(a)

        a = a.astype(np.float32, copy=True)  # writeable 보장
        a[a <= invalid_leq] = np.nan

        # (J,I)->(I,J) 변환이 필요하면, 샘플링도 동일하게 변환해두는 게 안전
        if transpose_needed:
            a = a.T

        flat = a.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            continue

        if flat.size > per_frame:
            idx = rng.choice(flat.size, size=per_frame, replace=False)
            flat = flat[idx]

        samples.append(flat)

    if not samples:
        raise ValueError(f"{var_name}에 유효값이 없습니다(모두 마스크/NaN/결측).")

    s = np.concatenate(samples).astype(np.float64, copy=False)
    return np.percentile(s, p)

vmin, vmax = robust_percentiles_from_var(
    var, p=(5, 95),
    invalid_leq=INVALID_LEQ,
    sample_max=PERCENTILE_SAMPLE_MAX,
    seed=0
)
print(f"[INFO] color scale vmin={vmin:.6g}, vmax={vmax:.6g}")

cmap = cm.get_cmap("jet")
norm = colors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)

# =========================================================
# 10) 프레임 PNG 생성 (데이터만 저장: 컬러바/제목/여백 제거)
#     - row=I(south+)가 아래로 내려가야 하므로 "그대로" 저장(추가 flip 없음)
# =========================================================
png_frames = []

for t in range(nt):
    data = var[t, :, :]
    if isinstance(data, ma.MaskedArray):
        data = data.filled(np.nan)
    else:
        data = np.asarray(data)

    data = data.astype(np.float32, copy=True)
    data[data <= INVALID_LEQ] = np.nan

    if transpose_needed:
        data = data.T  # (J,I)->(I,J)

    rgba = cmap(norm(data))
    rgba[..., 3] = np.where(np.isnan(data), 0.0, 1.0)  # NaN 투명

    png_name = f"frame_{t:03d}.png"
    png_path = os.path.join(output_dir, png_name)
    plt.imsave(png_path, rgba)

    png_frames.append((png_name, times[t]))

# =========================================================
# 11) 범례(컬러바) PNG 1장 생성 → ScreenOverlay
# =========================================================
legend_name = "legend.png"
legend_path = os.path.join(output_dir, legend_name)

fig, ax = plt.subplots(figsize=(1.2, 5))
fig.patch.set_alpha(0)
ax.set_facecolor("none")

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax)
cbar.set_label(var_name)

plt.savefig(legend_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
plt.close(fig)

# =========================================================
# 12) KML 생성 (프레임 GroundOverlay + 범례 ScreenOverlay + GCP Placemark)
# =========================================================
kml_name = "aem3d_animation.kml"
kml_path = os.path.join(output_dir, kml_name)

with open(kml_path, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
    f.write("<Document>\n")

    # --- 범례: 화면 고정 ---
    f.write(f"""
<ScreenOverlay>
  <name>Legend</name>
  <Icon><href>{legend_name}</href></Icon>
  <overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/>
  <screenXY  x="0.02" y="0.05" xunits="fraction" yunits="fraction"/>
  <size x="0" y="0" xunits="pixels" yunits="pixels"/>
</ScreenOverlay>
""")

    # --- 기준점(검증용) ---
    for idx, (I, J, lon_p, lat_p) in enumerate(GCP, start=1):
        f.write(f"""
<Placemark>
  <name>GCP {idx} (I={I}, J={J})</name>
  <Point><coordinates>{lon_p},{lat_p},0</coordinates></Point>
</Placemark>
""")

    # --- 프레임 GroundOverlay ---
    for png, t in png_frames:
        if t is None:
            when = ""
        else:
            # UTC(Z) 강제하지 않음(로컬/모델시간 그대로 표시)
            when = t.strftime("%Y-%m-%dT%H:%M:%S")

        f.write(f"""
<GroundOverlay>
  <TimeStamp><when>{when}</when></TimeStamp>
  <Icon><href>{png}</href></Icon>
  <LatLonBox>
    <north>{north}</north>
    <south>{south}</south>
    <east>{east}</east>
    <west>{west}</west>
    <rotation>0</rotation>
  </LatLonBox>
</GroundOverlay>
""")

    f.write("</Document>\n</kml>")

# =========================================================
# 13) KMZ 패키징
# =========================================================
kmz_file = os.path.join(output_dir, "aem3d_animation.kmz")

with zipfile.ZipFile(kmz_file, "w", zipfile.ZIP_DEFLATED) as kmz:
    kmz.write(kml_path, arcname=kml_name)
    kmz.write(legend_path, arcname=legend_name)
    for png, _ in png_frames:
        kmz.write(os.path.join(output_dir, png), arcname=png)

print(f"✅ KMZ 생성 완료: {kmz_file}")
