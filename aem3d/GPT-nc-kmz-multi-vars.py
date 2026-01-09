import os
import re
import zipfile
import math
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from netCDF4 import Dataset
from datetime import datetime, timedelta

# =========================================================
# 1) bath 헤더 파싱
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

# =========================================================
# 2) 미터<->도(deg) 국지 근사
# =========================================================
def meters_per_deg_lat(lat_deg: float) -> float:
    return 111_320.0

def meters_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))

# =========================================================
# 3) (I,J)->(lon,lat) / origin 추정
# =========================================================
def predict_lonlat(lat0, lon0, I, J, index_base, dx_m, dy_m):
    i0 = I - index_base
    j0 = J - index_base
    lat = lat0 - (i0 * dx_m) / meters_per_deg_lat(lat0)  # I south(+)
    lon = lon0 + (j0 * dy_m) / meters_per_deg_lon(lat)
    return lon, lat

def estimate_origin_from_gcp(gcp_list, index_base, dx_m, dy_m):
    est = []
    for I, J, lon_p, lat_p in gcp_list:
        i0 = I - index_base
        j0 = J - index_base
        lat0 = lat_p + (i0 * dx_m) / meters_per_deg_lat(lat_p)
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
        dlat_m = (lat_hat - lat_p) * meters_per_deg_lat(lat_p)
        dlon_m = (lon_hat - lon_p) * meters_per_deg_lon(lat_p)
        errs.append(math.hypot(dlat_m, dlon_m))
    return float(np.median(errs))

def choose_best_index_base(gcp, dx, dy):
    best = None
    for base in (0, 1):
        lat0, lon0, inlier = estimate_origin_from_gcp(gcp, base, dx, dy)
        mederr = gcp_median_error_m(lat0, lon0, gcp, base, dx, dy)
        score = (mederr, -int(np.sum(inlier)))
        cand = (score, base, lat0, lon0, inlier, mederr)
        if best is None or cand[0] < best[0]:
            best = cand
    _, base, lat0, lon0, inlier, mederr = best
    return base, lat0, lon0, inlier, mederr

# =========================================================
# 4) nc에서 시간 읽기
# =========================================================
def read_times(ds, nt):
    need = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
    if not all(k in ds.variables for k in need):
        return [None] * nt
    y = ds.variables["Year"][:]
    m = ds.variables["Month"][:]
    d = ds.variables["Day"][:]
    hh = ds.variables["Hour"][:]
    mm = ds.variables["Minute"][:]
    ss = ds.variables["Second"][:]
    out = []
    for yy, mo, dd, h, mi, s in zip(y, m, d, hh, mm, ss):
        out.append(datetime(int(yy), int(mo), int(dd), int(h), int(mi), int(s)))
    return out

def kml_time(dt: datetime) -> str:
    # ISO 8601 dateTime. GE에서 가장 안전하게 먹히는 형태가 Z 포함인 경우가 많음.
    # (모델 시간이 로컬이면 시간대만큼 숫자는 다를 수 있지만, 애니메이션 자체는 정상 동작)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# =========================================================
# 5) 결측 처리 + writeable ndarray 확보
# =========================================================
def to_writeable_float_array(a, invalid_leq=None, fill_value=None):
    if isinstance(a, ma.MaskedArray):
        a = a.filled(np.nan)
    else:
        a = np.asarray(a)
    arr = a.astype(np.float32, copy=True)  # writeable 보장
    if fill_value is not None:
        arr[arr == fill_value] = np.nan
    if invalid_leq is not None:
        arr[arr <= invalid_leq] = np.nan
    return arr

# =========================================================
# 6) robust percentile(vmin/vmax)
# =========================================================
def robust_percentiles_from_var(ncvar, transpose_needed, invalid_leq, fill_value,
                                p=(5, 95), sample_max=400_000, seed=0, z_index=None):
    rng = np.random.default_rng(seed)
    nt = ncvar.shape[0]

    samples = []
    per_frame = max(2000, sample_max // max(1, nt))

    for t in range(nt):
        if ncvar.ndim == 3:
            a = ncvar[t, :, :]
        elif ncvar.ndim == 4:
            zi = 0 if z_index is None else int(z_index)
            a = ncvar[t, zi, :, :]
        else:
            raise ValueError(f"지원하지 않는 변수 차원: {ncvar.ndim}D")

        arr = to_writeable_float_array(a, invalid_leq=invalid_leq, fill_value=fill_value)
        if transpose_needed:
            arr = arr.T

        flat = arr.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            continue

        if flat.size > per_frame:
            idx = rng.choice(flat.size, size=per_frame, replace=False)
            flat = flat[idx]

        samples.append(flat)

    if not samples:
        raise ValueError("유효 데이터가 없습니다(모두 NaN/결측).")

    s = np.concatenate(samples).astype(np.float64, copy=False)
    vmin, vmax = np.percentile(s, p)
    return float(vmin), float(vmax)

# =========================================================
# 7) 변수 1개: 폴더에 (frames + legend + var_kml) 생성
#    - 핵심 수정: TimeStamp -> TimeSpan(begin/end)
# =========================================================
def build_variable_folder(
    ds: Dataset,
    bath: dict,
    var_name: str,
    gcp: list,
    out_root: str,
    cmap_name: str,
    scale_rule: dict,
    invalid_leq: float = -999,
    auto_index_base: bool = True,
    index_base: int = 1,
    z_index: int | None = None,
    frame_stride: int = 1,
    sample_max: int = 400_000,
):
    if var_name not in ds.variables:
        raise KeyError(f"nc에 변수 {var_name}가 없습니다.")

    var = ds.variables[var_name]
    nt = var.shape[0]
    times = read_times(ds, nt)

    # dx/dy
    dx = float(ds.variables["DX_i"][0]) if "DX_i" in ds.variables and len(ds.variables["DX_i"][:]) > 0 else float(bath.get("x_grid", 100.0))
    dy = float(ds.variables["DY_j"][0]) if "DY_j" in ds.variables and len(ds.variables["DY_j"][:]) > 0 else float(bath.get("y_grid", 100.0))

    # shape / transpose 판단
    if var.ndim == 3:
        _, ny, nx = var.shape
    elif var.ndim == 4:
        _, _, ny, nx = var.shape
    else:
        raise ValueError(f"{var_name}: 지원하지 않는 차원({var.ndim}D)")

    bath_I = int(bath["x_rows"]) if "x_rows" in bath else None
    bath_J = int(bath["y_columns"]) if "y_columns" in bath else None

    transpose_needed = True
    I_count, J_count = nx, ny
    if bath_I and bath_J:
        if (ny, nx) == (bath_J, bath_I):
            transpose_needed = True
            I_count, J_count = bath_I, bath_J
        elif (ny, nx) == (bath_I, bath_J):
            transpose_needed = False
            I_count, J_count = bath_I, bath_J

    # fill_value
    fill_value = None
    for attr in ("_FillValue", "missing_value"):
        if hasattr(var, attr):
            try:
                fill_value = float(getattr(var, attr))
                break
            except Exception:
                pass

    # INDEX_BASE & origin
    if auto_index_base:
        index_base, lat0, lon0, inlier, mederr = choose_best_index_base(gcp, dx, dy)
        print(f"[{var_name}] AUTO INDEX_BASE={index_base} (GCP median err ~ {mederr:.1f} m, inlier={int(np.sum(inlier))})")
    else:
        lat0, lon0, inlier = estimate_origin_from_gcp(gcp, index_base, dx, dy)
        mederr = gcp_median_error_m(lat0, lon0, gcp, index_base, dx, dy)
        print(f"[{var_name}] INDEX_BASE={index_base} (GCP median err ~ {mederr:.1f} m)")

    # LatLonBox (half-cell)
    mperlat = meters_per_deg_lat(lat0)
    north = lat0 + (dx / 2) / mperlat
    south = lat0 - ((I_count - 0.5) * dx) / mperlat

    lat_mid = 0.5 * (north + south)
    mperlon = meters_per_deg_lon(lat_mid)
    west = lon0 - (dy / 2) / mperlon
    east = lon0 + ((J_count - 0.5) * dy) / mperlon

    # color scale
    mode = scale_rule.get("mode", "percentile").lower()
    if mode == "fixed":
        vmin = float(scale_rule["vmin"])
        vmax = float(scale_rule["vmax"])
    elif mode == "percentile":
        p_low = float(scale_rule.get("p_low", 5))
        p_high = float(scale_rule.get("p_high", 95))
        vmin, vmax = robust_percentiles_from_var(
            var,
            transpose_needed=transpose_needed,
            invalid_leq=invalid_leq,
            fill_value=fill_value,
            p=(p_low, p_high),
            sample_max=sample_max,
            seed=0,
            z_index=z_index
        )
    else:
        raise ValueError(f"[{var_name}] 알 수 없는 scale mode: {mode}")

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"[{var_name}] vmin/vmax 비정상: vmin={vmin}, vmax={vmax}")

    print(f"[{var_name}] vmin={vmin:.6g}, vmax={vmax:.6g} (mode={mode})")

    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    # output dir for this variable
    var_dir = os.path.join(out_root, var_name)
    os.makedirs(var_dir, exist_ok=True)

    # frames
    stride = max(1, int(frame_stride))
    frame_indices = list(range(0, nt, stride))
    png_frames = []

    for t in frame_indices:
        if var.ndim == 3:
            a = var[t, :, :]
        else:
            zi = 0 if z_index is None else int(z_index)
            a = var[t, zi, :, :]

        data = to_writeable_float_array(a, invalid_leq=invalid_leq, fill_value=fill_value)
        if transpose_needed:
            data = data.T

        rgba = cmap(norm(data))
        rgba[..., 3] = np.where(np.isnan(data), 0.0, 1.0)

        png_name = f"frame_{t:04d}.png"
        png_path = os.path.join(var_dir, png_name)
        plt.imsave(png_path, rgba)

        png_frames.append((png_name, times[t]))

    # legend (ScreenOverlay) - 작게 + 흰 글자 + 반투명 배경
    legend_name = "legend.png"
    legend_path = os.path.join(var_dir, legend_name)

    fig, ax = plt.subplots(figsize=(0.8, 3.0))  # <-- 크기 줄이기(가로, 세로)
    # 전체 그림은 투명 유지
    fig.patch.set_alpha(0)

    # 컬러바 뒤에만 반투명 어두운 배경(가독성용)
    ax.set_facecolor((0, 0, 0, 0.45))  # RGBA (검정, 알파 0.45)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax)

    # 글자/틱 색상 흰색
    cbar.set_label(var_name, color="white", fontsize=9)
    cbar.ax.tick_params(colors="white", labelsize=8, length=2)

    # 테두리도 흰색(지도 위에서 잘 보이게)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.8)

    # (선택) 지수표기/오프셋 텍스트도 흰색
    cbar.ax.yaxis.get_offset_text().set_color("white")

    plt.savefig(legend_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # TimeSpan용 end 계산(마지막 프레임 대비)
    # - 공식 권장: GroundOverlay는 TimeSpan으로 "구간"을 주는 방식 :contentReference[oaicite:1]{index=1}
    # - 마지막 end는 직전 간격을 사용하거나 1시간 기본값
    # -----------------------------
    # TimeSpan용 end 계산(프레임 간격 추정)
    # -----------------------------
    def safe_delta():
        # 프레임 간격(모델 시간) 추정: 가능한 delta들의 중앙값 사용
        deltas = []
        for i in range(len(frame_indices) - 1):
            t0 = times[frame_indices[i]]
            t1 = times[frame_indices[i + 1]]
            if t0 and t1 and t1 > t0:
                deltas.append(t1 - t0)

        if deltas:
            deltas.sort()
            return deltas[len(deltas) // 2]  # median
        return timedelta(hours=1)

    default_dt = safe_delta()

    # variable kml
    var_kml_name = f"{var_name}_animation.kml"
    var_kml_path = os.path.join(var_dir, var_kml_name)

    with open(var_kml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f.write("<Document>\n")
        f.write(f"<name>{var_name}</name>\n")

        # 리스트에서 프레임(자식) 숨김(선택 사항)
        f.write("""
<Style id="check-hide-children">
  <ListStyle>
    <listItemType>checkHideChildren</listItemType>
  </ListStyle>
</Style>
<styleUrl>#check-hide-children</styleUrl>
""")

        # Legend
        f.write(f"""
<ScreenOverlay>
  <name>CYANO Legend</name>
  <Icon><href>legend.png</href></Icon>
  <overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/>
  <screenXY  x="0.02" y="0.05" xunits="fraction" yunits="fraction"/>
  <size x="0.06" y="0.28" xunits="fraction" yunits="fraction"/>
</ScreenOverlay>
""")

        # GCP (검증용)
        for idx, (I, J, lon_p, lat_p) in enumerate(gcp, start=1):
            f.write(f"""
<Placemark>
  <name>GCP {idx} (I={I}, J={J})</name>
  <Point><coordinates>{lon_p},{lat_p},0</coordinates></Point>
</Placemark>
""")

        # Frames with TimeSpan
        for i, (png, t_begin) in enumerate(png_frames):
            if t_begin is None:
                # 시간이 없으면 시간태그 생략(애니메이션 불가)
                time_tag = ""
            else:
                if i < len(png_frames) - 1 and png_frames[i + 1][1] is not None:
                    t_end = png_frames[i + 1][1]
                    if t_end <= t_begin:
                        t_end = t_begin + default_dt
                else:
                    t_end = t_begin + default_dt

                time_tag = f"""
  <TimeSpan>
    <begin>{kml_time(t_begin)}</begin>
    <end>{kml_time(t_end)}</end>
  </TimeSpan>
"""

            f.write(f"""
<GroundOverlay>
  <name>{var_name} {i:04d}</name>
{time_tag}
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

    return {
        "var_name": var_name,
        "var_dir": var_dir,
        "var_kml_name": var_kml_name,
    }

# =========================================================
# 8) 마스터 KML(doc.kml): Folder + NetworkLink 레이어
# =========================================================
def write_master_kml(master_kml_path: str, layers: list[dict], default_visible_var: str | None = None):
    with open(master_kml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f.write("<Document>\n")
        f.write("<name>AEM3D Multi-Variable Layers</name>\n")
        f.write("<open>1</open>\n")

        for layer in layers:
            v = layer["var_name"]
            href = f"{v}/{layer['var_kml_name']}"
            vis = 1 if (default_visible_var is not None and v == default_visible_var) else 0

            f.write(f"""
<Folder>
  <name>{v}</name>
  <visibility>{vis}</visibility>
  <NetworkLink>
    <name>{v}</name>
    <Link>
      <href>{href}</href>
    </Link>
  </NetworkLink>
</Folder>
""")

        f.write("</Document>\n</kml>")

# =========================================================
# 9) 최종: 한 KMZ로 묶기
# =========================================================
def build_single_kmz_with_layers(
    nc_file: str,
    bath_file: str,
    var_list: list[str],
    gcp: list,
    out_root: str = "kml_multi_out",
    kmz_name: str = "aem3d_multi_layers.kmz",
    cmap_name: str = "jet",
    scale_rules: dict | None = None,
    default_scale_rule: dict | None = None,
    invalid_leq: float = -999,
    auto_index_base: bool = True,
    index_base: int = 1,
    z_index: int | None = None,
    frame_stride: int = 1,
    sample_max: int = 400_000,
    default_visible_var: str | None = None,
):
    os.makedirs(out_root, exist_ok=True)
    if default_scale_rule is None:
        default_scale_rule = {"mode": "percentile", "p_low": 5, "p_high": 95}

    bath = parse_bath_header(bath_file)
    layers = []

    ds = Dataset(nc_file)
    try:
        for v in var_list:
            rule = default_scale_rule if scale_rules is None else scale_rules.get(v, default_scale_rule)
            meta = build_variable_folder(
                ds=ds,
                bath=bath,
                var_name=v,
                gcp=gcp,
                out_root=out_root,
                cmap_name=cmap_name,
                scale_rule=rule,
                invalid_leq=invalid_leq,
                auto_index_base=auto_index_base,
                index_base=index_base,
                z_index=z_index,
                frame_stride=frame_stride,
                sample_max=sample_max,
            )
            layers.append(meta)
    finally:
        ds.close()

    # master KML at root
    master_kml_path = os.path.join(out_root, "doc.kml")
    write_master_kml(master_kml_path, layers, default_visible_var=default_visible_var)

    # pack into one KMZ
    kmz_path = os.path.join(out_root, kmz_name)
    with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as kmz:
        kmz.write(master_kml_path, arcname="doc.kml")
        for layer in layers:
            vdir = layer["var_dir"]
            vname = layer["var_name"]
            for fn in os.listdir(vdir):
                fp = os.path.join(vdir, fn)
                if os.path.isfile(fp):
                    kmz.write(fp, arcname=f"{vname}/{fn}")

    print(f"✅ 단일 KMZ(레이어+NetworkLink+TimeSpan 애니메이션) 생성 완료: {kmz_path}")
    return kmz_path

# =========================================================
# 10) 실행 예시
# =========================================================
if __name__ == "__main__":
    nc_file   = r"c:\dev\pythonProject\aem3d\sheet_top_cyano.nc"
    bath_file = r"c:\dev\pythonProject\aem3d\bath100_edm1_geo.xyz"

    GCP = [
        [45, 18, 127.480833, 36.477569],
        [193, 68, 127.650222, 36.349806],
        [99, 81, 127.550906, 36.426133],
        [160, 12, 127.477294, 36.370797]
    ]

    VARS = ["CYANO", "DO", "TP"]  # 예: ["CYANO", "DO", "NH4", "NO3", "TCHLA"]

    SCALE_RULES = {
        # "CYANO": {"mode": "fixed", "vmin": 0, "vmax": 50},
        # "DO": {"mode": "fixed", "vmin": 0, "vmax": 15},
    }
    DEFAULT_RULE = {"mode": "percentile", "p_low": 5, "p_high": 95}

    build_single_kmz_with_layers(
        nc_file=nc_file,
        bath_file=bath_file,
        var_list=VARS,
        gcp=GCP,
        out_root="kml_multi_out",
        kmz_name="aem3d_multi_layers.kmz",
        cmap_name="jet",
        scale_rules=SCALE_RULES,
        default_scale_rule=DEFAULT_RULE,
        invalid_leq=-999,
        auto_index_base=True,
        index_base=1,
        z_index=None,
        frame_stride=1,
        sample_max=400_000,
        default_visible_var="CYANO",
    )
