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
# 1) bath 헤더 파싱 (기존 유지)
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
# 2) 미터<->도(deg) 국지 근사 (기존 유지)
# =========================================================
def meters_per_deg_lat(lat_deg: float) -> float:
    return 111_320.0

def meters_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))

# =========================================================
# 3) (I,J)->(lon,lat) / origin 추정 (기존 유지)
# =========================================================
def predict_lonlat(lat0, lon0, I, J, index_base, dx_m, dy_m):
    i0 = I - index_base
    j0 = J - index_base
    lat = lat0 - (i0 * dx_m) / meters_per_deg_lat(lat0)
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
    lat_med, lon_med = float(np.median(est[:, 0])), float(np.median(est[:, 1]))
    mad_lat = float(np.median(np.abs(est[:, 0] - lat_med)))
    mad_lon = float(np.median(np.abs(est[:, 1] - lon_med)))
    thr_lat = 5 * mad_lat if mad_lat > 0 else 1e9
    thr_lon = 5 * mad_lon if mad_lon > 0 else 1e9
    inlier = (np.abs(est[:, 0] - lat_med) <= thr_lat) & (np.abs(est[:, 1] - lon_med) <= thr_lon)
    lat0, lon0 = float(np.mean(est[inlier, 0])), float(np.mean(est[inlier, 1]))
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
# 4) 시간 및 데이터 처리 (기존 유지)
# =========================================================
def read_times(ds, nt):
    need = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
    if not all(k in ds.variables for k in need): return [None] * nt
    y, m, d, hh, mm, ss = (ds.variables[k][:] for k in need)
    return [datetime(int(yy), int(mo), int(dd), int(h), int(mi), int(s)) for yy, mo, dd, h, mi, s in zip(y, m, d, hh, mm, ss)]

def kml_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def to_writeable_float_array(a, invalid_leq=None, fill_value=None):
    if isinstance(a, ma.MaskedArray): a = a.filled(np.nan)
    else: a = np.asarray(a)
    arr = a.astype(np.float32, copy=True)
    if fill_value is not None: arr[arr == fill_value] = np.nan
    if invalid_leq is not None: arr[arr <= invalid_leq] = np.nan
    return arr

def robust_percentiles_from_var(ncvar, transpose_needed, invalid_leq, fill_value, p=(5, 95), sample_max=400_000, seed=0, z_index=None):
    rng = np.random.default_rng(seed)
    nt = ncvar.shape[0]; samples = []
    per_frame = max(2000, sample_max // max(1, nt))
    for t in range(nt):
        a = ncvar[t, :, :] if ncvar.ndim == 3 else ncvar[t, z_index or 0, :, :]
        arr = to_writeable_float_array(a, invalid_leq=invalid_leq, fill_value=fill_value)
        if transpose_needed: arr = arr.T
        flat = arr.ravel(); flat = flat[np.isfinite(flat)]
        if flat.size == 0: continue
        if flat.size > per_frame: flat = flat[rng.choice(flat.size, size=per_frame, replace=False)]
        samples.append(flat)
    if not samples: raise ValueError("유효 데이터가 없습니다.")
    return float(np.percentile(np.concatenate(samples), p[0])), float(np.percentile(np.concatenate(samples), p[1]))

# =========================================================
# 7) 변수 1개 처리 (교수님 원래 구조 + 시계열 팝업 추가)
# =========================================================
def build_variable_folder(
    ds: Dataset, bath: dict, var_name: str, gcp: list, out_root: str, cmap_name: str, scale_rule: dict,
    invalid_leq: float = -999, auto_index_base: bool = True, index_base: int = 1,
    z_index: int | None = None, frame_stride: int = 1, sample_max: int = 400_000
):
    if var_name not in ds.variables: raise KeyError(f"nc에 {var_name}가 없습니다.")
    var = ds.variables[var_name]; nt = var.shape[0]; times = read_times(ds, nt)
    dx = float(ds.variables["DX_i"][0]) if "DX_i" in ds.variables and len(ds.variables["DX_i"][:]) > 0 else float(bath.get("x_grid", 100.0))
    dy = float(ds.variables["DY_j"][0]) if "DY_j" in ds.variables and len(ds.variables["DY_j"][:]) > 0 else float(bath.get("y_grid", 100.0))

    ny, nx = (var.shape[-2], var.shape[-1])
    bath_I, bath_J = int(bath.get("x_rows", 0)), int(bath.get("y_columns", 0))
    transpose_needed = True; I_count, J_count = nx, ny
    if bath_I and bath_J:
        if (ny, nx) == (bath_J, bath_I): transpose_needed, I_count, J_count = True, bath_I, bath_J
        elif (ny, nx) == (bath_I, bath_J): transpose_needed, I_count, J_count = False, bath_I, bath_J

    fill_value = getattr(var, "_FillValue", None)
    if auto_index_base:
        index_base, lat0, lon0, inlier, mederr = choose_best_index_base(gcp, dx, dy)
        print(f"[{var_name}] AUTO INDEX_BASE={index_base} (Err: {mederr:.1f}m)")
    else:
        lat0, lon0, inlier = estimate_origin_from_gcp(gcp, index_base, dx, dy)

    mperlat = meters_per_deg_lat(lat0)
    north, south = lat0 + (dx/2)/mperlat, lat0 - ((I_count - 0.5)*dx)/mperlat
    mperlon = meters_per_deg_lon(0.5*(north+south))
    west, east = lon0 - (dy/2)/mperlon, lon0 + ((J_count - 0.5)*dy)/mperlon

    if scale_rule.get("mode") == "fixed":
        vmin, vmax = float(scale_rule["vmin"]), float(scale_rule["vmax"])
    else:
        vmin, vmax = robust_percentiles_from_var(var, transpose_needed, invalid_leq, fill_value, z_index=z_index)

    cmap, norm = cm.get_cmap(cmap_name), colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    var_dir = os.path.join(out_root, var_name); os.makedirs(var_dir, exist_ok=True)

    # --- 시계열 그래프 생성 ---
    gcp_plots = []
    for idx, (I, J, lon_p, lat_p) in enumerate(gcp, start=1):
        ts_data = to_writeable_float_array(var[:, J-index_base, I-index_base] if var.ndim==3 else var[:, z_index or 0, J-index_base, I-index_base], invalid_leq)
        plt.figure(figsize=(5, 3)); plt.plot(times, ts_data, 'b-'); plt.grid(True, alpha=0.3)
        plt.title(f"GCP {idx} ({var_name})"); plt.tight_layout()
        fn = f"gcp_{idx}_ts.png"; plt.savefig(os.path.join(var_dir, fn), dpi=120); plt.close(); gcp_plots.append(fn)

    # --- 프레임 생성 ---
    png_frames = []
    for t in range(0, nt, max(1, frame_stride)):
        data = to_writeable_float_array(var[t, :, :] if var.ndim==3 else var[t, z_index or 0, :, :], invalid_leq, fill_value)
        if transpose_needed: data = data.T
        rgba = cmap(norm(data)); rgba[..., 3] = np.where(np.isnan(data), 0, 1)
        fn = f"frame_{t:04d}.png"; plt.imsave(os.path.join(var_dir, fn), rgba); png_frames.append((fn, times[t]))

    # 범례 생성
    fig, ax = plt.subplots(figsize=(0.8, 3.5)); fig.patch.set_alpha(0); ax.set_facecolor((0,0,0,0.45))
    # 항목명 추가 (위에만 표시)
    ax.text(0.5, 1.05, var_name, transform=ax.transAxes, ha='center', fontsize=36, fontweight='bold', color='white')
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cbar.ax.tick_params(colors="white", labelsize=20)
    plt.savefig(os.path.join(var_dir, "legend.png"), dpi=200, transparent=True, bbox_inches="tight"); plt.close()

    # KML 생성
    kml_path = os.path.join(var_dir, f"{var_name}_animation.kml")
    with open(kml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n')
        f.write(f'<name>{var_name}</name>\n<ScreenOverlay><name>Legend</name><Icon><href>legend.png</href></Icon><overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/><screenXY x="0.02" y="0.05" xunits="fraction" yunits="fraction"/><size x="0.06" y="0.28" xunits="fraction" yunits="fraction"/></ScreenOverlay>\n')
        for idx, (I, J, lon_p, lat_p) in enumerate(gcp, start=1):
            f.write(f'<Placemark><name>GCP {idx}</name><description><![CDATA[<img src="{gcp_plots[idx-1]}" width="400" />]]></description><Point><coordinates>{lon_p},{lat_p},0</coordinates></Point></Placemark>\n')
        dt_def = (times[1]-times[0]) if len(times)>1 else timedelta(hours=1)
        for i, (png, t_beg) in enumerate(png_frames):
            t_end = png_frames[i+1][1] if i < len(png_frames)-1 else t_beg + dt_def
            f.write(f'<GroundOverlay><name>{i:04d}</name><TimeSpan><begin>{kml_time(t_beg)}</begin><end>{kml_time(t_end)}</end></TimeSpan><Icon><href>{png}</href></Icon><LatLonBox><north>{north}</north><south>{south}</south><east>{east}</east><west>{west}</west></LatLonBox></GroundOverlay>\n')
        f.write("</Document>\n</kml>")
    return {"var_name": var_name, "var_dir": var_dir, "var_kml_name": f"{var_name}_animation.kml"}

# =========================================================
# 8) 마스터 KML & 9) KMZ 묶기 (기존 유지)
# =========================================================
def write_master_kml(master_kml_path, layers, default_visible_var=None):
    with open(master_kml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Layers</name>\n')
        for layer in layers:
            v = layer["var_name"]; vis = 1 if v == default_visible_var else 0
            f.write(f'<Folder><name>{v}</name><visibility>{vis}</visibility><NetworkLink><Link><href>{v}/{layer["var_kml_name"]}</href></Link></NetworkLink></Folder>\n')
        f.write("</Document></kml>")

def build_single_kmz_with_layers(nc_file, bath_file, var_list, gcp, out_root="kml_multi_out", kmz_name="aem3d_multi_layers.kmz", **kwargs):
    os.makedirs(out_root, exist_ok=True); bath = parse_bath_header(bath_file); layers = []
    ds = Dataset(nc_file); default_scale = {"mode": "percentile"}
    try:
        for v in var_list:
            rule = kwargs.get("scale_rules", {}).get(v, kwargs.get("default_scale_rule", default_scale))
            # [수정] build_variable_folder가 수용 가능한 인자만 명시적으로 전달
            layers.append(build_variable_folder(
                ds=ds, bath=bath, var_name=v, gcp=gcp, out_root=out_root,
                cmap_name=kwargs.get("cmap_name", "jet"), scale_rule=rule,
                invalid_leq=kwargs.get("invalid_leq", -999),
                auto_index_base=kwargs.get("auto_index_base", True),
                index_base=kwargs.get("index_base", 1),
                z_index=kwargs.get("z_index"),
                frame_stride=kwargs.get("frame_stride", 1),
                sample_max=kwargs.get("sample_max", 400000)
            ))
    finally: ds.close()
    master_kml = os.path.join(out_root, "doc.kml")
    write_master_kml(master_kml, layers, kwargs.get("default_visible_var"))
    with zipfile.ZipFile(os.path.join(out_root, kmz_name), "w", zipfile.ZIP_DEFLATED) as kmz:
        kmz.write(master_path := os.path.join(out_root, "doc.kml"), arcname="doc.kml")
        for layer in layers:
            vdir, vname = layer["var_dir"], layer["var_name"]
            for fn in os.listdir(vdir): kmz.write(os.path.join(vdir, fn), arcname=f"{vname}/{fn}")
    print(f"✅ KMZ 생성 완료: {os.path.join(out_root, kmz_name)}")

# =========================================================
# 10) 실행 예시 (교수님 원래 형식 그대로 유지)
# =========================================================
if __name__ == "__main__":
    nc_file   = r"d:\dev\pythonProject\aem3d\sheet_top_cyano.nc"
    bath_file = r"dc:\dev\pythonProject\aem3d\bath100_edm1_geo.xyz"

    GCP = [
        [45, 18, 127.480833, 36.477569],
        [193, 68, 127.650222, 36.349806],
        [99, 81, 127.550906, 36.426133],
        [160, 12, 127.477294, 36.370797]
    ]

    VARS = ["CYANO", "DO", "TP"]

    build_single_kmz_with_layers(
        nc_file=nc_file,
        bath_file=bath_file,
        var_list=VARS,
        gcp=GCP,
        out_root="kml_multi_out",
        kmz_name="aem3d_multi_layers.kmz",
        cmap_name="jet",
        default_visible_var="CYANO",
    )