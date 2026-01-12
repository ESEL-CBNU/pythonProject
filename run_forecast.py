import sys
import os
import json
# ensure module path
sys.path.insert(0, os.path.join(os.getcwd(), "mcp-usgs-agent"))
from mcp_server_usgs_forecast import usgs_stage_forecast_report

site = "10109000"
start = "2025-12-01T00:00:00Z"
end = "2025-12-31T23:59:59Z"

print("Running forecast... this may take a moment if fetching data")
res = usgs_stage_forecast_report(site=site, start=start, end=end, window_hours=24, horizon_hours=6, epochs=0, retrain=False)

# save JSON
os.makedirs('reports', exist_ok=True)
out_json = os.path.join('reports', f'usgs_{site}_forecast_summary.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

# save CSV of forecast head rows
import csv
csv_path = os.path.join('reports', f'usgs_{site}_forecast_head.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    if res.get('forecast_head_rows'):
        keys = list(res['forecast_head_rows'][0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(res['forecast_head_rows'])

print('SUMMARY_JSON:', out_json)
print('CSV_HEAD:', csv_path)
print('HTML_REPORT:', res.get('html_report_path'))
print(json.dumps(res, indent=2, ensure_ascii=False))
