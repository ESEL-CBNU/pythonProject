import sys
import os
import json
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.join(os.getcwd(), "mcp-usgs-agent"))
from mcp_server_usgs_forecast import usgs_stage_forecast_report

site = "10109000"
now = datetime.now(timezone.utc)
start_dt = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
end_dt = now.strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"Fetching USGS data for site={site}, start={start_dt}, end={end_dt}")
res = usgs_stage_forecast_report(site=site, start=start_dt, end=end_dt, window_hours=24, horizon_hours=6, epochs=0, retrain=False)

os.makedirs('reports', exist_ok=True)
summary_json = os.path.join('reports', f'usgs_{site}_now_forecast_summary.json')
with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

# CSV full forecast (future series)
import pandas as pd
if res.get('html_report_path'):
    html_path = res['html_report_path']
else:
    html_path = os.path.join('reports', f'usgs_{site}_now_forecast.html')

# forecast_head_rows contains head; try to reconstruct full future series if available in file name
csv_head = os.path.join('reports', f'usgs_{site}_now_forecast_head.csv')
import csv
with open(csv_head, 'w', newline='', encoding='utf-8') as f:
    if res.get('forecast_head_rows'):
        keys = list(res['forecast_head_rows'][0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(res['forecast_head_rows'])

print('SUMMARY_JSON:', summary_json)
print('CSV_HEAD:', csv_head)
print('HTML_REPORT:', html_path)
print(json.dumps(res, indent=2, ensure_ascii=False))
