
# Patrick Smart Irrigation - v4.0 (Flow-Integrated & Stage-Aware)

- Auto MAD by stage: 0.25 (initial), 0.35 (mid), 0.45 (late)
- Editable Flow rate (L/min) — measure each run (0 = not set)
- Irrigation time computed in seconds (rounded) and saved
- T1: Manual (no CSV); T2–T4: Weather/Sensor uploads with transparent formulas
- All tabs auto-created in Google Sheet: MAIN, Weather_Raw, Sensor_Raw, Weather_ETo, NDVI_Calibration, App_Metadata
- Sidebar button to open your Google Sheet

## Secrets
```toml
GCP_SERVICE_ACCOUNT_JSON = """{ your service account JSON }"""
SHEET_ID = "17lGpO4UeBDHt_NXMVoi4xO8EAyZMUmPBcWE6hCsvWvo"
```
