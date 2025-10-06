# 🌱 Patrick Smart Irrigation Dashboard — v2

**New in v2**
- Plot-aware logic:
  - Plot 1 = Baseline (manual only)
  - Plot 2 = Sensor + Weather
  - Plot 3 = NDVI + Weather
  - Plot 4 = Sensor + NDVI + Weather
- Auto-prefill **yesterday’s ETo** and rain from `Weather_ETo`.
- Unified NDVI (OCN→RGN) with per-stage coefficients; used automatically.
- Once-a-day decision with TAW/RAW/Trigger VWC and recommended liters.

## Deploy
1) Upload `patrick_irrigation.py`, `requirements.txt`, `README.md` to GitHub.
2) Streamlit Cloud → Main file: `patrick_irrigation.py`
3) Secrets (Settings → Secrets):
```
GCP_SERVICE_ACCOUNT_JSON = """{ ...your service account JSON on one line... }"""
```
4) Share your Google Sheet named **Patrick_Irrigation_Log** with the service account (Editor).

## Sheets auto-created
- `Weather_ETo`
- `NDVI_Calibration`
