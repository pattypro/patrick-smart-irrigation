# 🌱 Patrick Smart Irrigation Dashboard — v2.2

**This version restores your full dashboard** and keeps the automatic Google Sheet structure update at startup.

## What’s in v2.2
- ✅ Auto-creates/updates sheets & headers (Weather_ETo, NDVI_Calibration, App_Metadata)
- ✅ Plot logic: P1 baseline; P2 Sensor+Weather; P3 NDVI+Weather; P4 Sensor+NDVI+Weather
- ✅ Weather CSV upload (Japanese headers OK) → FAO-56 ETo
- ✅ Auto-prefill yesterday’s ETo from Weather_ETo
- ✅ NDVI fusion (RGN & OCN with OCN→RGN linear map and noise-aware weighting)
- ✅ Analytics with per-plot filter

## Deploy
1) Upload files to GitHub.
2) Streamlit Cloud → main file: `patrick_irrigation.py`
3) Secrets → add your service account JSON:
```
GCP_SERVICE_ACCOUNT_JSON = """{ ... }"""
```
4) Share Google Sheet named **Patrick_Irrigation_Log** with your service account (Editor).
