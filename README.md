# 🌿 Patrick Smart Irrigation Dashboard — v3.1

**Fixes**
- Consistent decision logic (no 'Skip' when liters > 0 unless rain override)
- Clear messages; full feature set (Weather→ETo, Daily Decision, Analytics, NDVI Harmonization)
- Uses YOUR Google Drive sheet

## Deploy
- Main file: `patrick_irrigation.py`
- Secrets: `GCP_SERVICE_ACCOUNT_JSON`, optional `SHEET_ID`
- Share your sheet with the service account (Editor)
