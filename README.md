# 🌿 Patrick Smart Irrigation Dashboard — v3.3

**What's new in v3.3**
- Full support for **English weather CSV** (Date, Pressure (hPa), Rainfall (mm), Temperature (°C), Humidity (%), Wind Speed (m/s), Sunshine (h)).
- Smart date handling: if Date is day-only (1–31), the app infers the current month/year.
- Cleaner, more readable **graphs** with compact ticks, gridlines, and clear titles.
- Keeps v3.1 decision consistency fix and all previous features.

## Deploy
1) Upload all files to GitHub.
2) Streamlit Cloud: set main file to `patrick_irrigation.py`.
3) Secrets:
```
GCP_SERVICE_ACCOUNT_JSON = """{ ... your service account json ... }"""
# Optional but recommended:
SHEET_ID = "your_google_sheet_id_here"
```
4) Share your Google Sheet with:
`patrick-irrigation-sa@patrick-irrigation-473904.iam.gserviceaccount.com` (Editor).
