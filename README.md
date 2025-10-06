# 🌿 Patrick Smart Irrigation Dashboard — v3.0 (Production)

**Highlights**
- Uses YOUR Google Drive Sheet (no service account storage).
- Auto-creates tabs & headers; preserves your old data.
- Weather → ETo (FAO-56) from Japanese CSV.
- Plot-aware daily decisions (Plot 1 = manual only).
- NDVI harmonization (RGN+OCN) and fusion.
- Analytics with clean design, compact ticks, and annotated logic overlays.

## Deploy
1) Upload to GitHub (replace prior files).
2) Streamlit Cloud → main file: `patrick_irrigation.py`
3) Secrets:
```
GCP_SERVICE_ACCOUNT_JSON = """{ ... }"""
# Optional but recommended for precision:
SHEET_ID = "your_google_sheet_id_here"
```
4) Share your sheet with: `patrick-irrigation-sa@patrick-irrigation-473904.iam.gserviceaccount.com` (Editor).
