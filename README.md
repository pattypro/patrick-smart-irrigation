
# Patrick Smart Irrigation - v4.1 (Cached & Quota-Safe)

## What's new
- **Cached Google Sheet reads** using `st.cache_data(ttl=300)` (refresh every 5 min)
- **Retry-safe writes** with automatic error handling
- **All v4.0 features preserved**
- Prevents 429 "quota exceeded" errors from Google Sheets

## Setup
Add this to your `.streamlit/secrets.toml`:
```toml
GCP_SERVICE_ACCOUNT_JSON = """{ your Google service account JSON here }"""
SHEET_ID = "17lGpO4UeBDHt_NXMVoi4xO8EAyZMUmPBcWE6hCsvWvo"
```

## Run
```bash
pip install -r requirements.txt
streamlit run patrick_irrigation.py
```
