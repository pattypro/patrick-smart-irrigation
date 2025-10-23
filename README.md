# Patrick Smart Irrigation - v3.7 (Auto)

Two tabs: Treatment Dashboard and Analytics. Upload **sensor CSV** and **weather CSV** directly in the Treatment tab.
The app computes **ETo**, harmonizes **NDVI**, makes decisions, **calculates liters**, and logs to **Google Sheets**.
No CSVs are saved by the app (processed in-memory).

## Streamlit secrets
```
GCP_SERVICE_ACCOUNT_JSON = "<service account JSON>"
SHEET_ID = "<optional; else open by name 'Patrick_Irrigation_Log'>"
```

## Run
```
pip install -r requirements.txt
streamlit run patrick_irrigation.py
```

**Note:** Plot area is fixed to 1.0 m² → 1 mm = 1 L.
