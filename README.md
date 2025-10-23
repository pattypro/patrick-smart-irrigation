# Patrick Smart Irrigation - v3.5

Dynamic treatments (T1-T4), spinach crop modeling, NDVI-integrated irrigation, sensor & weather CSV ingestion, and full Google Sheets logging.

## Google Sheets secrets

Add to your Streamlit secrets:
```
GCP_SERVICE_ACCOUNT_JSON = "<your service account JSON>"
SHEET_ID = "<optional - else it opens by name 'Patrick_Irrigation_Log'>"
```

## Run
```
pip install -r requirements.txt
streamlit run patrick_irrigation_v3_5.py
```
