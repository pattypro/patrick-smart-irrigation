# Patrick Smart Irrigation Dashboard v2.1

This Streamlit app automates precision irrigation scheduling using real-time sensor, NDVI, and weather data.

## 🌟 What's New in v2.1
- Auto-updates Google Sheets (creates missing tabs & headers)
- Detects and creates Weather_ETo, NDVI_Calibration, App_Metadata sheets automatically
- Keeps your old data safe in Patrick_Irrigation_Log
- Seeds initial NDVI calibration rows (initial/mid/late)
- Adds default metadata fields (version, date, researcher)

## 📂 Folder Contents
- `patrick_irrigation.py` — main Streamlit app
- `requirements.txt` — dependencies
- `README.md` — this file

## 🚀 Deployment Steps
1. Upload to GitHub (replace previous files)
2. In Streamlit Cloud:
   - Main file: `patrick_irrigation.py`
   - Add secret:  
     ```
     GCP_SERVICE_ACCOUNT_JSON = """{...}"""
     ```
3. Ensure your Google Sheet is named **Patrick_Irrigation_Log**
4. Share the sheet with your service account (Editor)

## 👨‍🔬 Author
Patrick Habyarimana — Smart Irrigation Research, 2025
