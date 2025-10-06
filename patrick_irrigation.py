import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

st.set_page_config(page_title="Patrick Smart Irrigation Dashboard v2.3", layout="wide")

# =====================================================
# GOOGLE SHEET CONNECTION (uses YOUR Drive storage)
# =====================================================
def connect_gsheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    raw = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
    creds_dict = json.loads(raw) if isinstance(raw, str) else dict(raw)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    try:
        if "SHEET_ID" in st.secrets:
            ss = client.open_by_key(st.secrets["SHEET_ID"])
        else:
            ss = client.open("Patrick_Irrigation_Log")
        st.sidebar.success("✅ Connected to your personal Google Drive Sheet!")
    except gspread.SpreadsheetNotFound:
        st.error("❌ Could not find the Google Sheet. Please share your sheet with the service account and refresh.")
        raise
    return client, ss

# =====================================================
# APP START
# =====================================================
st.title("Patrick Smart Irrigation Dashboard v2.3")

try:
    client, ss = connect_gsheet()
    st.success(f"Connected successfully to sheet: {ss.title}")
except Exception as e:
    st.error(f"Failed to connect: {e}")

st.markdown("""
### 🌿 How this version works
- Data is now saved directly to **your Google Drive**, not the service account’s storage.
- Just share your Google Sheet with this email:
  `patrick-irrigation-sa@patrick-irrigation-473904.iam.gserviceaccount.com`
- Grant **Editor** access.
- Then redeploy the app — it will connect directly to your personal sheet.
""")
