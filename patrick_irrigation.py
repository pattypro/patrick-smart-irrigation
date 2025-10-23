# Patrick Smart Irrigation v4.1 — Cached Edition
import streamlit as st, gspread, json, pandas as pd, numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

st.set_page_config(page_title="Patrick Smart Irrigation v4.1", layout="wide")
st.title("Patrick Smart Irrigation — v4.1 (Cached Edition)")

DEFAULT_SHEET_ID = "17lGpO4UeBDHt_NXMVoi4xO8EAyZMUmPBcWE6hCsvWvo"

@st.cache_resource
def connect_gsheet():
    scope=[
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds_json = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
    creds_dict = json.loads(creds_json) if isinstance(creds_json,str) else dict(creds_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet_id = st.secrets.get("SHEET_ID", DEFAULT_SHEET_ID)
    ss = client.open_by_key(sheet_id)
    return ss

@st.cache_data(ttl=300)
def read_records(sheet_name):
    try:
        ss = connect_gsheet()
        ws = ss.worksheet(sheet_name)
        return ws.get_all_records()
    except Exception as e:
        st.warning(f"Cache read failed: {e}")
        return []

def safe_append(sheet_name, row):
    try:
        ss = connect_gsheet()
        ws = ss.worksheet(sheet_name)
        ws.append_row(row)
        st.info(f"✅ Row saved to {sheet_name}")
    except Exception as e:
        st.warning(f"Write deferred (quota-safe): {e}")

sheet_link = f"https://docs.google.com/spreadsheets/d/{DEFAULT_SHEET_ID}/edit"
st.sidebar.markdown(f"[Open Google Sheet]({sheet_link})")

st.write("### Cached Data Read Example")
data = read_records("MAIN")
st.write(f"Loaded {len(data)} rows from cache. (refresh every 5 minutes)")

if st.button("Simulate new log entry"):
    ts = datetime.utcnow().isoformat()
    safe_append("MAIN", [ts, "T2", "Spinach", "mid", "Auto test entry"])
