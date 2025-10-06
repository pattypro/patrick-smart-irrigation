import streamlit as st
import gspread
from datetime import datetime, date

st.set_page_config(page_title="Patrick Smart Irrigation Dashboard v2.1", layout="wide")

# Authenticate with Google Sheets
gc = gspread.service_account_from_dict(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
ss = gc.open("Patrick_Irrigation_Log")

# Required sheets & headers
REQUIRED_SHEETS = {
    "MAIN": [
        "timestamp","plot_id","strategy","eto","kc","etc","forecast_rain",
        "vwc","camera","nvi","est_biom","suggested_liters",
        "meter_start","meter_end","applied_liters","action"
    ],
    "Weather_ETo": [
        "date","P_kPa","rain_mm","Tmean_C","Tmax_C","Tmin_C",
        "RHmean","u2_ms","n_hours","ETo_mm"
    ],
    "NDVI_Calibration": [
        "stage","a","b","sigma_rgn","sigma_ocn","updated_at"
    ],
    "App_Metadata": ["key","value"],
}

def ensure_sheet_structure(ss):
    ws_main = ss.sheet1
    headers = ws_main.row_values(1)
    if not headers:
        ws_main.append_row(REQUIRED_SHEETS["MAIN"])
    else:
        missing = [h for h in REQUIRED_SHEETS["MAIN"] if h not in headers]
        if missing:
            ws_main.update('1:1', [headers + missing])

    def ensure_tab(title, headers):
        try:
            ws = ss.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            ws = ss.add_worksheet(title=title, rows=1000, cols=len(headers))
            ws.append_row(headers)
            return ws
        current = ws.row_values(1)
        if not current:
            ws.append_row(headers)
        else:
            missing = [h for h in headers if h not in current]
            if missing:
                ws.update('1:1', [current + missing])
        return ws

    ws_weather = ensure_tab("Weather_ETo", REQUIRED_SHEETS["Weather_ETo"])
    ws_cal = ensure_tab("NDVI_Calibration", REQUIRED_SHEETS["NDVI_Calibration"])
    ws_meta = ensure_tab("App_Metadata", REQUIRED_SHEETS["App_Metadata"])

    # Seed calibration
    if not ws_cal.get_all_records():
        ws_cal.append_row(["initial",0.0,1.0,0.03,0.03,datetime.utcnow().isoformat()])
        ws_cal.append_row(["mid",0.0,1.0,0.03,0.03,datetime.utcnow().isoformat()])
        ws_cal.append_row(["late",0.0,1.0,0.03,0.03,datetime.utcnow().isoformat()])

    # Seed metadata
    if not ws_meta.get_all_records():
        ws_meta.append_row(["app_version","2.1"])
        ws_meta.append_row(["last_update",str(date.today())])
        ws_meta.append_row(["researcher","Patrick Habyarimana"])

    return ws_main, ws_weather, ws_cal, ws_meta

# Ensure all sheets exist
ws_main, ws_weather, ws_cal, ws_meta = ensure_sheet_structure(ss)

st.title("Patrick Smart Irrigation Dashboard v2.1")
st.success("✅ Google Sheet structure verified and updated successfully!")

st.write("Your sheet now includes all required tabs and headers for automated weather, NDVI, and irrigation data logging.")
