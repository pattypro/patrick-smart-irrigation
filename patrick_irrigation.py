import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Patrick Smart Irrigation Dashboard", layout="wide")

# ------------------------
# CONFIG (edit as needed)
# ------------------------
PLOT_AREA_M2 = 1.0
EFFICIENCY = 0.85
MAD = 0.20
FC = 30.0   # set your measured Field Capacity (% VWC)
PWP = 10.0  # set your measured Permanent Wilting Point (% VWC)

Kc_VALUES = {"initial": 0.55, "mid": 1.0, "late": 0.85}

# Example camera calibration coefficients (replace after field calibration)
camera_models = {
    "RGN": {"a": 0.0, "b": 100.0},  # biomass_per_plant_g = a + b * nVI
    "OCN": {"a": 0.0, "b": 100.0}
}

plots = {
    "Plot1_Manual": "Manual (baseline)",
    "Plot2_Sensor": "Sensor + Weather",
    "Plot3_VI": "VI + Weather",
    "Plot4_Combined": "Sensor + VI + Weather"
}

SHEET_NAME = "Patrick_Irrigation_Log"  # exact Google Sheet name

# ------------------------
# GOOGLE SHEETS FUNCTIONS
# ------------------------
@st.cache_resource
def connect_gsheet():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    # read JSON string from secrets and parse
    
    # ✅ Read JSON string from secrets and ensure it's a dict
    try:
        raw_secret = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
        if isinstance(raw_secret, str):
            creds_dict = json.loads(raw_secret)
        else:
            creds_dict = dict(raw_secret)  # Convert Streamlit AttrDict to regular dict
    except Exception as e:
        st.error(f"Google credentials not found or invalid: {e}")
        raise

    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1
    return sheet

def load_log():
    try:
        sheet = connect_gsheet()
        data = sheet.get_all_records()
        if len(data)==0:
            return pd.DataFrame(columns=["timestamp","plot_id","strategy","eto","kc","etc","forecast_rain","vwc","camera","nvi","est_biom","suggested_liters","meter_start","meter_end","applied_liters","action"])
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not load Google Sheet: {e}")
        return pd.DataFrame(columns=["timestamp","plot_id","strategy","eto","kc","etc","forecast_rain","vwc","camera","nvi","est_biom","suggested_liters","meter_start","meter_end","applied_liters","action"])

def save_log(row_dict):
    sheet = connect_gsheet()
    headers = ["timestamp","plot_id","strategy","eto","kc","etc","forecast_rain","vwc","camera","nvi","est_biom","suggested_liters","meter_start","meter_end","applied_liters","action"]
    row = [row_dict.get(h, "") for h in headers]
    sheet.append_row(row)

# ------------------------
# UTILS
# ------------------------
def compute_trigger_vwc(fc, pwp, mad):
    return fc - mad * (fc - pwp)

def etc_from_eto(eto, kc):
    return eto * kc

def irrigation_depth(etc, efficiency):
    return etc / efficiency

def liters_from_mm(mm, area_m2):
    return mm * area_m2

def biomass_estimate(camera_type, nvi):
    model = camera_models.get(camera_type, {"a":0,"b":0})
    return model["a"] + model["b"] * nvi

# ------------------------
# UI
# ------------------------
st.title("🌱 Patrick Smart Irrigation Dashboard")

tabs = st.tabs(["Data Entry","Analytics","Calibration"])

with tabs[0]:
    st.header("Data entry & irrigation recommendations")
    col1, col2, col3 = st.columns(3)
    with col1:
        eto = st.number_input("ETo (mm/day)", value=4.0, step=0.1)
    with col2:
        stage = st.selectbox("Crop Stage", list(Kc_VALUES.keys()))
        kc = Kc_VALUES[stage]
    with col3:
        forecast_rain = st.number_input("Predicted Rainfall (mm, 24h)", value=0.0, step=0.5)
    etc = etc_from_eto(eto, kc)
    st.markdown(f"**Calculated ETc (mm/day): {etc:.2f}**")

    st.markdown("---")
    st.subheader("Enter plot-level sensor & camera values (one row per plot)")

    entries = []
    for plot_id, strategy in plots.items():
        with st.expander(f"{plot_id} — {strategy}", expanded=False):
            vwc = st.number_input(f"{plot_id} — Soil Moisture VWC (%)", value=28.0, step=0.1, key=plot_id+"_vwc")
            camera_type = st.selectbox(f"{plot_id} — Camera Type", ["RGN","OCN"], key=plot_id+"_cam")
            nvi = st.number_input(f"{plot_id} — nVI Value", value=0.6, step=0.01, key=plot_id+"_nvi")

            est_biom = biomass_estimate(camera_type, nvi)

            trigger_vwc = compute_trigger_vwc(FC, PWP, MAD)
            vwc_trigger = (vwc <= trigger_vwc)

            irrigation_mm = irrigation_depth(etc, EFFICIENCY)
            suggested_liters = liters_from_mm(irrigation_mm, PLOT_AREA_M2)

            # Decision rules per strategy
            action = "No irrigation"
            if plot_id == "Plot1_Manual":
                action = "Manual decision (log only)"
            elif plot_id == "Plot2_Sensor":
                if forecast_rain >= 10:
                    action = "Skip (rain forecast)"
                elif vwc_trigger:
                    action = f"Irrigate {suggested_liters:.2f} L (sensor trigger)"
            elif plot_id == "Plot3_VI":
                if forecast_rain >= 10:
                    action = "Skip (rain forecast)"
                elif est_biom < 50:  # placeholder threshold; replace after calibration
                    action = f"Irrigate {suggested_liters:.2f} L (VI trigger)"
            elif plot_id == "Plot4_Combined":
                if forecast_rain >= 10:
                    action = "Skip (rain forecast)"
                elif vwc_trigger or est_biom < 50:
                    action = f"Irrigate {suggested_liters:.2f} L (combined trigger)"

            st.write(f"**Biomass est (per plant): {est_biom:.2f} g**")
            st.write(f"**Trigger VWC: {trigger_vwc:.2f}% | Current: {vwc:.2f}%**")
            st.info(f"Recommended Action: {action}")

            st.markdown("**Log irrigation event (if applied)**")
            meter_start = st.number_input(f"{plot_id} — Meter Start (L)", value=0.0, step=0.1, key=plot_id+"_start")
            meter_end = st.number_input(f"{plot_id} — Meter End (L)", value=0.0, step=0.1, key=plot_id+"_end")
            applied = meter_end - meter_start

            if st.button(f"Save Log for {plot_id}", key=plot_id+"_save"):
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "plot_id": plot_id,
                    "strategy": strategy,
                    "eto": eto,
                    "kc": kc,
                    "etc": etc,
                    "forecast_rain": forecast_rain,
                    "vwc": vwc,
                    "camera": camera_type,
                    "nvi": nvi,
                    "est_biom": round(est_biom,3),
                    "suggested_liters": round(suggested_liters,3),
                    "meter_start": meter_start,
                    "meter_end": meter_end,
                    "applied_liters": round(applied,3),
                    "action": action
                }
                try:
                    save_log(row)
                    st.success(f"Log saved for {plot_id} ✅")
                except Exception as e:
                    st.error(f"Failed to save log: {e}")

with tabs[1]:
    st.header("Analytics")
    df = load_log()
    if df.empty:
        st.info("No logs yet. Use Data Entry tab to save irrigation events.")
    else:
        # ensure correct dtypes
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['applied_liters'] = pd.to_numeric(df['applied_liters'], errors='coerce').fillna(0)
        df['est_biom'] = pd.to_numeric(df['est_biom'], errors='coerce').fillna(0)
        df = df.sort_values('timestamp')

        plot_choice = st.selectbox("Select plot for time-series", df['plot_id'].unique())
        df_plot = df[df['plot_id']==plot_choice].copy()

        # Soil moisture plot
        fig, ax = plt.subplots()
        ax.plot(df_plot['timestamp'], pd.to_numeric(df_plot['vwc'], errors='coerce'), marker='o', label='VWC (%)')
        ax.axhline(compute_trigger_vwc(FC,PWP,MAD), color='r', linestyle='--', label='Trigger VWC')
        ax.set_ylabel("VWC (%)")
        ax.set_title(f"Soil Moisture — {plot_choice}")
        ax.legend()
        st.pyplot(fig)

        # Biomass per plant trend
        fig, ax = plt.subplots()
        ax.plot(df_plot['timestamp'], df_plot['est_biom'], marker='o', color='green')
        ax.set_ylabel("Biomass per plant (g)")
        ax.set_title(f"Biomass Estimate — {plot_choice}")
        st.pyplot(fig)

        # cumulative applied
        df_plot['cum_applied'] = df_plot['applied_liters'].cumsum()
        fig, ax = plt.subplots()
        ax.plot(df_plot['timestamp'], df_plot['cum_applied'], marker='o', color='blue')
        ax.set_ylabel("Cumulative applied (L)")
        ax.set_title(f"Cumulative Water Applied — {plot_choice}")
        st.pyplot(fig)

        # WUE = (est_biom * 6 plants) / cum_applied
        df_plot['wue'] = (df_plot['est_biom'] * 6) / df_plot['cum_applied'].replace(0, pd.NA)
        fig, ax = plt.subplots()
        ax.plot(df_plot['timestamp'], df_plot['wue'], marker='o', color='purple')
        ax.set_ylabel("WUE (g per L)")
        ax.set_title(f"WUE — {plot_choice}")
        st.pyplot(fig)

        st.markdown("### Raw data (last 50 rows)")
        st.dataframe(df.tail(50))

with tabs[2]:
    st.header("Calibration helpers")
    st.markdown("Use this section to paste camera-model coefficients after running your field calibration.")
    st.markdown("**Current camera models (edit in code or upload a JSON later):**")
    st.json(camera_models)
    st.markdown("If you need a CSV template for calibration samples, download below:")
    if st.button("Download calibration CSV template"):
        df_tmp = pd.DataFrame(columns=['timestamp','sample_id','plot_id','camera_type','nvi','ratio','gci','spad','height_cm','leaf_count','fresh_g','dry_g','notes'])
        st.download_button("Download CSV", df_tmp.to_csv(index=False).encode('utf-8'), file_name='calibration_template.csv', mime='text/csv')
    st.info("Calibration: collect 12-20 paired samples across canopy conditions. Fit biomass_per_plant = a + b * nVI per camera and paste coefficients into the app code or a config file.")
