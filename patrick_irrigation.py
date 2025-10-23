# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gspread, json, math
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta, date
from pathlib import Path

st.set_page_config(page_title="Patrick Smart Irrigation - v3.7 (Auto)", layout="wide")
PRIMARY = "#0EA5E9"; ACCENT="#22C55E"; WARN="#F59E0B"; MUTED="#6B7280"
st.markdown(f"""
<style>
.block-container {{ padding-top: 0.8rem; }}
h1, h2, h3 {{ color: #0f172a; margin-top: .5rem; }}
.info-box {{ border-left: 6px solid {PRIMARY}; padding:.6rem .8rem; background:#f0f9ff; border-radius:8px; margin:.4rem 0 1rem 0; }}
.small-note {{ font-size:.92rem; color:{MUTED}; }}
.stButton>button {{ border-radius: 12px; padding:.45rem .9rem; font-weight:600; }}
</style>
""", unsafe_allow_html=True)

banner = Path("assets/header_smart_irrigation.png")
if banner.exists():
    st.image(str(banner), use_column_width=True)
st.title("Patrick Smart Irrigation - v3.7 (Auto)")

SPREADSHEET_NAME = "Patrick_Irrigation_Log"
WEATHER_SHEET = "Weather_ETo"
CALIB_SHEET="NDVI_Calibration"
META_SHEET="App_Metadata"

CROP_PARAMS = {
    "Spinach": {"Kc_ini":0.70, "Kc_mid":1.05, "Kc_end":0.95, "root_depth":0.20, "base_ETc_adj":1.10},
    "Lettuce": {"Kc_ini":0.65, "Kc_mid":1.00, "Kc_end":0.90, "root_depth":0.25, "base_ETc_adj":1.15},
    "Tomato":  {"Kc_ini":0.60, "Kc_mid":1.15, "Kc_end":0.85, "root_depth":0.45, "base_ETc_adj":1.05},
    "Maize":   {"Kc_ini":0.45, "Kc_mid":1.20, "Kc_end":0.80, "root_depth":0.60, "base_ETc_adj":1.00},
    "Beans":   {"Kc_ini":0.50, "Kc_mid":1.05, "Kc_end":0.85, "root_depth":0.35, "base_ETc_adj":1.00},
}
AREA = 1.0
DEFAULTS={"FC":30.0,"PWP":10.0,"MAD":0.20,"EFFICIENCY":0.85}
RAIN_SKIP_MM = 10.0

def connect_gsheet():
    scope=[
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    raw = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
    creds_dict = json.loads(raw) if isinstance(raw,str) else dict(raw)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    try:
        ss = client.open_by_key(st.secrets["SHEET_ID"]) if "SHEET_ID" in st.secrets else client.open(SPREADSHEET_NAME)
        st.sidebar.success("Connected to Google Sheet")
    except gspread.SpreadsheetNotFound:
        st.error("Sheet not found. Share your sheet with the service account and/or set SHEET_ID in secrets.")
        raise
    return client, ss

REQUIRED_SHEETS={
    "MAIN":[
        "timestamp","treatment","crop","stage",
        "eto_yday","eto_next","kc","Kndvi","PWRF","Ks_soil",
        "ETc_yday","ETc_next",
        "rain_yday","forecast_rain","eff_rain_y","eff_rain_next",
        "vwc_predawn",
        "ndvi_rgn","ndvi_ocn","nvi",
        "TAW","RAW","depletion_prev","depletion_start","need_mm",
        "suggested_liters","applied_liters","decision"
    ],
    WEATHER_SHEET:["date","ETo_mm","rain_mm"],
    CALIB_SHEET:["stage","a","b","sigma_rgn","sigma_ocn","updated_at"],
    META_SHEET:["key","value"],
}

def ensure_sheet_structure(ss):
    ws_main = ss.sheet1
    headers = ws_main.row_values(1)
    if not headers: ws_main.append_row(REQUIRED_SHEETS["MAIN"])
    else:
        missing=[h for h in REQUIRED_SHEETS["MAIN"] if h not in headers]
        if missing: ws_main.update('1:1',[headers+missing])
    def ensure_tab(title, headers):
        try: ws = ss.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            ws = ss.add_worksheet(title=title, rows=3000, cols=max(10,len(headers))); ws.append_row(headers); return ws
        cur = ws.row_values(1)
        if not cur: ws.append_row(headers)
        else:
            miss = [h for h in headers if h not in cur]
            if miss: ws.update('1:1',[cur+miss])
        return ws
    ws_weather=ensure_tab(WEATHER_SHEET, REQUIRED_SHEETS[WEATHER_SHEET])
    ws_cal=ensure_tab(CALIB_SHEET, REQUIRED_SHEETS[CALIB_SHEET])
    ws_meta=ensure_tab(META_SHEET, REQUIRED_SHEETS[META_SHEET])
    if not ws_cal.get_all_records():
        for stg in ["initial","mid","late"]:
            ws_cal.append_row([stg,0.0,1.0,0.03,0.03,datetime.utcnow().isoformat()])
    if not ws_meta.get_all_records():
        ws_meta.append_row(["app_version","3.7 (Auto)"]); ws_meta.append_row(["last_update", str(date.today())]); ws_meta.append_row(["researcher","Patrick"])
    return ws_main, ws_weather, ws_cal, ws_meta

def sheet_to_df(ws):
    data = ws.get_all_records()
    return pd.DataFrame(data) if data else pd.DataFrame()

def append_row(ws, dct, header_order):
    ws.append_row([dct.get(h,"") for h in header_order])

def saturation_vapor_pressure(T): return 0.6108 * math.exp((17.27*T)/(T+237.3))
def slope_vapor_pressure_curve(T): es = saturation_vapor_pressure(T); return 4098*es/((T+237.3)**2)
def psychrometric_constant(P_kPa): return 0.000665 * P_kPa

def eto_hourly(Tair, RH_pct, Rs_MJ_m2_h, u2, P_kPa):
    es = saturation_vapor_pressure(Tair)
    ea = es * max(min(RH_pct/100.0, 1.0), 0.0)
    delta = slope_vapor_pressure_curve(Tair)
    gamma = psychrometric_constant(P_kPa)
    albedo = 0.23
    Rns = (1 - albedo) * max(Rs_MJ_m2_h, 0.0)
    G = 0.1 * Rns if Rns > 0 else 0.0
    Rn = Rns
    num = 0.408*delta*(Rn - G) + gamma*(37.0/(Tair+273.0))*u2*(es - ea)
    den = delta + gamma*(1 + 0.34*u2)
    eto = num/max(den,1e-6)
    return max(0.0, eto)

def ndvi_fuse(ndvi_rgn, ndvi_ocn, a=0.0, b=1.0, sigma_rgn=0.03, sigma_ocn=0.03):
    r = None if ndvi_rgn is None or ndvi_rgn<=0 else float(ndvi_rgn)
    o = None if ndvi_ocn is None or ndvi_ocn<=0 else float(ndvi_ocn)
    ocn_rg = a + b*o if o is not None else None
    if r is not None and ocn_rg is not None:
        w_r = 1.0/(sigma_rgn**2); w_o = 1.0/(sigma_ocn**2)
        s = max(w_r + w_o, 1e-6); w_r/=s; w_o/=s
        return float(np.clip(w_r*r + w_o*ocn_rg, 0, 1))
    elif r is not None: return float(np.clip(r,0,1))
    elif ocn_rg is not None: return float(np.clip(ocn_rg,0,1))
    else: return None

def ndvi_to_canopy_factor(nvi):
    if nvi is None: return 1.0
    return float(np.clip(0.50 + 0.50*float(nvi), 0.60, 1.10))

st.sidebar.header("Settings")
crop_choice = st.sidebar.selectbox("Crop", list(CROP_PARAMS.keys()), index=0)
stage = st.sidebar.selectbox("Stage", ["initial","mid","late"])
FC = st.sidebar.number_input("Field Capacity (VWC %)", value=DEFAULTS["FC"], step=0.5)
PWP = st.sidebar.number_input("Permanent Wilting Point (VWC %)", value=DEFAULTS["PWP"], step=0.5)
MAD = st.sidebar.number_input("Management Allowable Depletion (0-1)", value=DEFAULTS["MAD"], step=0.05, min_value=0.05, max_value=0.6)
EFF = st.sidebar.number_input("Irrigation Efficiency (0-1)", value=DEFAULTS["EFFICIENCY"], step=0.05, min_value=0.5, max_value=1.0)
st.sidebar.caption("Plot area fixed at 1.0 m² → 1 mm = 1 L.")

try:
    client, ss = connect_gsheet()
    ws_main, ws_weather, ws_cal, ws_meta = ensure_sheet_structure(ss)
    connected = True
except Exception as e:
    st.error(f"Failed to connect to Google Sheet: {e}")
    connected = False

tabs = st.tabs(["Treatment Dashboard", "Analytics"])

with tabs[0]:
    st.subheader("Treatment Dashboard")
    params = CROP_PARAMS[crop_choice]
    kc_base = {"initial":params["Kc_ini"], "mid":params["Kc_mid"], "late":params["Kc_end"]}[stage]
    Zr = params["root_depth"]; PWRF = params["base_ETc_adj"]
    TAW = (FC - PWP) * Zr * 10.0
    RAW = MAD * TAW
    trigger_vwc = FC - MAD*(FC - PWP)

    treatment = st.radio("Select Treatment", ["1","2","3","4"], horizontal=True,
                         format_func=lambda k: {"1":"T1 - Baseline (Manual)",
                                                "2":"T2 - Sensor + Weather",
                                                "3":"T3 - NDVI + Weather",
                                                "4":"T4 - Sensor + NDVI + Weather"}[k])

    up_w = st.file_uploader("Weather CSV (hourly): timestamp, Tair_C, RH, wind_ms, solar_MJ_m2, P_kPa, (optional) rain_mm", type=["csv"], key="wx")
    up_s = st.file_uploader("Sensor CSV (hourly): timestamp, vwc (%)", type=["csv"], key="sens") if treatment in ("2","4") else None
    ndvi_rgn = ndvi_ocn = None
    if treatment in ("3","4"):
        c1,c2 = st.columns(2)
        with c1: ndvi_rgn = st.number_input("NDVI_RGN", value=0.0, step=0.01, min_value=0.0, max_value=1.0)
        with c2: ndvi_ocn = st.number_input("NDVI_OCN", value=0.0, step=0.01, min_value=0.0, max_value=1.0)
    forecast_rain = st.number_input("Rain forecast next 24h (mm)", value=0.0, step=0.5) if treatment in ("2","3","4") else 0.0

    logic_boxes = {
        "1": "💧 Logic: Manual entry only. You record applied liters. No automatic computation.",
        "2": "💧 Logic: 1) ETo from weather CSV, 2) Predawn VWC (4–6 AM) from sensor CSV, "
             "3) ETc = ETo × Kc × PWRF × Ks_soil, 4) If VWC ≤ trigger or depletion ≥ RAW → Irrigate. "
             "Rain skip if forecast ≥ 10 mm.",
        "3": "🌱 Logic: 1) ETo from weather CSV, 2) Harmonize NDVI_RGN + NDVI_OCN → nVI, "
             "3) ETc = ETo × Kc × PWRF × Kndvi, 4) If nVI < 0.55 or depletion ≥ RAW → Irrigate. "
             "Rain skip if forecast ≥ 10 mm.",
        "4": "🌿 Logic: 1) ETo from weather CSV, 2) Harmonize NDVI → nVI, 3) Predawn VWC from sensor CSV, "
             "4) ETc = ETo × Kc × PWRF × Kndvi × Ks_soil, 5) Irrigate if any trigger true. "
             "Rain skip if forecast ≥ 10 mm."
    }
    st.markdown(f"<div class='info-box'>{logic_boxes[treatment]}</div>", unsafe_allow_html=True)

    # WEATHER handling
    eto_yday = 0.0; eto_next = 0.0; rain_yday = 0.0
    if up_w is not None:
        wx = pd.read_csv(up_w)
        def find_col(df, names):
            cols = {c.lower(): c for c in df.columns}
            for n in names:
                if n.lower() in cols: return cols[n.lower()]
            for c in df.columns:
                for n in names:
                    if n.lower() in c.lower(): return c
            return None
        c_ts = find_col(wx, ["timestamp","datetime","time"])
        c_T = find_col(wx, ["Tair_C","T","Temp_C","Temperature"])
        c_RH = find_col(wx, ["RH","Humidity","RH_%"])
        c_u2 = find_col(wx, ["wind_ms","u2","wind speed"])
        c_Rs = find_col(wx, ["solar_MJ_m2","Rs","Solar","Solar_Radiation"])
        c_P = find_col(wx, ["P_kPa","Pressure_kPa","Pressure"])
        c_rain = find_col(wx, ["rain_mm","rain","precipitation"])
        ok = all([c_ts, c_T, c_RH, c_u2, c_Rs, c_P])
        if not ok:
            st.error("Weather CSV missing columns. Need: timestamp, Tair_C, RH, wind_ms, solar_MJ_m2, P_kPa.")
        else:
            wx[c_ts] = pd.to_datetime(wx[c_ts], errors="coerce")
            wx = wx.dropna(subset=[c_ts]).sort_values(c_ts).reset_index(drop=True)
            T = pd.to_numeric(wx[c_T], errors="coerce")
            RH = pd.to_numeric(wx[c_RH], errors="coerce")
            u2 = pd.to_numeric(wx[c_u2], errors="coerce")
            Rs = pd.to_numeric(wx[c_Rs], errors="coerce")
            P = pd.to_numeric(wx[c_P], errors="coerce"); 
            if P.mean()>20: P = P*0.1
            eto_hourly_vals = []
            for i in range(len(wx)):
                eto_h = (0.0 if pd.isna(T.iloc[i]) or pd.isna(RH.iloc[i]) or pd.isna(Rs.iloc[i]) or pd.isna(u2.iloc[i]) or pd.isna(P.iloc[i])
                         else eto_hourly(float(T.iloc[i]), float(RH.iloc[i]), float(Rs.iloc[i]), float(u2.iloc[i]), float(P.iloc[i])))
                eto_hourly_vals.append(eto_h)
            wx["ETo_hourly_mm"] = eto_hourly_vals
            wx["date"] = pd.to_datetime(wx[c_ts]).dt.date
            daily = wx.groupby("date", as_index=False)["ETo_hourly_mm"].sum().rename(columns={"ETo_hourly_mm":"ETo_mm"})
            if c_rain:
                rain_series = pd.to_numeric(wx[c_rain], errors="coerce").fillna(0.0)
                rain_daily = wx.groupby("date", as_index=False)[c_rain].sum().rename(columns={c_rain:"rain_mm"})
                daily = daily.merge(rain_daily, on="date", how="left")
            else:
                daily["rain_mm"] = 0.0

            today = datetime.utcnow().date()
            yday = today - timedelta(days=1)
            if not daily.empty:
                if yday in set(daily["date"]):
                    eto_yday = float(daily.loc[daily["date"]==yday,"ETo_mm"].iloc[0])
                    rain_yday = float(daily.loc[daily["date"]==yday,"rain_mm"].iloc[0])
                if today in set(daily["date"]):
                    eto_next = float(daily.loc[daily["date"]==today,"ETo_mm"].iloc[0])
                else:
                    eto_next = eto_yday

            if connected:
                ex = sheet_to_df(ws_weather)
                ex_dates = set(pd.to_datetime(ex["date"], errors="coerce").dt.date) if not ex.empty and "date" in ex.columns else set()
                for _,r in daily.iterrows():
                    if r["date"] not in ex_dates:
                        append_row(ws_weather, {"date": str(r["date"]), "ETo_mm": float(r["ETo_mm"]), "rain_mm": float(r["rain_mm"])}, ["date","ETo_mm","rain_mm"])
                st.success("Weather_ETo updated with computed ETo.")

    # SENSOR handling
    vwc_predawn = None
    if up_s is not None:
        df = pd.read_csv(up_s)
        ts_col = None
        for cand in ["timestamp","time","datetime","date_time"]:
            if cand in [c.lower() for c in df.columns]:
                for c in df.columns:
                    if c.lower()==cand: ts_col=c; break
                break
        if ts_col is None:
            st.error("Sensor CSV needs a 'timestamp' column.")
        else:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
            vwc_col = None
            for cand in ["vwc","soil_moisture","soil_vwc"]:
                for c in df.columns:
                    if c.lower()==cand: vwc_col=c; break
                if vwc_col: break
            if vwc_col is None:
                st.error("Sensor CSV missing a VWC column (e.g., 'vwc').")
            else:
                predawn = df.between_time("04:00","06:00")[vwc_col]
                if not predawn.empty:
                    vwc_predawn = float(predawn.mean())
                    st.metric("Predawn VWC (%)", f"{vwc_predawn:.1f}")

    # NDVI fusion (T3/T4)
    nvi = None; Kndvi = 1.0
    if treatment in ("3","4"):
        a=b=1.0; sr=so=0.03; a=0.0
        if connected:
            try:
                dfcal = sheet_to_df(ws_cal)
                if not dfcal.empty:
                    row = dfcal[dfcal["stage"].str.lower()==stage]
                    if not row.empty:
                        a=float(row["a"].iloc[0]); b=float(row["b"].iloc[0])
                        sr=float(row["sigma_rgn"].iloc[0]); so=float(row["sigma_ocn"].iloc[0])
            except: pass
        ndvi_r = ndvi_rgn if ndvi_rgn and ndvi_rgn>0 else None
        ndvi_o = ndvi_ocn if ndvi_ocn and ndvi_ocn>0 else None
        nvi = ndvi_fuse(ndvi_r, ndvi_o, a=a,b=b,sigma_rgn=sr,sigma_ocn=so)
        if nvi is not None: st.metric("Harmonized NDVI (nVI)", f"{nvi:.3f}")
        Kndvi = ndvi_to_canopy_factor(nvi)

    # Decision calculation
    eff_rain_y = 0.8*(rain_yday if up_w is not None else 0.0)
    eff_rain_next = 0.8*(forecast_rain if treatment in ("2","3","4") else 0.0)
    Ks_soil = 1.0
    if treatment in ("2","4") and vwc_predawn is not None and FC>0:
        Ks_soil = float(np.clip(vwc_predawn/FC, 0.3, 1.0))

    ETc_yday = (eto_yday * kc_base * Kndvi * params["base_ETc_adj"] * (Ks_soil if treatment in ("2","4") else 1.0))
    ETc_next = (eto_next * kc_base * Kndvi * params["base_ETc_adj"] * (Ks_soil if treatment in ("2","4") else 1.0))

    depletion_prev = st.number_input("Depletion_prev (mm)", value=float(RAW/2.0), step=0.5)
    depletion_start = max(0.0, depletion_prev + ETc_yday - eff_rain_y)
    need_mm = min(max(0.0, depletion_start + ETc_next - eff_rain_next), TAW)
    suggested_liters = need_mm * AREA / max(DEFAULTS["EFFICIENCY"], 1e-6)

    if treatment == "1":
        st.info("T1 - Manual logging only.")
        decision = "Manual entry"
    elif treatment == "2":
        trigger = ((vwc_predawn is not None and vwc_predawn <= (FC - MAD*(FC-PWP))) or (depletion_start >= RAW))
        decision = "Skip (Rain forecast)" if (forecast_rain >= RAIN_SKIP_MM) else ("Irrigate" if trigger and suggested_liters>=0.5 else "Skip")
    elif treatment == "3":
        trigger = ((nvi is not None and nvi < 0.55) or (depletion_start >= RAW))
        decision = "Skip (Rain forecast)" if (forecast_rain >= RAIN_SKIP_MM) else ("Irrigate" if trigger and suggested_liters>=0.5 else "Skip")
    else:
        trigger = ((vwc_predawn is not None and vwc_predawn <= (FC - MAD*(FC-PWP))) or (depletion_start >= RAW) or (nvi is not None and nvi < 0.55))
        decision = "Skip (Rain forecast)" if (forecast_rain >= RAIN_SKIP_MM) else ("Irrigate" if trigger and suggested_liters>=0.5 else "Skip")

    if treatment != "1":
        if "Irrigate" in decision:
            st.success(f"Decision: {decision} — Apply **{suggested_liters:.2f} L** per 1 m² plot")
        else:
            st.warning(f"Decision: {decision} — Suggested (calc) = {suggested_liters:.2f} L")

    applied_liters = st.number_input("Applied liters (today)", value=0.0, step=0.1)

    if connected and st.button("Save to Google Sheet"):
        headers = REQUIRED_SHEETS["MAIN"]
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "treatment": f"T{treatment}",
            "crop": crop_choice, "stage": stage,
            "eto_yday": round(eto_yday,3), "eto_next": round(eto_next,3),
            "kc": round(kc_base,3), "Kndvi": round(ndvi_to_canopy_factor(nvi),3) if treatment in ("3","4") else round(1.0,3),
            "PWRF": round(params['base_ETc_adj'],3),
            "Ks_soil": round(Ks_soil,3),
            "ETc_yday": round(ETc_yday,3), "ETc_next": round(ETc_next,3),
            "rain_yday": round(rain_yday,3), "forecast_rain": round(forecast_rain,3),
            "eff_rain_y": round(eff_rain_y,3), "eff_rain_next": round(eff_rain_next,3),
            "vwc_predawn": vwc_predawn if vwc_predawn is not None else "",
            "ndvi_rgn": ndvi_rgn if ndvi_rgn is not None else "",
            "ndvi_ocn": ndvi_ocn if ndvi_ocn is not None else "",
            "nvi": round(nvi,3) if nvi is not None else "",
            "TAW": round(TAW,2), "RAW": round(RAW,2),
            "depletion_prev": round(depletion_prev,2), "depletion_start": round(depletion_start,2),
            "need_mm": round(need_mm,2),
            "suggested_liters": round(suggested_liters,2),
            "applied_liters": round(applied_liters,2),
            "decision": decision
        }
        append_row(ws_main, row, headers); st.success("Saved ✅")

with tabs[1]:
    st.subheader("Analytics")
    if not connected:
        st.info("Connect Google Sheets to view analytics.")
    else:
        df = sheet_to_df(ws_main)
        if df.empty: st.info("No logs yet.")
        else:
            if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            for c in ["vwc_predawn","nvi","applied_liters","eto_yday","eto_next","ETc_next","suggested_liters","forecast_rain"]:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.sort_values("timestamp")

            ftr = st.selectbox("Filter by treatment", options=["All","T1","T2","T3","T4"], index=0)
            dff = df if ftr=="All" else df[df["treatment"]==ftr]

            if "suggested_liters" in dff.columns and dff["suggested_liters"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff["timestamp"], dff["suggested_liters"].fillna(0), marker="o", label="Suggested liters")
                if "ETc_next" in dff.columns: ax.plot(dff["timestamp"], dff["ETc_next"].fillna(0), linestyle="--", label="ETc (mm) next")
                if "forecast_rain" in dff.columns: ax.bar(dff["timestamp"], (0.8*dff["forecast_rain"]).fillna(0), alpha=0.3, label="Eff. rain next (mm)")
                ax.set_title("Decision Components & Suggested Liters")
                ax.grid(True, alpha=0.3); ax.legend()
                st.pyplot(fig)

            dff_vwc = dff[dff["treatment"].isin(["T2","T4"])] if "treatment" in dff.columns else pd.DataFrame()
            if not dff_vwc.empty and dff_vwc["vwc_predawn"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff_vwc["timestamp"], dff_vwc["vwc_predawn"], marker="o")
                ax.axhline(DEFAULTS["FC"] - DEFAULTS["MAD"]*(DEFAULTS["FC"]-DEFAULTS["PWP"]), linestyle="--", label="Trigger VWC (default)")
                ax.set_ylabel("VWC (%)"); ax.set_title("Predawn Soil Moisture (T2 & T4)")
                ax.legend(); ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            dff_ndvi = dff[dff["treatment"].isin(["T3","T4"])] if "treatment" in dff.columns else pd.DataFrame()
            if not dff_ndvi.empty and dff_ndvi["nvi"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff_ndvi["timestamp"], dff_ndvi["nvi"], marker="o")
                ax.set_ylabel("Harmonized NDVI"); ax.set_title("Canopy NDVI (fused)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
