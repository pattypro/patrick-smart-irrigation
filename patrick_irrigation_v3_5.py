# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gspread, json, math
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta, date, time
from pathlib import Path

st.set_page_config(page_title="Patrick Smart Irrigation - v3.5", layout="wide")
PRIMARY = "#0EA5E9"; ACCENT="#22C55E"; WARN="#F59E0B"; MUTED="#6B7280"
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.0rem; }}
h1, h2, h3 {{ color: #0f172a; margin-top: .5rem; }}
.section-title {{ font-size: 1.15rem; font-weight: 700; margin: .35rem 0 .6rem 0; }}
.stTabs [data-baseweb="tab-list"] button {{ font-weight: 700; padding: 8px 14px; gap: 8px; }}
.stTabs [data-baseweb="tab"] {{ font-size: 0.98rem; }}
.stButton>button {{ border-radius: 12px; padding: .45rem .9rem; font-weight: 600; }}
.small-note {{ font-size:.92rem; color:{MUTED}; }}
</style>
""", unsafe_allow_html=True)

banner = Path("assets/header_smart_irrigation.png")
if banner.exists():
    st.image(str(banner), use_column_width=True)
st.title("Patrick Smart Irrigation - v3.5")

SPREADSHEET_NAME = "Patrick_Irrigation_Log"
WEATHER_SHEET = "Weather_ETo"
CALIB_SHEET="NDVI_Calibration"
META_SHEET="App_Metadata"
TREATMENT_TECH = {
    "1":"T1 - Baseline (Manual Only)",
    "2":"T2 - Sensor + Weather",
    "3":"T3 - NDVI + Weather",
    "4":"T4 - Sensor + NDVI + Weather",
}
CROP_PARAMS = {
    "Spinach": {"Kc_ini":0.70, "Kc_mid":1.05, "Kc_end":0.95, "root_depth":0.20, "base_ETc_adj":1.10},
    "Maize":   {"Kc_ini":0.45, "Kc_mid":1.20, "Kc_end":0.80, "root_depth":0.60, "base_ETc_adj":1.00},
    "Tomato":  {"Kc_ini":0.60, "Kc_mid":1.15, "Kc_end":0.85, "root_depth":0.45, "base_ETc_adj":1.05},
    "Lettuce": {"Kc_ini":0.65, "Kc_mid":1.00, "Kc_end":0.90, "root_depth":0.25, "base_ETc_adj":1.15},
    "Beans":   {"Kc_ini":0.50, "Kc_mid":1.05, "Kc_end":0.85, "root_depth":0.35, "base_ETc_adj":1.00},
}
DEFAULTS={"FC":30.0,"PWP":10.0,"MAD":0.20,"EFFICIENCY":0.85,"PLOT_AREA_M2":1.0}

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
        "suggested_liters","meter_start","meter_end","applied_liters","action"
    ],
    "Weather_ETo":["date","timestamp","P_kPa","rain_mm","Tmean_C","Tmax_C","Tmin_C","RHmean","u2_ms","n_hours","ETo_mm","ETo_hourly_mm"],
    "NDVI_Calibration":["stage","a","b","sigma_rgn","sigma_ocn","updated_at"],
    "App_Metadata":["key","value"],
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
            ws = ss.add_worksheet(title=title, rows=4000, cols=max(26,len(headers))); ws.append_row(headers); return ws
        cur = ws.row_values(1)
        if not cur: ws.append_row(headers)
        else:
            miss = [h for h in headers if h not in cur]
            if miss: ws.update('1:1',[cur+miss])
        return ws
    ws_weather=ensure_tab("Weather_ETo", REQUIRED_SHEETS["Weather_ETo"])
    ws_cal=ensure_tab("NDVI_Calibration", REQUIRED_SHEETS["NDVI_Calibration"])
    ws_meta=ensure_tab("App_Metadata", REQUIRED_SHEETS["App_Metadata"])
    if not ws_cal.get_all_records():
        for stg in ["initial","mid","late"]:
            ws_cal.append_row([stg,0.0,1.0,0.03,0.03,datetime.utcnow().isoformat()])
    if not ws_meta.get_all_records():
        ws_meta.append_row(["app_version","3.5"]); ws_meta.append_row(["last_update", str(date.today())]); ws_meta.append_row(["researcher","Patrick"])
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

def ndvi_to_canopy_factor(ndvi_final):
    if ndvi_final is None: return 1.0
    return float(np.clip(0.50 + 0.50*ndvi_final, 0.60, 1.10))

st.sidebar.header("Settings")
FC = st.sidebar.number_input("Field Capacity (VWC %)", value=DEFAULTS["FC"], step=0.5)
PWP = st.sidebar.number_input("Permanent Wilting Point (VWC %)", value=DEFAULTS["PWP"], step=0.5)
MAD = st.sidebar.number_input("Management Allowable Depletion (0-1)", value=DEFAULTS["MAD"], step=0.05, min_value=0.05, max_value=0.6)
EFF = st.sidebar.number_input("Irrigation Efficiency (0-1)", value=DEFAULTS["EFFICIENCY"], step=0.05, min_value=0.5, max_value=1.0)
AREA = st.sidebar.number_input("Plot Area (m^2)", value=DEFAULTS["PLOT_AREA_M2"], step=0.1, min_value=0.1)
crop_choice = st.sidebar.selectbox("Crop Type", options=list(CROP_PARAMS.keys()), index=list(CROP_PARAMS.keys()).index("Spinach"))
st.sidebar.markdown("<div class='small-note'>T1 is manual-only. T2-T4 auto-compute from selected sources.</div>", unsafe_allow_html=True)

try:
    client, ss = connect_gsheet()
    ws_main, ws_weather, ws_cal, ws_meta = ensure_sheet_structure(ss)
    connected = True
except Exception as e:
    st.error(f"Failed to connect to Google Sheet: {e}")
    connected = False

tabs = st.tabs(["Treatment Dashboard", "Sensor Data", "Weather -> ETo (Hourly/Daily)", "Analytics", "NDVI Harmonization"])

with tabs[0]:
    st.subheader("Treatment Dashboard")
    eto_yday_default, rain_yday_default, eto_next_default = 0.0, 0.0, 0.0
    if connected:
        try:
            wdf = sheet_to_df(ws_weather)
            if not wdf.empty and "date" in wdf.columns:
                wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce")
                yday = (datetime.utcnow().date() - timedelta(days=1))
                match = wdf[wdf["date"].dt.date == yday]
                if not match.empty:
                    eto_yday_default = float(pd.to_numeric(match.get("ETo_mm", pd.Series([0.0])), errors="coerce").iloc[0] or 0.0)
                    rain_yday_default = float(pd.to_numeric(match.get("rain_mm", pd.Series([0.0])), errors="coerce").iloc[0] or 0.0)
                    eto_next_default = eto_yday_default
        except Exception:
            pass

    c1, c2, c3 = st.columns(3)
    with c1:
        treatment = st.radio("Select Treatment", ["1","2","3","4"], horizontal=True, format_func=lambda k: TREATMENT_TECH[k])
    with c2:
        stage = st.selectbox("Crop Stage", ["initial","mid","late"])
    with c3:
        st.caption("Only relevant inputs appear depending on treatment")

    Kc_ini = CROP_PARAMS[crop_choice]["Kc_ini"]
    Kc_mid = CROP_PARAMS[crop_choice]["Kc_mid"]
    Kc_end = CROP_PARAMS[crop_choice]["Kc_end"]
    Zr = CROP_PARAMS[crop_choice]["root_depth"]
    PWRF = CROP_PARAMS[crop_choice]["base_ETc_adj"]
    kc_stage_map = {"initial":Kc_ini,"mid":Kc_mid,"late":Kc_end}
    kc_base = kc_stage_map[stage]

    TAW = (FC - PWP) * Zr * 10.0
    RAW = MAD * TAW
    trigger_vwc = FC - MAD*(FC - PWP)

    eto_yday = st.number_input("ETo (yesterday, mm)", value=float(eto_yday_default), step=0.1)
    eto_next = st.number_input("ETo forecast (next 24h, mm)", value=float(eto_next_default), step=0.1)
    rain_yday = st.number_input("Rain (yesterday, mm)", value=float(rain_yday_default), step=0.5)
    forecast_rain = st.number_input("Rain forecast next 24h (mm)", value=0.0, step=0.5)

    vwc_predawn, ndvi_rgn, ndvi_ocn, nvi_final = None, None, None, None
    Ks_soil, Kndvi = 1.0, 1.0

    if treatment == "1":
        st.info("T1 - Manual baseline entry (no automation)")
        meter_start = st.number_input("Meter Start (L)", value=0.0, step=0.1)
        meter_end = st.number_input("Meter End (L)", value=0.0, step=0.1)
        applied = max(0.0, meter_end - meter_start)
        st.success(f"Applied today: {applied:.2f} L")
        if connected and st.button("Save T1 manual entry"):
            headers = REQUIRED_SHEETS["MAIN"]
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "treatment":"T1","crop": crop_choice,"stage": stage,
                "meter_start": meter_start,"meter_end": meter_end,"applied_liters": round(applied,2),
                "action":"Manual irrigation"
            }
            append_row(ws_main, row, headers); st.success("Saved T1")
    else:
        if treatment in ("2","4"):
            vwc_predawn = st.number_input("Predawn VWC (%, 04-06h)", value=28.0, step=0.1)
            Ks_soil = float(np.clip((vwc_predawn/FC) if FC>0 else 1.0, 0.3, 1.0))
        if treatment in ("3","4"):
            ndvi_rgn = st.number_input("NDVI_RGN", value=0.0, step=0.01, min_value=0.0, max_value=1.0)
            ndvi_ocn = st.number_input("NDVI_OCN", value=0.0, step=0.01, min_value=0.0, max_value=1.0)
            a,b,sr,so = 0.0,1.0,0.03,0.03
            if connected:
                try:
                    dfcal = sheet_to_df(ws_cal)
                    if not dfcal.empty:
                        row = dfcal[dfcal["stage"].str.lower()==stage]
                        if not row.empty:
                            a=float(row["a"].iloc[0]); b=float(row["b"].iloc[0])
                            sr=float(row["sigma_rgn"].iloc[0]); so=float(row["sigma_ocn"].iloc[0])
                except: pass
            nvi_final = ndvi_fuse(ndvi_rgn if ndvi_rgn>0 else None, ndvi_ocn if ndvi_ocn>0 else None, a=a,b=b,sigma_rgn=sr,sigma_ocn=so)
            if nvi_final is not None: st.metric("Harmonized NDVI (nVI)", f"{nvi_final:.3f}")
            Kndvi = ndvi_to_canopy_factor(nvi_final)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Kc (base)", f"{kc_base:.2f}")
        with c2: st.metric("Kndvi", f"{Kndvi:.2f}")
        with c3: st.metric("TAW (mm)", f"{TAW:.1f}")
        with c4: st.metric("RAW (mm)", f"{RAW:.1f}")
        st.markdown("<div class='small-note'>ETc = ETo × Kc × Kndvi × PWRF × Ks_soil.</div>", unsafe_allow_html=True)

        applied_liters_yday = st.number_input("Applied yesterday (L) for balance correction", value=0.0, step=0.1)
        applied_mm_yday = (applied_liters_yday / AREA) if AREA>0 else 0.0

        ETc_yday = eto_yday * kc_base * Kndvi * PWRF
        ETc_next = eto_next * kc_base * Kndvi * PWRF * Ks_soil

        eff_rain_y = 0.8*rain_yday
        eff_rain_next = 0.8*forecast_rain

        depletion_prev = st.number_input("Depletion_prev (mm)", value=RAW/2.0, step=0.5)
        depletion_start = max(0.0, depletion_prev + ETc_yday - eff_rain_y - applied_mm_yday)
        need_mm = min(max(0.0, depletion_start + ETc_next - eff_rain_next), TAW)
        suggested_liters = (need_mm * AREA) / max(EFF, 1e-6)

        rain_skip = forecast_rain >= 10
        if treatment == "2":
            trigger = (vwc_predawn is not None and vwc_predawn <= trigger_vwc) or (depletion_start >= RAW)
        elif treatment == "3":
            trigger = (depletion_start >= RAW) or (nvi_final is not None and nvi_final < 0.55)
        else:
            trigger = ((vwc_predawn is not None and vwc_predawn <= trigger_vwc) or (depletion_start >= RAW) or (nvi_final is not None and nvi_final < 0.55))

        if rain_skip: decision="Skip (rain forecast)"
        elif suggested_liters < 0.5: decision="Skip (no irrigation needed)"
        elif trigger: decision="Irrigate"
        else: decision="Skip"

        if "Skip" not in decision:
            st.success(f"Decision: {decision} - Apply {suggested_liters:.2f} L per plot")
        else:
            st.warning(f"Decision: {decision} - Suggested water = {suggested_liters:.2f} L (not applied)")

        meter_start = st.number_input("Meter Start (L)", value=0.0, step=0.1)
        meter_end = st.number_input("Meter End (L)", value=0.0, step=0.1)
        applied = max(0.0, meter_end - meter_start)

        if connected and st.button(f"Save decision for {TREATMENT_TECH[treatment]}"):
            headers = REQUIRED_SHEETS["MAIN"]
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "treatment": f"T{treatment}",
                "crop": crop_choice, "stage": stage,
                "eto_yday": round(eto_yday,3), "eto_next": round(eto_next,3),
                "kc": round(kc_base,3), "Kndvi": round(Kndvi,3), "PWRF": round(PWRF,3), "Ks_soil": round(Ks_soil,3),
                "ETc_yday": round(ETc_yday,3), "ETc_next": round(ETc_next,3),
                "rain_yday": rain_yday, "forecast_rain": forecast_rain,
                "eff_rain_y": round(eff_rain_y,3), "eff_rain_next": round(eff_rain_next,3),
                "vwc_predawn": vwc_predawn if vwc_predawn is not None else "",
                "ndvi_rgn": ndvi_rgn if ndvi_rgn is not None else "",
                "ndvi_ocn": ndvi_ocn if ndvi_ocn is not None else "",
                "nvi": round(nvi_final,3) if nvi_final is not None else "",
                "TAW": round(TAW,2), "RAW": round(RAW,2),
                "depletion_prev": round(depletion_prev,2), "depletion_start": round(depletion_start,2),
                "need_mm": round(need_mm,2),
                "suggested_liters": round(suggested_liters,2),
                "meter_start": meter_start,"meter_end": meter_end,"applied_liters": round(applied,2),
                "action": decision
            }
            append_row(ws_main, row, headers); st.success("Saved to Google Sheet")

with tabs[1]:
    st.subheader("Sensor Data Upload & Analysis")
    up = st.file_uploader("Upload hourly soil sensor CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        ts_col = None
        for cand in ["timestamp","time","datetime","date_time"]:
            if cand in [c.lower() for c in df.columns]:
                for c in df.columns:
                    if c.lower()==cand: ts_col=c; break
                break
        if ts_col is None:
            st.error("No timestamp column found. Include 'timestamp'.")
        else:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=[ts_col])
            df = df.set_index(ts_col).sort_index()
            vwc_col = None
            for cand in ["vwc","soil_moisture","soil_vwc"]:
                for c in df.columns:
                    if c.lower()==cand: vwc_col=c; break
                if vwc_col: break
            if vwc_col is None:
                st.warning("No 'vwc' column found. Showing head only.")
                st.dataframe(df.head())
            else:
                predawn = df.between_time("04:00", "06:00")[vwc_col]
                if not predawn.empty:
                    vwc_predawn = float(predawn.mean())
                    st.metric("Predawn VWC (%)", f"{vwc_predawn:.1f}")
                    st.session_state["predawn_vwc"] = vwc_predawn
                daily = df[vwc_col].resample("D").agg(["mean","min","max"])
                st.write("Daily VWC summary (mean/min/max):")
                st.dataframe(daily.tail(14))
                fig, ax = plt.subplots()
                ax.plot(df.index, df[vwc_col])
                ax.set_title("Hourly Soil Moisture (VWC %)")
                ax.set_ylabel("VWC (%)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

with tabs[2]:
    st.subheader("Hourly Weather CSV -> ETo")
    upw = st.file_uploader("Upload hourly weather CSV", type=["csv"])
    if upw is not None:
        wx = pd.read_csv(upw)

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

        if c_ts is None or c_T is None or c_RH is None or c_u2 is None or c_Rs is None or c_P is None:
            st.error("Missing required columns. Need timestamp, Tair_C, RH, wind_ms, solar_MJ_m2, P_kPa.")
        else:
            wx[c_ts] = pd.to_datetime(wx[c_ts], errors="coerce")
            wx = wx.dropna(subset=[c_ts]).sort_values(c_ts).reset_index(drop=True)
            T = pd.to_numeric(wx[c_T], errors="coerce")
            RH = pd.to_numeric(wx[c_RH], errors="coerce")
            u2 = pd.to_numeric(wx[c_u2], errors="coerce")
            Rs = pd.to_numeric(wx[c_Rs], errors="coerce")
            P = pd.to_numeric(wx[c_P], errors="coerce")
            if P.mean()>20:
                P = P*0.1
            eto_hourly_vals = []
            for i in range(len(wx)):
                eto_h = eto_hourly(
                    float(T.iloc[i]) if not pd.isna(T.iloc[i]) else 20.0,
                    float(RH.iloc[i]) if not pd.isna(RH.iloc[i]) else 60.0,
                    float(Rs.iloc[i]) if not pd.isna(Rs.iloc[i]) else 0.0,
                    float(u2.iloc[i]) if not pd.isna(u2.iloc[i]) else 2.0,
                    float(P.iloc[i]) if not pd.isna(P.iloc[i]) else 101.3
                )
                eto_hourly_vals.append(eto_h)
            wx["ETo_hourly_mm"] = eto_hourly_vals
            wx = wx.rename(columns={c_ts:"timestamp", c_T:"Tair_C", c_RH:"RH", c_u2:"u2_ms", c_Rs:"solar_MJ_m2", c_P:"P_kPa"})
            st.success("Computed hourly ETo (mm/h).")
            st.dataframe(wx.head())

            wx["date"] = pd.to_datetime(wx["timestamp"]).dt.date.astype(str)
            daily = wx.groupby("date", as_index=False).agg({
                "P_kPa":"mean",
                "Tair_C":"mean",
                "RH":"mean",
                "u2_ms":"mean",
                "solar_MJ_m2":"sum",
                "ETo_hourly_mm":"sum"
            }).rename(columns={"ETo_hourly_mm":"ETo_mm"})
            if c_rain:
                import pandas as pd
                rain = pd.to_numeric(pd.Series(wx[c_rain]), errors="coerce").fillna(0.0)
                wx["rain_mm"] = rain
                rain_daily = wx.groupby("date", as_index=False)["rain_mm"].sum()
                daily = daily.merge(rain_daily, on="date", how="left")
            else:
                daily["rain_mm"] = 0.0

            fig1, ax1 = plt.subplots()
            ax1.plot(pd.to_datetime(wx["timestamp"]), wx["ETo_hourly_mm"])
            ax1.set_title("Hourly ETo (mm/h)")
            ax1.set_ylabel("ETo (mm)")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(pd.to_datetime(daily["date"]), daily["ETo_mm"], marker="o")
            ax2.set_title("Daily ETo (mm/day)")
            ax2.set_ylabel("ETo (mm)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

            if connected and st.button("Append daily to Google Sheet (Weather_ETo)"):
                existing = sheet_to_df(ws_weather)
                if not existing.empty and "date" in existing.columns:
                    existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.date.astype(str)
                for _,row in daily.iterrows():
                    if existing.empty or row["date"] not in set(existing["date"]):
                        data = {
                            "date": row["date"], "timestamp":"",
                            "P_kPa": row.get("P_kPa",""),
                            "rain_mm": row.get("rain_mm",""),
                            "Tmean_C": row.get("Tair_C",""),
                            "Tmax_C":"", "Tmin_C":"",
                            "RHmean": row.get("RH",""),
                            "u2_ms": row.get("u2_ms",""),
                            "n_hours":"",
                            "ETo_mm": row.get("ETo_mm",""),
                            "ETo_hourly_mm":""
                        }
                        append_row(ws_weather, data, REQUIRED_SHEETS["Weather_ETo"])
                st.success("Weather_ETo sheet updated.")

with tabs[3]:
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

            dff_vwc = dff[dff["treatment"].isin(["T2","T4"])] if "treatment" in dff.columns else pd.DataFrame()
            if not dff_vwc.empty and dff_vwc["vwc_predawn"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff_vwc["timestamp"], dff_vwc["vwc_predawn"], marker="o")
                ax.set_ylabel("VWC (%)"); ax.set_title("Predawn Soil Moisture (T2 & T4)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            dff_ndvi = dff[dff["treatment"].isin(["T3","T4"])] if "treatment" in dff.columns else pd.DataFrame()
            if not dff_ndvi.empty and dff_ndvi["nvi"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff_ndvi["timestamp"], dff_ndvi["nvi"], marker="o")
                ax.set_ylabel("Harmonized NDVI"); ax.set_title("Canopy NDVI (RGN+OCN fused)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            if "suggested_liters" in dff.columns and dff["suggested_liters"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff["timestamp"], dff["suggested_liters"].fillna(0), marker="o", label="Suggested liters")
                if "ETc_next" in dff.columns: ax.plot(dff["timestamp"], dff["ETc_next"].fillna(0), linestyle="--", label="ETc (mm) next")
                if "forecast_rain" in dff.columns: ax.bar(dff["timestamp"], (0.8*dff["forecast_rain"]).fillna(0), alpha=0.3, label="Eff. rain next (mm)")
                ax.set_title("Decision Logic Components")
                ax.grid(True, alpha=0.3); ax.legend()
                st.pyplot(fig)

            st.markdown("Latest 50 rows:")
            st.dataframe(dff.tail(50))

with tabs[4]:
    st.subheader("NDVI Harmonization (OCN->RGN)")
    try:
        st.caption("Connect Google Sheets to manage calibration.")
        dfcal = sheet_to_df(ws_cal) if connected else pd.DataFrame()
        if connected and dfcal.empty:
            ws_cal.append_row(["stage","a","b","sigma_rgn","sigma_ocn","updated_at"])
            for stg in ["initial","mid","late"]:
                ws_cal.append_row([stg,0.0,1.0,0.03,0.03,datetime.utcnow().isoformat()])
            dfcal = sheet_to_df(ws_cal)
        if connected:
            st.dataframe(dfcal)
            st.caption("Adjust linear map a + b*OCN to RGN space and sensor noise for weighted fusion.")
            for stg in ["initial","mid","late"]:
                row = dfcal[dfcal["stage"].str.lower()==stg]
                a0=float(row["a"].iloc[0]) if not row.empty else 0.0
                b0=float(row["b"].iloc[0]) if not row.empty else 1.0
                s_r=float(row["sigma_rgn"].iloc[0]) if not row.empty else 0.03
                s_o=float(row["sigma_ocn"].iloc[0]) if not row.empty else 0.03
                c1,c2,c3,c4 = st.columns(4)
                with c1: a_new = st.number_input(f"{stg} a", value=a0, step=0.01, key=f"a_{stg}")
                with c2: b_new = st.number_input(f"{stg} b", value=b0, step=0.01, key=f"b_{stg}")
                with c3: srg_new = st.number_input(f"{stg} sigma_rgn", value=s_r, step=0.005, key=f"srg_{stg}")
                with c4: soc_new = st.number_input(f"{stg} sigma_ocn", value=s_o, step=0.005, key=f"soc_{stg}")
                if st.button(f"Save {stg} coefficients"):
                    df_now = sheet_to_df(ws_cal)
                    idx = df_now.index[df_now["stage"].str.lower()==stg].tolist()
                    if idx:
                        rownum = idx[0] + 2
                        ws_cal.update(f"B{rownum}:E{rownum}", [[a_new,b_new,srg_new,soc_new]])
                        ws_cal.update(f"F{rownum}", [[datetime.utcnow().isoformat()]])
                    else:
                        ws_cal.append_row([stg, a_new, b_new, srg_new, soc_new, datetime.utcnow().isoformat()])
                    st.success(f"Saved {stg}")
    except Exception as e:
        st.warning(f"NDVI tab error: {e}")
