import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gspread, json, math
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta, date

# ---------------- UI THEME ----------------
st.set_page_config(page_title="Patrick Smart Irrigation — v3.0", layout="wide")

PRIMARY = "#0EA5E9"   # Tailwind sky-500
ACCENT  = "#22C55E"   # Tailwind green-500
WARN    = "#F59E0B"   # amber-500
DANGER  = "#EF4444"   # red-500
MUTED   = "#6B7280"   # gray-500

st.markdown(f"""
    <style>
    .block-container {{ padding-top: 1.5rem; }}
    h1, h2, h3 {{ color: #111827; }}
    .stButton>button {{ border-radius: 14px; padding: 0.5rem 1rem; font-weight: 600; }}
    .metric-small span {{ font-size: 0.9rem !important; color:{MUTED}; }}
    .note {{ font-size: 0.9rem; color:{MUTED}; }}
    </style>
""", unsafe_allow_html=True)

# --------------- CONSTANTS ----------------
SPREADSHEET_NAME = "Patrick_Irrigation_Log"
WEATHER_SHEET = "Weather_ETo"
CALIB_SHEET = "NDVI_Calibration"
META_SHEET = "App_Metadata"

PLOT_TECH = {
    "1": "Baseline (manual only)",
    "2": "Sensor + Weather",
    "3": "NDVI + Weather",
    "4": "Sensor + NDVI + Weather",
}

Kc_STAGE = {"initial": 0.55, "mid": 1.00, "late": 0.85}
ROOT_DEPTH = {"initial": 0.15, "mid": 0.25, "late": 0.30}  # m
DEFAULTS = {"FC": 30.0, "PWP": 10.0, "MAD": 0.20, "EFFICIENCY": 0.85, "PLOT_AREA_M2": 1.0}
DEFAULT_LAT = -1.95
DEFAULT_ALT = 1500.0

# --------------- SHEET CONNECTION ----------------
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
            ss = client.open(SPREADSHEET_NAME)
        st.sidebar.success("✅ Connected to your personal Google Drive Sheet")
    except gspread.SpreadsheetNotFound:
        st.error("❌ Sheet not found. Share your sheet with the service account and/or set SHEET_ID in secrets.")
        raise
    return client, ss

REQUIRED_SHEETS = {
    "MAIN": [
        "timestamp","plot_id","strategy","eto","kc","etc","forecast_rain",
        "vwc","camera","nvi","est_biom","suggested_liters",
        "meter_start","meter_end","applied_liters","action"
    ],
    WEATHER_SHEET: [
        "date","P_kPa","rain_mm","Tmean_C","Tmax_C","Tmin_C","RHmean","u2_ms","n_hours","ETo_mm"
    ],
    CALIB_SHEET: [
        "stage","a","b","sigma_rgn","sigma_ocn","updated_at"
    ],
    META_SHEET: ["key","value"],
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
            ws = ss.add_worksheet(title=title, rows=1000, cols=max(26, len(headers)))
            ws.append_row(headers)
            return ws
        cur = ws.row_values(1)
        if not cur:
            ws.append_row(headers)
        else:
            miss = [h for h in headers if h not in cur]
            if miss:
                ws.update('1:1', [cur + miss])
        return ws

    ws_weather = ensure_tab(WEATHER_SHEET, REQUIRED_SHEETS[WEATHER_SHEET])
    ws_cal = ensure_tab(CALIB_SHEET, REQUIRED_SHEETS[CALIB_SHEET])
    ws_meta = ensure_tab(META_SHEET, REQUIRED_SHEETS[META_SHEET])

    if not ws_cal.get_all_records():
        for stg in ["initial","mid","late"]:
            ws_cal.append_row([stg, 0.0, 1.0, 0.03, 0.03, datetime.utcnow().isoformat()])
    if not ws_meta.get_all_records():
        ws_meta.append_row(["app_version","3.0"])
        ws_meta.append_row(["last_update", str(date.today())])
        ws_meta.append_row(["researcher","Patrick Habyarimana"])

    return ws_main, ws_weather, ws_cal, ws_meta

def sheet_to_df(ws):
    data = ws.get_all_records()
    return pd.DataFrame(data) if data else pd.DataFrame()

def append_row(ws, dct, header_order):
    ws.append_row([dct.get(h,"") for h in header_order])

# --------------- FAO-56 CORE ----------------
def saturation_vapor_pressure(T):
    return 0.6108 * math.exp((17.27 * T) / (T + 237.3))

def slope_vapor_pressure_curve(T):
    es = saturation_vapor_pressure(T)
    return 4098 * es / ((T + 237.3) ** 2)

def psychrometric_constant(P_kPa):
    return 0.000665 * P_kPa

def extraterrestrial_radiation(lat_rad, J):
    dr = 1 + 0.033 * math.cos(2 * math.pi * J / 365)
    delta = 0.409 * math.sin(2 * math.pi * J / 365 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))
    return (24 * 60 / math.pi) * 0.0820 * dr * (ws * math.sin(lat_rad) * math.sin(delta) + math.cos(lat_rad) * math.cos(delta) * math.sin(ws))

def daylight_hours(lat_rad, J):
    delta = 0.409 * math.sin(2 * math.pi * J / 365 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))
    return 24 / math.pi * ws

def solar_radiation_from_sunshine(n_hours, N, as_coef=0.25, bs_coef=0.5, Ra=None):
    if Ra is None: return None
    n = max(0.0, min(n_hours, N))
    return (as_coef + bs_coef * (n / N)) * Ra

def net_radiation(Rs, Tmean_C, RHmean):
    albedo = 0.23
    Rns = (1 - albedo) * Rs
    sigma = 4.903e-9
    Tk = Tmean_C + 273.16
    ea = RHmean * saturation_vapor_pressure(Tmean_C)
    Rnl = sigma * (Tk**4) * (0.34 - 0.14 * max(ea,1e-6)**0.5) * 1.35 - 0.35
    return max(0.0, Rns - max(0.0, Rnl))

def eto_fao56_daily(Tmean, Tmax, Tmin, RHmean, u2, P_kPa, Ra, n_hours, lat_rad, J):
    N = daylight_hours(lat_rad, J)
    Rs = solar_radiation_from_sunshine(n_hours, N, Ra=Ra)
    if Rs is None: return None
    Rn = net_radiation(Rs, Tmean, RHmean)
    G = 0.0
    es_Tmax = saturation_vapor_pressure(Tmax)
    es_Tmin = saturation_vapor_pressure(Tmin)
    es = (es_Tmax + es_Tmin) / 2.0
    ea = RHmean * es
    delta = slope_vapor_pressure_curve(Tmean)
    gamma = psychrometric_constant(P_kPa)
    num = 0.408*delta*(Rn-G) + gamma*(900.0/(Tmean+273.0))*u2*(es-ea)
    den = delta + gamma*(1+0.34*u2)
    return max(0.0, num/max(den,1e-6))

# --------------- NDVI FUSION ----------------
def fuse_ndvi(ndvi_rgn, ndvi_ocn, a=0.0, b=1.0, sigma_rgn=0.03, sigma_ocn=0.03, q_rgn=1.0, q_ocn=1.0, tau=0.10, ndvi7d_median=None):
    ndvi_ocn_rg = a + b*ndvi_ocn if ndvi_ocn is not None else None
    if ndvi_rgn is not None and ndvi_ocn is not None:
        if abs(ndvi_rgn - ndvi_ocn_rg) > tau and ndvi7d_median is not None:
            cand = ndvi_rgn if abs(ndvi_rgn - ndvi7d_median) <= abs(ndvi_ocn_rg - ndvi7d_median) else ndvi_ocn_rg
            ndvi_fuse = float(np.clip(cand,0,1))
        else:
            w_r = (q_rgn/(sigma_rgn**2)); w_o = (q_ocn/(sigma_ocn**2))
            s = max(w_r+w_o, 1e-6); w_r/=s; w_o/=s
            ndvi_fuse = float(np.clip(w_r*ndvi_rgn + w_o*ndvi_ocn_rg, 0, 1))
    elif ndvi_rgn is not None:
        ndvi_fuse = float(np.clip(ndvi_rgn,0,1))
    elif ndvi_ocn is not None:
        ndvi_fuse = float(np.clip(ndvi_ocn_rg,0,1))
    else:
        return None
    return ndvi_fuse

# --------------- SIDEBAR ----------------
st.sidebar.header("Settings")
lat = st.sidebar.number_input("Latitude (deg)", value=DEFAULT_LAT, step=0.01)
alt = st.sidebar.number_input("Altitude (m)", value=float(DEFAULT_ALT), step=10.0)
FC = st.sidebar.number_input("Field Capacity (VWC %)", value=DEFAULTS["FC"], step=0.5)
PWP = st.sidebar.number_input("Permanent Wilting Point (VWC %)", value=DEFAULTS["PWP"], step=0.5)
MAD = st.sidebar.number_input("Management Allowable Depletion (0-1)", value=DEFAULTS["MAD"], step=0.05, min_value=0.05, max_value=0.6)
EFF = st.sidebar.number_input("Irrigation Efficiency (0-1)", value=DEFAULTS["EFFICIENCY"], step=0.05, min_value=0.5, max_value=1.0)
AREA = st.sidebar.number_input("Plot Area (m²)", value=DEFAULTS["PLOT_AREA_M2"], step=0.1, min_value=0.1)
plot_id = st.sidebar.selectbox("Plot", options=list(PLOT_TECH.keys()), format_func=lambda k: f"Plot {k} — {PLOT_TECH[k]}")

st.sidebar.markdown(f"<div class='note'>Plot 1 is manual-only. Plots 2–4 use decision logic.</div>", unsafe_allow_html=True)

# --------------- CONNECT ----------------
try:
    client, ss = connect_gsheet()
    ws_main, ws_weather, ws_cal, ws_meta = ensure_sheet_structure(ss)
    connected = True
except Exception as e:
    st.error(f"Failed to connect to Google Sheet: {e}")
    connected = False

def load_calibration(ws_cal):
    df = sheet_to_df(ws_cal)
    out = {"initial": (0.0,1.0,0.03,0.03), "mid": (0.0,1.0,0.03,0.03), "late": (0.0,1.0,0.03,0.03)}
    if not df.empty and "stage" in df.columns:
        for _,r in df.iterrows():
            stg = str(r["stage"]).strip().lower()
            out[stg] = (float(r.get("a",0.0)), float(r.get("b",1.0)), float(r.get("sigma_rgn",0.03)), float(r.get("sigma_ocn",0.03)))
    return out

calib = load_calibration(ws_cal) if connected else {"initial": (0.0,1.0,0.03,0.03), "mid": (0.0,1.0,0.03,0.03), "late": (0.0,1.0,0.03,0.03)}

# --------------- TABS ----------------
tabs = st.tabs(["Weather → ETo", "Daily Decision", "Analytics", "NDVI Harmonization"])

# ----- TAB 1: Weather → ETo -----
with tabs[0]:
    st.header("Weather → ETo (FAO-56)")
    st.caption("Upload daily CSV with columns like: 日, 気圧(hPa), 降水量(mm), 気温(℃), 湿度(％), 風速(m/s), 日照時間(h)")

    up = st.file_uploader("Upload weather CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up, encoding="utf-8")
        except Exception:
            df = pd.read_csv(up, encoding="shift_jis")

        def find_col(keys):
            for c in df.columns:
                for k in keys:
                    if k in str(c):
                        return c
            return None

        col_date = find_col(["日"])
        col_p = find_col(["気圧"])
        col_rain = find_col(["降水量"])
        col_t = find_col(["気温"])
        col_rh = find_col(["湿度"])
        col_u2 = find_col(["風速"])
        col_sun = find_col(["日照"])

        if col_date is None:
            st.error("Date column (日) not found.")
        else:
            df["date"] = pd.to_datetime(df[col_date], errors="coerce")
            df = df.dropna(subset=["date"])

            P_kPa = (pd.to_numeric(df[col_p], errors="coerce")*0.1) if col_p else 101.3*((293-0.0065*alt)/293)**5.26
            rain_mm = pd.to_numeric(df[col_rain], errors="coerce") if col_rain else 0.0
            Tmean = pd.to_numeric(df[col_t], errors="coerce") if col_t else np.nan
            Tmax = Tmean + 2.0
            Tmin = Tmean - 2.0
            RHmean = (pd.to_numeric(df[col_rh], errors="coerce")/100.0) if col_rh else 0.6
            u2 = pd.to_numeric(df[col_u2], errors="coerce") if col_u2 else 2.0
            n_hours = pd.to_numeric(df[col_sun], errors="coerce") if col_sun else 7.0

            lat_rad = math.radians(lat)
            recs = []
            for i, r in df.iterrows():
                J = int(pd.to_datetime(r["date"]).dayofyear)
                Ra = extraterrestrial_radiation(lat_rad, J)
                eto = eto_fao56_daily(
                    Tmean=float(Tmean.iloc[i]) if not pd.isna(Tmean.iloc[i]) else float( ( (Tmax.iloc[i] if hasattr(Tmax,'iloc') else Tmean.iloc[i]) + (Tmin.iloc[i] if hasattr(Tmin,'iloc') else Tmean.iloc[i]) )/2.0 ),
                    Tmax=float(Tmax.iloc[i]) if hasattr(Tmax,"iloc") else float(Tmean.iloc[i])+2.0,
                    Tmin=float(Tmin.iloc[i]) if hasattr(Tmin,"iloc") else float(Tmean.iloc[i])-2.0,
                    RHmean=float(RHmean.iloc[i]) if not pd.isna(RHmean.iloc[i]) else 0.6,
                    u2=float(u2.iloc[i]) if not pd.isna(u2.iloc[i]) else 2.0,
                    P_kPa=float(P_kPa.iloc[i]) if hasattr(P_kPa,"iloc") else float(P_kPa),
                    Ra=Ra,
                    n_hours=float(n_hours.iloc[i]) if not pd.isna(n_hours.iloc[i]) else 7.0,
                    lat_rad=lat_rad,
                    J=J,
                )
                recs.append({
                    "date": pd.to_datetime(r["date"]).date().isoformat(),
                    "P_kPa": float(P_kPa.iloc[i]) if hasattr(P_kPa,"iloc") else float(P_kPa),
                    "rain_mm": float(rain_mm.iloc[i]) if hasattr(rain_mm,"iloc") and not pd.isna(rain_mm.iloc[i]) else 0.0,
                    "Tmean_C": float(Tmean.iloc[i]) if not pd.isna(Tmean.iloc[i]) else np.nan,
                    "Tmax_C": float((Tmax.iloc[i] if hasattr(Tmax,'iloc') else np.nan)),
                    "Tmin_C": float((Tmin.iloc[i] if hasattr(Tmin,'iloc') else np.nan)),
                    "RHmean": float(RHmean.iloc[i]) if not pd.isna(RHmean.iloc[i]) else 0.6,
                    "u2_ms": float(u2.iloc[i]) if not pd.isna(u2.iloc[i]) else 2.0,
                    "n_hours": float(n_hours.iloc[i]) if not pd.isna(n_hours.iloc[i]) else 7.0,
                    "ETo_mm": round(float(eto),3) if eto is not None else None
                })
            etodf = pd.DataFrame(recs)
            st.success("Parsed CSV and computed ETo.")
            st.dataframe(etodf)

            if not etodf.empty:
                fig, ax = plt.subplots()
                ax.plot(pd.to_datetime(etodf["date"]), etodf["ETo_mm"], marker="o")
                ax.set_ylabel("ETo (mm/day)")
                ax.set_title("Daily ETo")
                ax.grid(True, alpha=0.2)
                # compact ticks
                ax.tick_params(axis='x', labelrotation=0, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                st.pyplot(fig)

            if connected and st.button("Append to Google Sheet (Weather_ETo)"):
                existing = sheet_to_df(ws_weather)
                if not existing.empty:
                    existing["date"] = pd.to_datetime(existing["date"]).dt.date.astype(str)
                for _,row in etodf.iterrows():
                    if existing.empty or row["date"] not in set(existing["date"]):
                        append_row(ws_weather, row, ["date","P_kPa","rain_mm","Tmean_C","Tmax_C","Tmin_C","RHmean","u2_ms","n_hours","ETo_mm"])
                st.success("Weather_ETo sheet updated.")

# ----- TAB 2: Daily Decision -----
with tabs[1]:
    st.header(f"Daily decision — {PLOT_TECH[plot_id]}")
    st.caption("One decision per day. Plot 1 is manual-only. For Plots 2–4, the engine uses VWC, NDVI fusion, and ETo.")

    # Auto-prefill yesterday's ETo/rain
    eto_yday_default, rain_yday_default, eto_next_default = 0.0, 0.0, 0.0
    if connected:
        try:
            wdf = sheet_to_df(ws_weather)
            if not wdf.empty and "date" in wdf.columns:
                wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce")
                yday = (datetime.utcnow().date() - timedelta(days=1))
                match = wdf[wdf["date"].dt.date == yday]
                if not match.empty:
                    eto_yday_default = float(pd.to_numeric(match["ETo_mm"], errors="coerce").iloc[0] or 0.0)
                    rain_yday_default = float(pd.to_numeric(match["rain_mm"], errors="coerce").iloc[0] or 0.0) if "rain_mm" in match.columns else 0.0
                    eto_next_default = eto_yday_default
        except Exception:
            pass

    c1, c2, c3 = st.columns(3)
    with c1:
        stage = st.selectbox("Crop Stage", ["initial","mid","late"])
        kc_stage = Kc_STAGE[stage]; zr = ROOT_DEPTH[stage]
    with c2:
        eto_yday = st.number_input("ETo (yesterday, mm)", value=float(eto_yday_default), step=0.1)
        eto_next = st.number_input("ETo forecast (next 24h, mm)", value=float(eto_next_default), step=0.1)
    with c3:
        rain_yday = st.number_input("Rain (yesterday, mm)", value=float(rain_yday_default), step=0.5)
        forecast_rain = st.number_input("Rain forecast next 24h (mm)", value=0.0, step=0.5)

    if plot_id == "1":
        st.subheader("Baseline (manual entry only)")
        meter_start = st.number_input("Meter Start (L)", value=0.0, step=0.1)
        meter_end = st.number_input("Meter End (L)", value=0.0, step=0.1)
        applied = max(0.0, meter_end - meter_start)
        st.success(f"Applied today: **{applied:.2f} L**")

        if connected and st.button("Save baseline entry"):
            headers = ["timestamp","plot_id","strategy","eto","kc","etc","forecast_rain","vwc","camera","nvi","est_biom","suggested_liters","meter_start","meter_end","applied_liters","action"]
            if sheet_to_df(ws_main).empty:
                ws_main.append_row(headers)
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "plot_id": f"Plot {plot_id}",
                "strategy": PLOT_TECH[plot_id],
                "eto": eto_next, "kc":"", "etc":"",
                "forecast_rain": forecast_rain,
                "vwc":"", "camera":"", "nvi":"", "est_biom":"",
                "suggested_liters":"",
                "meter_start": meter_start, "meter_end": meter_end,
                "applied_liters": round(applied,2),
                "action": "Manual irrigation"
            }
            append_row(ws_main, row, headers)
            st.success("Saved baseline entry ✅")
    else:
        c4, c5, c6 = st.columns(3)
        with c4:
            vwc_predawn = st.number_input("Soil Moisture VWC (predawn %, 04–06h)", value=28.0, step=0.1, disabled=(plot_id=="3"))
        with c5:
            ndvi_rgn = st.number_input("NDVI_RGN (if measured)", value=0.0, step=0.01, min_value=0.0, max_value=1.0, disabled=(plot_id=="2"))
        with c6:
            ndvi_ocn = st.number_input("NDVI_OCN (if measured)", value=0.0, step=0.01, min_value=0.0, max_value=1.0, disabled=(plot_id=="2"))

        # Load calibration
        a,b,sr,so = calib.get(stage, (0.0,1.0,0.03,0.03))
        ndvi7 = None
        if connected:
            try:
                dfm = sheet_to_df(ws_main)
                if not dfm.empty and "nvi" in [c.lower() for c in dfm.columns]:
                    nvi_col = [c for c in dfm.columns if c.lower()=="nvi"][0]
                    ndvi7 = float(pd.to_numeric(dfm.tail(7)[nvi_col], errors="coerce").median())
            except Exception:
                pass
        ndvi_final = None
        if plot_id in ("3","4"):
            ndvi_final = fuse_ndvi(ndvi_rgn if ndvi_rgn>0 else None, ndvi_ocn if ndvi_ocn>0 else None, a=a, b=b, sigma_rgn=sr, sigma_ocn=so, ndvi7d_median=ndvi7)
            st.markdown(f"Unified NDVI (nVI): **{ndvi_final:.3f}**" if ndvi_final is not None else "No NDVI today.")

        kc = kc_stage
        if ndvi_final is not None:
            kc_ndvi = float(np.clip(0.30 + 0.70*ndvi_final, 0.30, 1.10))
            kc = 0.7*kc_stage + 0.3*kc_ndvi

        etc_yday = eto_yday * kc
        etc_next = eto_next * kc

        TAW = (FC - PWP) * zr * 10.0
        RAW = MAD * TAW
        trigger_vwc = FC - MAD*(FC - PWP)

        # Compact metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Used Kc", f"{kc:.2f}", help="Fusion: 70% stage Kc + 30% NDVI-based Kc if NDVI provided")
        m2.metric("TAW (mm)", f"{TAW:.1f}", help="Total Available Water")
        m3.metric("RAW (mm)", f"{RAW:.1f}", help="Readily Available Water (MAD×TAW)")
        m4.metric("Trigger VWC (%)", f"{trigger_vwc:.1f}", help="FC − MAD×(FC−PWP)")

        st.markdown(f"<div class='note'>ETc = ETo × Kc.  Suggestion = (Depletion_start + ETc_next − EffectiveRain_next) × Area.</div>", unsafe_allow_html=True)

        depletion_prev = st.number_input("Depletion_prev at start of day (mm)", value=RAW/2.0, step=0.5)
        eff_rain_y = 0.8*rain_yday
        eff_rain_next = 0.8*forecast_rain
        depletion_start = max(0.0, depletion_prev + etc_yday - eff_rain_y)
        need_mm = max(0.0, depletion_start + etc_next - eff_rain_next)
        need_mm = min(need_mm, TAW)
        sugg_liters = need_mm * AREA

        # Decision logic
        if plot_id == "2":  # Sensor + Weather
            decision = "Irrigate" if (vwc_predawn <= trigger_vwc) or (depletion_start >= RAW) else "Skip"
        elif plot_id == "3":  # NDVI + Weather
            guard = False
            if ndvi_final is not None and ndvi7 is not None:
                guard = (ndvi_final < 0.95*ndvi7) and (abs(vwc_predawn - trigger_vwc) <= 3.0)
            decision = "Irrigate" if guard or (depletion_start >= RAW) else "Skip"
        else:  # Plot 4
            guard = False
            if ndvi_final is not None and ndvi7 is not None:
                guard = (ndvi_final < 0.95*ndvi7) and (abs(vwc_predawn - trigger_vwc) <= 3.0)
            decision = "Irrigate" if guard or (vwc_predawn <= trigger_vwc) or (depletion_start >= RAW) else "Skip"
        if forecast_rain >= 10:
            decision = "Skip (rain forecast)"

        st.success(f"Decision: **{decision}** — Recommended **{sugg_liters:.2f} L** per plot")

        meter_start = st.number_input("Meter Start (L)", value=0.0, step=0.1)
        meter_end = st.number_input("Meter End (L)", value=0.0, step=0.1)
        applied = max(0.0, meter_end - meter_start)

        if connected and st.button("Save daily decision"):
            headers = ["timestamp","plot_id","strategy","eto","kc","etc","forecast_rain","vwc","camera","nvi","est_biom","suggested_liters","meter_start","meter_end","applied_liters","action"]
            if sheet_to_df(ws_main).empty:
                ws_main.append_row(headers)
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "plot_id": f"Plot {plot_id}",
                "strategy": PLOT_TECH[plot_id],
                "eto": eto_next,
                "kc": kc,
                "etc": etc_next,
                "forecast_rain": forecast_rain,
                "vwc": vwc_predawn if plot_id in ("2","4") else "",
                "camera": ("RGN+OCN" if (ndvi_rgn>0 and ndvi_ocn>0) else ("RGN" if ndvi_rgn>0 else ("OCN" if ndvi_ocn>0 else ""))) if plot_id in ("3","4") else "",
                "nvi": round(ndvi_final,3) if (plot_id in ("3","4") and ndvi_final is not None) else "",
                "est_biom": "",
                "suggested_liters": round(sugg_liters,2),
                "meter_start": meter_start,
                "meter_end": meter_end,
                "applied_liters": round(applied,2),
                "action": decision
            }
            append_row(ws_main, row, headers)
            st.success("Saved to Google Sheet ✅")

# ----- TAB 3: Analytics -----
with tabs[2]:
    st.header("Analytics")
    if connected:
        df = sheet_to_df(ws_main)
        if df.empty:
            st.info("No logs yet.")
        else:
            # Normalize
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            for c in ["vwc","nvi","applied_liters","eto","kc","etc","suggested_liters"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "plot_id" in df.columns:
                df["plot_id"] = df["plot_id"].astype(str)
            df = df.sort_values("timestamp")

            fplot = st.selectbox("Filter by plot", options=["All","1","2","3","4"], index=0, format_func=lambda x: "All Plots" if x=="All" else f"Plot {x}")
            dff = df if fplot=="All" else df[df["plot_id"].str.contains(fplot)]

            # 1) VWC (only Plots 2 & 4)
            dff_vwc = dff[(dff["plot_id"].isin(["Plot 2","Plot 4"])) if "plot_id" in dff.columns else []]
            if not dff_vwc.empty and dff_vwc["vwc"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff_vwc["timestamp"], dff_vwc["vwc"], marker="o")
                ax.axhline(DEFAULTS["FC"] - DEFAULTS["MAD"]*(DEFAULTS["FC"]-DEFAULTS["PWP"]), linestyle="--", color="#999", label="Trigger VWC")
                ax.set_ylabel("VWC (%)"); ax.set_title("Predawn Soil Moisture (Plots 2 & 4)")
                ax.legend()
                ax.grid(True, alpha=0.2)
                ax.tick_params(axis='x', labelsize=8); ax.tick_params(axis='y', labelsize=8)
                st.pyplot(fig)

            # 2) NDVI (Plots 3 & 4)
            dff_ndvi = dff[(dff["plot_id"].isin(["Plot 3","Plot 4"])) if "plot_id" in dff.columns else []]
            if not dff_ndvi.empty and dff_ndvi["nvi"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff_ndvi["timestamp"], dff_ndvi["nvi"], marker="o")
                ax.set_ylabel("Unified NDVI"); ax.set_title("Canopy NDVI (RGN+OCN fused)")
                ax.grid(True, alpha=0.2)
                ax.tick_params(axis='x', labelsize=8); ax.tick_params(axis='y', labelsize=8)
                st.pyplot(fig)

            # 3) Applied water — Plot 1 only (as requested)
            dff_p1 = dff[dff["plot_id"]=="Plot 1"] if "plot_id" in dff.columns else pd.DataFrame()
            if not dff_p1.empty and dff_p1["applied_liters"].notna().any():
                fig, ax = plt.subplots()
                ax.bar(dff_p1["timestamp"], dff_p1["applied_liters"].fillna(0))
                ax.set_ylabel("Applied (L)"); ax.set_title("Manual Irrigation — Plot 1 Only")
                ax.grid(True, axis='y', alpha=0.2)
                ax.tick_params(axis='x', labelsize=8); ax.tick_params(axis='y', labelsize=8)
                st.pyplot(fig)

            # 4) Decision logic overlay chart (formula view)
            # Shows ETc_next, Depletion_start, EffectiveRain_next and Suggested liters timeline
            if dff["suggested_liters"].notna().any():
                fig, ax = plt.subplots()
                ax.plot(dff["timestamp"], dff["suggested_liters"].fillna(0), marker="o", label="Suggested liters")
                if "etc" in dff.columns:
                    ax.plot(dff["timestamp"], dff["etc"].fillna(0), linestyle="--", label="ETc (mm/day)")
                if "forecast_rain" in dff.columns:
                    ax.plot(dff["timestamp"], (0.8*dff["forecast_rain"]).fillna(0), linestyle=":", label="Eff. rain next (mm)")
                ax.set_title("Decision Logic Components")
                ax.grid(True, alpha=0.2); ax.legend()
                ax.tick_params(axis='x', labelsize=8); ax.tick_params(axis='y', labelsize=8)
                st.pyplot(fig)

            st.markdown("Latest 50 rows:")
            st.dataframe(dff.tail(50))
    else:
        st.info("Connect Google Sheets to view analytics.")

# ----- TAB 4: NDVI Harmonization -----
with tabs[3]:
    st.header("NDVI Harmonization (OCN→RGN)")
    if connected:
        dfcal = sheet_to_df(ws_cal)
        if dfcal.empty:
            dfcal = pd.DataFrame({"stage":["initial","mid","late"],"a":[0,0,0],"b":[1,1,1],"sigma_rgn":[0.03,0.03,0.03],"sigma_ocn":[0.03,0.03,0.03]})
        st.dataframe(dfcal)

        st.caption("Tune the linear map (a + b×OCN → RGN space) and per-sensor noise for weighted fusion.")

        for stg in ["initial","mid","late"]:
            row = dfcal[dfcal["stage"].str.lower()==stg]
            a0 = float(row["a"].iloc[0]) if not row.empty else 0.0
            b0 = float(row["b"].iloc[0]) if not row.empty else 1.0
            s_r = float(row["sigma_rgn"].iloc[0]) if not row.empty else 0.03
            s_o = float(row["sigma_ocn"].iloc[0]) if not row.empty else 0.03
            c1,c2,c3,c4 = st.columns(4)
            with c1: a_new = st.number_input(f"{stg} a", value=a0, step=0.01, key=f"a_{stg}")
            with c2: b_new = st.number_input(f"{stg} b", value=b0, step=0.01, key=f"b_{stg}")
            with c3: srg_new = st.number_input(f"{stg} sigma_rgn", value=s_r, step=0.005, key=f"srg_{stg}")
            with c4: soc_new = st.number_input(f"{stg} sigma_ocn", value=s_o, step=0.005, key=f"soc_{stg}")
            if st.button(f"Save {stg} coefficients"):
                df_now = sheet_to_df(ws_cal)
                if df_now.empty:
                    ws_cal.append_row(["stage","a","b","sigma_rgn","sigma_ocn","updated_at"])
                    ws_cal.append_row([stg, a_new, b_new, srg_new, soc_new, datetime.utcnow().isoformat()])
                else:
                    idx = df_now.index[df_now["stage"].str.lower()==stg].tolist()
                    if idx:
                        rownum = idx[0] + 2
                        ws_cal.update(f"B{rownum}:E{rownum}", [[a_new,b_new,srg_new,soc_new]])
                        ws_cal.update(f"F{rownum}", [[datetime.utcnow().isoformat()]])
                    else:
                        ws_cal.append_row([stg, a_new, b_new, srg_new, soc_new, datetime.utcnow().isoformat()])
                st.success(f"Saved {stg} ✅")
    else:
        st.info("Connect Google Sheets to manage calibration.")
