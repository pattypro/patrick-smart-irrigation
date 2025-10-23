# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gspread, json, math, hashlib
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta, date

st.set_page_config(page_title='Patrick Smart Irrigation - v3.9', layout='wide')
PRIMARY='#0EA5E9'; MUTED='#6B7280'
st.markdown(f"""
<style>
.block-container {{ padding-top:.6rem; }}
h1, h2, h3 {{ color:#0f172a; margin-top:.4rem; }}
.info-box {{ border-left:6px solid {PRIMARY}; padding:.6rem .8rem; background:#f0f9ff; border-radius:8px; margin:.4rem 0 1rem; }}
.small {{ font-size:.92rem; color:{MUTED}; }}
</style>
""", unsafe_allow_html=True)
st.title('Patrick Smart Irrigation — v3.9')

SPREADSHEET_NAME='Patrick_Irrigation_Log'
WEATHER_SHEET='Weather_ETo'
CALIB_SHEET='NDVI_Calibration'
META_SHEET='App_Metadata'
SENSOR_RAW='Sensor_Raw'
WEATHER_RAW='Weather_Raw'

CROP_PARAMS={
 'Spinach': {'Kc_ini':0.70,'Kc_mid':1.05,'Kc_end':0.95,'root_depth':0.20,'base_ETc_adj':1.10},
 'Lettuce': {'Kc_ini':0.65,'Kc_mid':1.00,'Kc_end':0.90,'root_depth':0.25,'base_ETc_adj':1.15},
 'Tomato':  {'Kc_ini':0.60,'Kc_mid':1.15,'Kc_end':0.85,'root_depth':0.45,'base_ETc_adj':1.05},
 'Maize':   {'Kc_ini':0.45,'Kc_mid':1.20,'Kc_end':0.80,'root_depth':0.60,'base_ETc_adj':1.00},
 'Beans':   {'Kc_ini':0.50,'Kc_mid':1.05,'Kc_end':0.85,'root_depth':0.35,'base_ETc_adj':1.00},
}
AREA=1.0
DEFAULTS={'FC':30.0,'PWP':10.0,'MAD':0.20,'EFFICIENCY':0.85}
RAIN_SKIP_MM=10.0

def connect_gsheet():
    scope=['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive','https://www.googleapis.com/auth/drive.file','https://www.googleapis.com/auth/drive.readonly']
    raw=st.secrets['GCP_SERVICE_ACCOUNT_JSON']
    creds_dict=json.loads(raw) if isinstance(raw,str) else dict(raw)
    creds=ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client=gspread.authorize(creds)
    try:
        ss=client.open_by_key(st.secrets['SHEET_ID']) if 'SHEET_ID' in st.secrets else client.open(SPREADSHEET_NAME)
        st.sidebar.success('Connected to Google Sheet')
    except gspread.SpreadsheetNotFound:
        st.error('Sheet not found. Share with service account or set SHEET_ID.'); raise
    return client, ss

REQUIRED_SHEETS={
 'MAIN':['timestamp','treatment','crop','stage','eto_yday','eto_next','kc','Kndvi','PWRF','Ks_soil','ETc_yday','ETc_next','rain_yday','forecast_rain','eff_rain_y','eff_rain_next','vwc_predawn','ndvi_rgn','ndvi_ocn','nvi','TAW','RAW','depletion_prev','depletion_start','need_mm','suggested_liters','applied_liters','decision'],
 WEATHER_SHEET:['date','ETo_mm','rain_mm'],
 CALIB_SHEET:['stage','a','b','sigma_rgn','sigma_ocn','updated_at'],
 META_SHEET:['key','value'],
 SENSOR_RAW:['timestamp','vwc'],
 WEATHER_RAW:['timestamp','Tair_C','RH','wind_ms','solar_MJ_m2','P_kPa','rain_mm']
}

def ensure_sheet_structure(ss):
    ws_main=ss.sheet1
    headers=ws_main.row_values(1)
    if not headers: ws_main.append_row(REQUIRED_SHEETS['MAIN'])
    else:
        missing=[h for h in REQUIRED_SHEETS['MAIN'] if h not in headers]
        if missing: ws_main.update('1:1',[headers+missing])
    def ensure(title, hdrs):
        try: ws=ss.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            ws=ss.add_worksheet(title=title, rows=8000, cols=max(10,len(hdrs))); ws.append_row(hdrs); return ws
        cur=ws.row_values(1)
        if not cur: ws.append_row(hdrs)
        else:
            miss=[h for h in hdrs if h not in cur]
            if miss: ws.update('1:1',[cur+miss])
        return ws
    ws_weather=ensure(WEATHER_SHEET, REQUIRED_SHEETS[WEATHER_SHEET])
    ws_cal=ensure(CALIB_SHEET, REQUIRED_SHEETS[CALIB_SHEET])
    ws_meta=ensure(META_SHEET, REQUIRED_SHEETS[META_SHEET])
    ws_sraw=ensure(SENSOR_RAW, REQUIRED_SHEETS[SENSOR_RAW])
    ws_wraw=ensure(WEATHER_RAW, REQUIRED_SHEETS[WEATHER_RAW])
    if not ws_cal.get_all_records():
        for stg in ['initial','mid','late']:
            ws_cal.append_row([stg,0.0,1.0,0.03,0.03,datetime.utcnow().isoformat()])
    if not ws_meta.get_all_records():
        ws_meta.append_row(['app_version','3.9']); ws_meta.append_row(['last_update', str(date.today())]); ws_meta.append_row(['researcher','Patrick'])
    return ws_main, ws_weather, ws_cal, ws_meta, ws_sraw, ws_wraw

def sheet_to_df(ws):
    try: data=ws.get_all_records(); return pd.DataFrame(data) if data else pd.DataFrame()
    except: return pd.DataFrame()

def append_row(ws, dct, header): ws.append_row([dct.get(h,'') for h in header])

def svp(T): return 0.6108*np.exp((17.27*T)/(T+237.3))
def slope_vpc(T): es=svp(T); return 4098*es/((T+237.3)**2)
def psy(P): return 0.000665*P
def eto_hourly(T,RH,Rs,u2,P):
    es=svp(T); ea=es*max(min(RH/100.0,1.0),0.0)
    delta=slope_vpc(T); gamma=psy(P)
    Rns=(1-0.23)*max(Rs,0.0); G=0.1*Rns if Rns>0 else 0.0
    num=0.408*delta*(Rns-G)+gamma*(37.0/(T+273.0))*u2*(es-ea)
    den=delta+gamma*(1+0.34*u2)
    return max(0.0, num/max(den,1e-6))

def ndvi_fuse(rgn,ocn,a=0.0,b=1.0,sr=0.03,so=0.03):
    r=None if rgn is None or rgn<=0 else float(rgn)
    o=None if ocn is None or ocn<=0 else float(ocn)
    oc=a+b*o if o is not None else None
    if r is not None and oc is not None:
        wr=1.0/(sr**2); wo=1.0/(so**2); s=max(wr+wo,1e-6); wr/=s; wo/=s; return float(np.clip(wr*r+wo*oc,0,1))
    if r is not None: return float(np.clip(r,0,1))
    if oc is not None: return float(np.clip(oc,0,1))
    return None

def kndvi(nvi): 
    if nvi is None: return 1.0
    return float(np.clip(0.50+0.50*float(nvi),0.60,1.10))

# Sidebar
st.sidebar.header('Settings')
crop_choice=st.sidebar.selectbox('Crop', list(CROP_PARAMS.keys()), index=0)
stage=st.sidebar.selectbox('Stage', ['initial','mid','late'])
FC=st.sidebar.number_input('Field Capacity (VWC %)', value=DEFAULTS['FC'], step=0.5)
PWP=st.sidebar.number_input('Permanent Wilting Point (VWC %)', value=DEFAULTS['PWP'], step=0.5)
MAD=st.sidebar.number_input('Management Allowable Depletion (0-1)', value=DEFAULTS['MAD'], step=0.05, min_value=0.05, max_value=0.6)
EFF=st.sidebar.number_input('Irrigation Efficiency (0-1)', value=DEFAULTS['EFFICIENCY'], step=0.05, min_value=0.5, max_value=1.0)
st.sidebar.caption('Area 1.0 m² → 1 mm = 1 L')

# Connect Sheets
try:
    client, ss = connect_gsheet()
    ws_main, ws_weather, ws_cal, ws_meta, ws_sraw, ws_wraw = ensure_sheet_structure(ss)
    connected=True
except Exception as e:
    st.error(f'Failed to connect to Google Sheet: {e}'); connected=False

tabs=st.tabs(['Treatment Dashboard','Analytics'])

with tabs[0]:
    st.subheader('Treatment Dashboard')
    params=CROP_PARAMS[crop_choice]
    kc_base={'initial':params['Kc_ini'],'mid':params['Kc_mid'],'late':params['Kc_end']}[stage]
    Zr=params['root_depth']; PWRF=params['base_ETc_adj']
    TAW=(FC-PWP)*Zr*10.0; RAW=MAD*TAW; trigger_vwc=FC-MAD*(FC-PWP)

    treatment=st.radio('Select Treatment', ['1','2','3','4'], horizontal=True,
        format_func=lambda k:{'1':'T1 - Manual (no CSV)','2':'T2 - Sensor + Weather','3':'T3 - NDVI + Weather','4':'T4 - Sensor + NDVI + Weather'}[k])

    up_w = st.file_uploader('Weather CSV (hourly)', type=['csv'], key='wx') if treatment in ('2','3','4') else None
    up_s = st.file_uploader('Sensor CSV (hourly)', type=['csv'], key='sens') if treatment in ('2','4') else None
    ndvi_rgn=ndvi_ocn=None
    if treatment in ('3','4'):
        c1,c2=st.columns(2)
        with c1: ndvi_rgn=st.number_input('NDVI_RGN', value=0.0, step=0.01, min_value=0.0, max_value=1.0)
        with c2: ndvi_ocn=st.number_input('NDVI_OCN', value=0.0, step=0.01, min_value=0.0, max_value=1.0)
    forecast_rain=st.number_input('Rain forecast next 24h (mm)', value=0.0, step=0.5) if treatment in ('2','3','4') else 0.0

    logic={
        '1': '💧 T1: Manual log only.',
        '2': '💧 T2: ETo (weather) + Predawn VWC (sensor) → Ks_soil. ETc = ETo×Kc×PWRF×Ks_soil.',
        '3': '🌱 T3: ETo (weather) + Harmonized NDVI → Kndvi. ETc = ETo×Kc×PWRF×Kndvi.',
        '4': '🌿 T4: ETo + NDVI + VWC → Kndvi×Ks_soil.'
    }
    st.markdown(f"<div class='info-box'>{logic[treatment]}</div>", unsafe_allow_html=True)

    # WEATHER -> ETo
    eto_yday=0.0; eto_next=0.0; rain_yday=0.0
    if up_w is not None:
        wx=pd.read_csv(up_w)
        def find(df,names):
            lower={c.lower():c for c in df.columns}
            for n in names:
                if n.lower() in lower: return lower[n.lower()]
            for c in df.columns:
                if any(n.lower() in c.lower() for n in names): return c
            return None
        c_ts=find(wx,['timestamp','datetime','time']); c_T=find(wx,['Tair_C','T','Temp_C','Temperature']); c_RH=find(wx,['RH','Humidity','RH_%']); c_u2=find(wx,['wind_ms','u2','wind speed']); c_Rs=find(wx,['solar_MJ_m2','Rs','Solar','Solar_Radiation']); c_P=find(wx,['P_kPa','Pressure_kPa','Pressure']); c_rain=find(wx,['rain_mm','rain','precipitation'])
        ok=all([c_ts,c_T,c_RH,c_u2,c_Rs,c_P])
        if not ok: st.error('Weather CSV missing required columns.')
        else:
            wx[c_ts]=pd.to_datetime(wx[c_ts], errors='coerce'); wx=wx.dropna(subset=[c_ts]).sort_values(c_ts).reset_index(drop=True)
            # RAW save
            if connected:
                try:
                    existing=pd.DataFrame(ws_wraw.get_all_records()); have=set(pd.to_datetime(existing['timestamp'], errors='coerce').astype(str)) if not existing.empty else set()
                    added=0
                    for _,r in wx.iterrows():
                        ts=pd.to_datetime(r[c_ts]).isoformat()
                        if ts not in have:
                            ws_wraw.append_row([ts, r.get(c_T,''), r.get(c_RH,''), r.get(c_u2,''), r.get(c_Rs,''), r.get(c_P,''), r.get(c_rain,0.0) if c_rain else 0.0])
                            added+=1
                    if added: st.success(f'Saved {added} weather RAW rows.')
                except Exception as e:
                    st.warning(f'Weather RAW save issue: {e}')
            # ETo
            T=pd.to_numeric(wx[c_T], errors='coerce'); RH=pd.to_numeric(wx[c_RH], errors='coerce'); u2=pd.to_numeric(wx[c_u2], errors='coerce'); Rs=pd.to_numeric(wx[c_Rs], errors='coerce'); P=pd.to_numeric(wx[c_P], errors='coerce'); P = P*0.1 if P.mean()>20 else P
            eto_hourly=[0.0 if any(pd.isna(x) for x in (T.iloc[i],RH.iloc[i],Rs.iloc[i],u2.iloc[i],P.iloc[i])) else eto_hourly(T.iloc[i],RH.iloc[i],Rs.iloc[i],u2.iloc[i],P.iloc[i]) for i in range(len(wx))]
            wx['ETo_hourly_mm']=eto_hourly; wx['date']=pd.to_datetime(wx[c_ts]).dt.date
            daily=wx.groupby('date', as_index=False)['ETo_hourly_mm'].sum().rename(columns={'ETo_hourly_mm':'ETo_mm'})
            if c_rain:
                rain_daily=wx.groupby('date', as_index=False)[c_rain].sum().rename(columns={c_rain:'rain_mm'}); daily=daily.merge(rain_daily, on='date', how='left')
            else: daily['rain_mm']=0.0
            today=datetime.utcnow().date(); yday=today-timedelta(days=1)
            if not daily.empty:
                if yday in set(daily['date']): eto_yday=float(daily.loc[daily['date']==yday,'ETo_mm'].iloc[0]); rain_yday=float(daily.loc[daily['date']==yday,'rain_mm'].iloc[0])
                if today in set(daily['date']): eto_next=float(daily.loc[daily['date']==today,'ETo_mm'].iloc[0])
                else: eto_next=eto_yday
            if connected:
                ex=pd.DataFrame(ws_weather.get_all_records()); ex_dates=set(pd.to_datetime(ex['date'], errors='coerce').dt.date) if not ex.empty else set()
                add=0
                for _,r in daily.iterrows():
                    if r['date'] not in ex_dates:
                        ws_weather.append_row([str(r['date']), float(r['ETo_mm']), float(r['rain_mm'])]); add+=1
                if add: st.success(f'Weather_ETo updated with {add} rows.')

    # SENSOR -> predawn
    vwc_predawn=None
    if up_s is not None:
        df=pd.read_csv(up_s)
        ts_col=None
        for cand in ['timestamp','time','datetime','date_time']:
            for c in df.columns:
                if c.lower()==cand: ts_col=c; break
            if ts_col: break
        if ts_col is None: st.error("Sensor CSV needs 'timestamp'.")
        else:
            df[ts_col]=pd.to_datetime(df[ts_col], errors='coerce'); df=df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
            # RAW
            if connected:
                try:
                    ex=pd.DataFrame(ws_sraw.get_all_records()); have=set(pd.to_datetime(ex['timestamp'], errors='coerce').astype(str)) if not ex.empty else set(); cnt=0
                    if 'vwc' in df.columns:
                        for ts,val in df['vwc'].items():
                            iso=pd.to_datetime(ts).isoformat()
                            if iso not in have:
                                ws_sraw.append_row([iso, float(val)]); cnt+=1
                    if cnt: st.success(f'Saved {cnt} sensor RAW rows.')
                except Exception as e:
                    st.warning(f'Sensor RAW save issue: {e}')
            if 'vwc' in df.columns:
                predawn=df.between_time('04:00','06:00')['vwc']
                if not predawn.empty: vwc_predawn=float(predawn.mean()); st.metric('Predawn VWC (%)', f'{vwc_predawn:.1f}')
            else: st.error("Sensor CSV missing 'vwc' column.")

    # NDVI fuse
    nvi=None; Kndvi=1.0
    if treatment in ('3','4'):
        a=b=1.0; sr=so=0.03; a=0.0
        if connected:
            try:
                cal=pd.DataFrame(ws_cal.get_all_records())
                if not cal.empty:
                    row=cal[cal['stage'].str.lower()==stage]
                    if not row.empty:
                        a=float(row['a'].iloc[0]); b=float(row['b'].iloc[0]); sr=float(row['sigma_rgn'].iloc[0]); so=float(row['sigma_ocn'].iloc[0])
            except: pass
        r=ndvi_rgn if ndvi_rgn and ndvi_rgn>0 else None
        o=ndvi_ocn if ndvi_ocn and ndvi_ocn>0 else None
        nvi=ndvi_fuse(r,o,a,b,sr,so); Kndvi=kndvi(nvi) if nvi is not None else 1.0
        if nvi is not None: st.metric('Harmonized NDVI (nVI)', f'{nvi:.3f}')

    # last depletion
    def last_dep_t(ws_main, code):
        try:
            df=pd.DataFrame(ws_main.get_all_records())
            if df.empty: return None
            df['timestamp']=pd.to_datetime(df['timestamp'], errors='coerce'); df=df[df['treatment']==code].sort_values('timestamp')
            if df.empty: return None
            v=df.iloc[-1].get('depletion_start', None)
            return float(v) if v not in (None, '') else None
        except: return None
    last_dep=last_dep_t(ws_main, f'T{treatment}') if connected else None

    # compute
    eff_rain_y=0.8*(rain_yday if up_w is not None else 0.0)
    eff_rain_next=0.8*(forecast_rain if treatment in ('2','3','4') else 0.0)
    Ks_soil=1.0
    if treatment in ('2','4') and vwc_predawn is not None and FC>0: Ks_soil=float(np.clip(vwc_predawn/FC,0.3,1.0))
    Kndvi_used=Kndvi if treatment in ('3','4') else 1.0
    ETc_yday=(eto_yday*kc_base*Kndvi_used*params['base_ETc_adj']*(Ks_soil if treatment in ('2','4') else 1.0))
    ETc_next=(eto_next*kc_base*Kndvi_used*params['base_ETc_adj']*(Ks_soil if treatment in ('2','4') else 1.0))
    depletion_prev=last_dep if (last_dep is not None and last_dep>=0) else (0.5*RAW)
    depletion_start=max(0.0, depletion_prev+ETc_yday-eff_rain_y)
    need_mm=min(max(0.0, depletion_start+ETc_next-eff_rain_next), TAW)
    suggested_liters=need_mm*AREA/max(DEFAULTS['EFFICIENCY'],1e-6)

    if treatment=='1':
        decision='Manual entry'; applied=0.0; st.info('T1 - Manual logging only.')
    else:
        if forecast_rain>=RAIN_SKIP_MM: decision='Skip (Rain forecast)'; applied=0.0
        else:
            if treatment=='2': trigger=((vwc_predawn is not None and vwc_predawn <= (FC-MAD*(FC-PWP))) or (depletion_start>=RAW))
            elif treatment=='3': trigger=((nvi is not None and nvi<0.55) or (depletion_start>=RAW))
            else: trigger=((vwc_predawn is not None and vwc_predawn <= (FC-MAD*(FC-PWP))) or (depletion_start>=RAW) or (nvi is not None and nvi<0.55))
            if trigger and suggested_liters>=0.5: decision='Irrigate'; applied=suggested_liters
            else: decision='Skip'; applied=0.0

    if treatment in ('2','3','4'):
        st.markdown('### Formula & Inputs')
        st.code(f"""ETc_yday = ETo_yday × Kc × PWRF × Ks_soil × Kndvi
    = {eto_yday:.2f} × {kc_base:.2f} × {params['base_ETc_adj']:.2f} × {(Ks_soil if treatment in ('2','4') else 1.0):.2f} × {Kndvi_used:.2f} = {ETc_yday:.2f} mm
    
ETc_next = ETo_next × Kc × PWRF × Ks_soil × Kndvi
    = {eto_next:.2f} × {kc_base:.2f} × {params['base_ETc_adj']:.2f} × {(Ks_soil if treatment in ('2','4') else 1.0):.2f} × {Kndvi_used:.2f} = {ETc_next:.2f} mm

depletion_start = max(0, depletion_prev + ETc_yday - eff_rain_y)
    = max(0, {depletion_prev:.2f} + {ETc_yday:.2f} - {eff_rain_y:.2f}) = {depletion_start:.2f} mm

need_mm = clamp( depletion_start + ETc_next - eff_rain_next , 0..TAW )
    = clamp( {depletion_start:.2f} + {ETc_next:.2f} - {eff_rain_next:.2f} , 0..{TAW:.2f} ) = {need_mm:.2f} mm

suggested_liters = need_mm × AREA / efficiency
    = {need_mm:.2f} × {AREA:.2f} / {DEFAULTS['EFFICIENCY']:.2f} = {suggested_liters:.2f} L""")
        with st.expander('Inputs used (debug)'):
            st.write({'ETo_yday(mm)':round(eto_yday,3),'ETo_next(mm)':round(eto_next,3),'Kc':round(kc_base,3),'PWRF':round(params['base_ETc_adj'],3),'Ks_soil':round(Ks_soil,3),'Kndvi':round(Kndvi_used,3),'nVI':None if nvi is None else round(nvi,3),'VWC_predawn(%)':None if vwc_predawn is None else round(vwc_predawn,2),'TAW':round(TAW,2),'RAW':round(RAW,2),'depletion_prev':round(depletion_prev,2),'rain_yday':round(rain_yday,2),'forecast_rain':round(forecast_rain,2),'eff_rain_y':round(eff_rain_y,2),'eff_rain_next':round(eff_rain_next,2)})

    if treatment!='1':
        st.markdown('---')
        if 'Irrigate' in decision: st.success(f'Decision: {decision} — Apply {applied:.2f} L')
        else: st.warning(f'Decision: {decision} — Suggested = {suggested_liters:.2f} L')

    # Auto-save MAIN (dedup simple)
    if connected and treatment!='1':
        row={'timestamp':datetime.utcnow().isoformat(),'treatment':f'T{treatment}','crop':crop_choice,'stage':stage,'eto_yday':round(eto_yday,3),'eto_next':round(eto_next,3),'kc':round(kc_base,3),'Kndvi':round((Kndvi if treatment in ('3','4') else 1.0),3),'PWRF':round(params['base_ETc_adj'],3),'Ks_soil':round(Ks_soil,3),'ETc_yday':round(ETc_yday,3),'ETc_next':round(ETc_next,3),'rain_yday':round(rain_yday,3),'forecast_rain':round(forecast_rain,3),'eff_rain_y':round(eff_rain_y,3),'eff_rain_next':round(eff_rain_next,3),'vwc_predawn':vwc_predawn if vwc_predawn is not None else '','ndvi_rgn':ndvi_rgn if ndvi_rgn is not None else '','ndvi_ocn':ndvi_ocn if ndvi_ocn is not None else '','nvi':(None if nvi is None else round(nvi,3)),'TAW':round(TAW,2),'RAW':round(RAW,2),'depletion_prev':round(depletion_prev,2),'depletion_start':round(depletion_start,2),'need_mm':round(need_mm,2),'suggested_liters':round(suggested_liters,2),'applied_liters':round(applied,2),'decision':decision}
        try:
            ex=pd.DataFrame(ws_main.get_all_records())
            write=True
            if not ex.empty:
                ex['timestamp']=pd.to_datetime(ex['timestamp'], errors='coerce')
                recent=ex[ex['treatment']==row['treatment']].sort_values('timestamp').tail(1)
                if not recent.empty:
                    r=recent.iloc[0]
                    if (r.get('decision')==row['decision']) and (abs(float(r.get('suggested_liters',0))-row['suggested_liters'])<1e-6):
                        write=False
            if write: ws_main.append_row([row[k] for k in REQUIRED_SHEETS['MAIN']]); st.success('Auto-saved to MAIN ✅')
            else: st.info('No change since last save (skipped duplicate).')
        except Exception as e:
            st.warning(f'Auto-save issue: {e}')

with tabs[1]:
    st.subheader('Analytics')
    if not connected: st.info('Connect Google Sheets to view analytics.')
    else:
        df=pd.DataFrame(ws_main.get_all_records())
        if df.empty: st.info('No logs yet.')
        else:
            if 'timestamp' in df.columns: df['timestamp']=pd.to_datetime(df['timestamp'], errors='coerce')
            for c in ['vwc_predawn','nvi','applied_liters','eto_yday','eto_next','ETc_next','suggested_liters','forecast_rain']:
                if c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
            df=df.sort_values('timestamp')
            ftr=st.selectbox('Filter by treatment', options=['All','T1','T2','T3','T4'], index=0)
            dff=df if ftr=='All' else df[df['treatment']==ftr]
            if 'suggested_liters' in dff.columns and dff['suggested_liters'].notna().any():
                fig,ax=plt.subplots()
                ax.plot(dff['timestamp'], dff['suggested_liters'].fillna(0), marker='o', label='Suggested liters')
                if 'ETc_next' in dff.columns: ax.plot(dff['timestamp'], dff['ETc_next'].fillna(0), linestyle='--', label='ETc (mm) next')
                if 'forecast_rain' in dff.columns: ax.bar(dff['timestamp'], (0.8*dff['forecast_rain']).fillna(0), alpha=0.3, label='Eff. rain next (mm)')
                ax.set_title('Decision Components & Suggested Liters'); ax.grid(True, alpha=0.3); ax.legend()
                st.pyplot(fig)
