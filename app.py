import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
import ee
import google.generativeai as genai
import pandas as pd
import numpy as np
import json
import datetime
import re
import copy
from math import radians, sin, cos, sqrt, atan2, pi

# 1. Page Configuration & Professional CSS Overhaul
st.set_page_config(layout="wide", page_title="Sahaj Urja Site Prospector", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* App Background */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 600 !important;
    }
    
    /* Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-weight: 500;
        color: #334155;
    }
    
    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        border: 1px solid #cbd5e1;
    }
    .stButton>button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    /* Primary Button Override */
    .stButton>button[kind="primary"] {
        background-color: #2563eb;
        color: white;
        border: none;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #1d4ed8;
        color: white;
    }

    /* Map Container Shadow */
    iframe {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Custom Status Banners */
    .status-banner {
        padding: 16px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .status-green { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .status-amber { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .status-red { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    
    /* Info/Warning Box Styling */
    div.stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# Custom Header
st.markdown("""
<div style="padding-bottom: 1.5rem; margin-bottom: 1.5rem; border-bottom: 2px solid #e2e8f0;">
    <h1 style="margin: 0; color: #1e3a8a; display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 2.5rem;">🌊</span> Sahaj Urja Site Prospector
    </h1>
    <p style="margin: 5px 0 0 0; color: #64748b; font-size: 1.1rem; font-weight: 500;">Arunachal Pradesh Run-of-River Triage Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Initialize GEE
if "gee_authenticated" not in st.session_state:
    try:
        json_creds = json.loads(st.secrets["GEE_SERVICE_ACCOUNT_JSON"])
        email = json_creds["client_email"]
        credentials = ee.ServiceAccountCredentials(email, key_data=st.secrets["GEE_SERVICE_ACCOUNT_JSON"])
        ee.Initialize(credentials=credentials)
        st.session_state["gee_authenticated"] = True
    except Exception as e:
        st.error(f"Google Earth Engine Authentication Failed: {str(e)}")
        st.stop()

# Initialize Gemini API
if "GENAI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GENAI_API_KEY"])
else:
    st.error("Gemini API key (GENAI_API_KEY) missing from Streamlit secrets.")
    st.stop()

# 2. Hardcoded Constants & Databases
SUBSTATIONS = [
    ('Ranganadi', 27.3426, 93.8168), ('Nirjuli', 27.1464, 93.7389), ('Ziro', 27.5381, 93.8290),
    ('Khupi', 27.2215, 92.6521), ('Pasighat', 28.0645, 95.3259), ('Roing', 28.1408, 95.8360),
    ('Tezu', 27.9174, 96.1685), ('Namsai', 27.6749, 95.8785), ('Deomali', 27.1725, 95.4682),
    ('Aalo', 28.1712, 94.8023), ('Daporijo', 27.9860, 94.2210), ('Changlang', 27.1170, 95.7430),
    ('Bomdila', 27.2640, 92.4080), ('Seppa', 27.3280, 92.8890), ('Likabali', 27.8170, 94.6710),
    ('Kharsang', 27.4130, 95.9980), ('Sagalee', 27.2720, 93.5710), ('Yazali', 27.6180, 93.7340)
]

# 3. Sidebar & History Stack Configuration
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_obj_click" not in st.session_state:
    st.session_state["last_obj_click"] = None

def push_state_to_history():
    st.session_state["history"].append({
        "workflow_state": st.session_state["workflow_state"],
        "site_data": copy.deepcopy(st.session_state["site_data"])
    })
    if len(st.session_state["history"]) > 15:
        st.session_state["history"].pop(0)

def pop_state_from_history():
    if st.session_state["history"]:
        last_state = st.session_state["history"].pop()
        st.session_state["workflow_state"] = last_state["workflow_state"]
        st.session_state["site_data"] = last_state["site_data"]
        st.session_state["last_processed_click"] = None
        st.session_state["last_obj_click"] = None

st.sidebar.header("Map Navigation")
undo_btn = st.sidebar.button("↩️ Undo Last Action (Ctrl + Z)", use_container_width=True)
if undo_btn:
    pop_state_from_history()
    st.rerun()

st.sidebar.header("Configuration Parameters")
cfg_min_mw = st.sidebar.number_input("Policy Target Minimum (MW)", value=5.0, step=0.5)
cfg_max_mw = st.sidebar.number_input("Ideal Target Ceiling (MW)", value=12.0, step=0.5)
cfg_efficiency = st.sidebar.slider("Combined Efficiency (η)", 0.70, 0.90, 0.85, 0.01)
cfg_head_loss = st.sidebar.slider("Head Loss Factor (h_f)", 0.05, 0.25, 0.13, 0.01)
cfg_terrain_mult = st.sidebar.slider("Penstock Terrain Multiplier", 1.10, 1.80, 1.45, 0.05)
cfg_yield_west = st.sidebar.number_input("Specific Yield West (<94°E)", value=0.025, format="%.4f")
cfg_yield_east = st.sidebar.number_input("Specific Yield East (≥94°E)", value=0.035, format="%.4f")
cfg_default_plf = st.sidebar.slider("Base PLF Default", 0.40, 0.70, 0.60, 0.05)
cfg_substation_limit = st.sidebar.number_input("Max Substation Distance Buffer (km)", value=20.0, step=5.0)
cfg_is_apst = st.sidebar.checkbox("APST Local Partner Active", value=True)
cfg_road_km = st.sidebar.number_input("Assumed Access Road Construction (km)", value=2.0, step=0.5)
cfg_gemini_model = st.sidebar.selectbox("Gemini Model Version", ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"])

# 4. Helper Math, DMS Parsers & Engines
def parse_combined_coords(coord_str):
    if not coord_str: return None, None
    coord_str = str(coord_str).strip()
    
    cleaned = coord_str.replace("''", '"').replace('°', ' ').replace("'", ' ').replace('"', ' ').strip()
    matches = re.findall(r'(\d+)\s+(\d+)\s+([\d\.]+)\s*([NSEWnsew])', cleaned)
    
    if len(matches) >= 2:
        lat_dd = float(matches[0][0]) + float(matches[0][1])/60.0 + float(matches[0][2])/3600.0
        if matches[0][3].upper() in ['S', 'W']: lat_dd = -lat_dd
        
        lng_dd = float(matches[1][0]) + float(matches[1][1])/60.0 + float(matches[1][2])/3600.0
        if matches[1][3].upper() in ['S', 'W']: lng_dd = -lng_dd
        return lat_dd, lng_dd
    
    try:
        parts = [p for p in re.split(r'[,\s]+', coord_str) if p]
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except:
        pass
    return None, None

def format_dms(dd, is_lat):
    if dd is None: return ""
    direction = ('N' if dd >= 0 else 'S') if is_lat else ('E' if dd >= 0 else 'W')
    dd = abs(dd)
    d = int(dd)
    m = int((dd - d) * 60)
    s = (dd - d - m/60) * 3600
    return f"{d}°{m}'{s:.2f}\"{direction}"

def format_combined_dms(lat, lng):
    if lat is None or lng is None: return ""
    return f"{format_dms(lat, True)} {format_dms(lng, False)}"

def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    dlat, dlng = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def compute_capex(mw, penstock_m, transmission_km, road_km, is_apst):
    civil = 6.5 * (mw ** 0.7)
    em = 2.4 * mw
    penstock = 0.00103 * (mw ** 0.5) * penstock_m
    cost_per_km = 0.25 if mw <= 10.0 else 0.70
    transmission = transmission_km * cost_per_km
    road = road_km * 1.0
    land = (1.0 + 0.2 * mw) if is_apst else (6.0 + 0.8 * mw)
    predev = 1.0 + 0.1 * mw
    subtotal = civil + em + penstock + transmission + road + land + predev
    contingency = 0.10 * subtotal
    total = subtotal + contingency
    return {'civil': civil, 'em': em, 'penstock': penstock, 'transmission': transmission,
            'road': road, 'land': land, 'predev': predev, 'contingency': contingency,
            'total': total, 'per_mw': total / mw if mw > 0 else 0}

def process_intake(click_lat, click_lng):
    pt = ee.Geometry.Point([click_lng, click_lat])
    merit_hydro = ee.Image('MERIT/Hydro/v1_0_1')
    try:
        upa_val = merit_hydro.select('upa').sample(pt, scale=90).first().get('upa').getInfo()
        if upa_val is None or upa_val < 10.0:
            buffer = pt.buffer(250)
            clipped = merit_hydro.select('upa').clip(buffer)
            max_dict = clipped.reduceRegion(ee.Reducer.max(), buffer, 90).getInfo()
            if max_dict and max_dict.get('upa') and max_dict.get('upa') >= 10.0:
                upa_val = max_dict.get('upa')
        
        if upa_val and upa_val >= 10.0:
            st.session_state["site_data"]["intake_lat"] = click_lat
            st.session_state["site_data"]["intake_lng"] = click_lng
            st.session_state["site_data"]["catchment_km2"] = upa_val
            st.session_state["workflow_state"] = "CONFIRM_INTAKE"
            return True
        else:
            st.error("Catchment area falls below 10 sq km baseline. Re-click nearer a verified stream reach.")
            return False
    except Exception as ex:
        st.error(f"GEE processing anomaly: {str(ex)}")
        return False

def process_calculations(intake_lat, intake_lng, catchment_km2, ph_lat, ph_lng):
    try:
        dem = ee.Image('MERIT/DEM/v1_0_3').select('dem')
        elev_in = dem.sample(ee.Geometry.Point([intake_lng, intake_lat]), scale=90).first().get('dem').getInfo()
        elev_p = dem.sample(ee.Geometry.Point([ph_lng, ph_lat]), scale=90).first().get('dem').getInfo()
        
        gross_h = elev_in - elev_p
        if gross_h <= 0:
            return {"error": f"Hydraulic violation: Powerhouse elevation ({elev_p}m) equal to or higher than Intake ({elev_in}m)."}
        
        net_h = gross_h * (1.0 - cfg_head_loss)
        specific_yield = cfg_yield_west if intake_lng < 94.0 else cfg_yield_east
        q_design = catchment_km2 * specific_yield
        p_mw = q_design * net_h * 9.81 * cfg_efficiency / 1000.0
        
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate('2000-01-01', '2024-12-31').filter(ee.Filter.calendarRange(1, 3, 'month'))
        mean_lean_daily = chirps.mean().reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([intake_lng, intake_lat]).buffer(5000), 5500).get('precipitation').getInfo()
        
        effective_plf = cfg_default_plf * 0.5 if (mean_lean_daily and (mean_lean_daily * 30.0) < 40.0) else cfg_default_plf
        annual_mwh = p_mw * 8760.0 * effective_plf
        
        straight_line_km = haversine_km(intake_lat, intake_lng, ph_lat, ph_lng)
        penstock_m = straight_line_km * 1000.0 * cfg_terrain_mult
        penstock_dia_mm = round(sqrt(4.0 * q_design / (pi * 4.0)) * 1000.0 / 50.0) * 50
        turbine_type = "Pelton (Vertical)" if net_h > 200.0 else ("Francis (Horizontal)" if net_h > 30.0 else "Kaplan / Crossflow")
        num_units = 2 if p_mw >= 4.0 else 1
        
        distances = [(name, haversine_km(ph_lat, ph_lng, s_lat, s_lng)) for name, s_lat, s_lng in SUBSTATIONS]
        near_sub, near_dist = min(distances, key=lambda x: x[1])
        voltage = 33 if p_mw <= 10.0 else 132
        
        wdpa = ee.FeatureCollection('WCMC/WDPA/current/polygons')
        within_wdpa = wdpa.filterBounds(ee.Geometry.Point([intake_lng, intake_lat])).size().getInfo() > 0
        
        if within_wdpa: flag, reason = "RED", "WDPA Protected Area Exclusion Area"
        elif p_mw < cfg_min_mw: flag, reason = "RED", f"Below Policy Target Threshold ({p_mw:.2f} MW < {cfg_min_mw} MW)"
        elif p_mw > cfg_max_mw: flag, reason = "AMBER", f"Exceeds Target Ceiling Profile ({p_mw:.2f} MW > {cfg_max_mw} MW)"
        elif near_dist > cfg_substation_limit: flag, reason = "AMBER", f"Grid Isolation Warning ({near_dist:.1f} km > {cfg_substation_limit} km)"
        else: flag, reason = "GREEN", "Optimal Hydrological & Civil Target Envelope"
            
        capex_data = compute_capex(p_mw, penstock_m, near_dist, cfg_road_km, cfg_is_apst)
        if capex_data['per_mw'] > 13.0 and flag == "GREEN": flag, reason = "AMBER", f"CAPEX Outlier Projections (₹{capex_data['per_mw']:.2f} Cr/MW)"

        return {"intake_lat": intake_lat, "intake_lng": intake_lng, "ph_lat": ph_lat, "ph_lng": ph_lng,
                "elev_intake": elev_in, "elev_ph": elev_p, "catchment_km2": catchment_km2,
                "gross_head_m": gross_h, "net_head_m": net_h, "q_design_cumecs": q_design, "capacity_MW": p_mw,
                "annual_MWh": annual_mwh, "plf": effective_plf * 100, "penstock_m": penstock_m,
                "penstock_dia_mm": penstock_dia_mm, "turbine_type": turbine_type, "num_units": num_units,
                "nearest_substation": near_sub, "substation_km": near_dist, "voltage_kv": voltage,
                "within_wdpa": within_wdpa, "flag": flag, "flag_reason": reason, "capex": capex_data,
                "specific_yield": specific_yield}
    except Exception as ex:
        return {"error": str(ex)}

# 5. State Machine & Map Centering
if "workflow_state" not in st.session_state:
    st.session_state["workflow_state"] = "AWAITING_INTAKE"
    st.session_state["site_data"] = {}

map_center, map_zoom = [28.0, 94.5], 7
sd = st.session_state["site_data"]

if "intake_lat" in sd and "ph_lat" not in sd:
    map_center, map_zoom = [sd["intake_lat"], sd["intake_lng"]], 13
elif "ph_lat" in sd and "intake_lat" not in sd:
    map_center, map_zoom = [sd["ph_lat"], sd["ph_lng"]], 13
elif "intake_lat" in sd and "ph_lat" in sd:
    map_center, map_zoom = [(sd["intake_lat"] + sd["ph_lat"]) / 2, (sd["intake_lng"] + sd["ph_lng"]) / 2], 12

# 6. User Interface Layout
col_map, col_dash = st.columns([0.55, 0.45], gap="large")

with col_map:
    st.markdown("<h3 style='margin-bottom: 15px;'>Geospatial Triage Map</h3>", unsafe_allow_html=True)
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles=None)
    
    # Layer Overrides
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google", name="High-Res Satellite Hybrid (Labels + Borders)", show=True
    ).add_to(m)
    folium.TileLayer("OpenTopoMap", name="Topographic Relief Map", show=False).add_to(m)
    
    # Crosshair cursor injection
    if st.session_state["workflow_state"] in ["RELOCATE_INTAKE", "RELOCATE_POWERHOUSE"]:
        m.get_root().html.add_child(folium.Element("<style>.leaflet-container { cursor: crosshair !important; }</style>"))
    
    # Substations off by default
    substation_group = folium.FeatureGroup(name="18 Verified Substation Radii", show=False).add_to(m)
    for name, s_lat, s_lng in SUBSTATIONS:
        folium.Marker([s_lat, s_lng], tooltip=f"Substation: {name}", icon=folium.Icon(color="orange", icon="flash")).add_to(substation_group)
        folium.Circle([s_lat, s_lng], radius=cfg_substation_limit * 1000, color="orange", fill=False, dash_array="5, 5").add_to(substation_group)
        
    if "intake_lat" in sd:
        folium.Marker([sd["intake_lat"], sd["intake_lng"]], tooltip="Click to Relocate Intake Pin", popup="Proposed Weir Intake", icon=folium.Icon(color="green", icon="cloud")).add_to(m)
    if "ph_lat" in sd:
        folium.Marker([sd["ph_lat"], sd["ph_lng"]], tooltip="Click to Relocate Powerhouse Pin", popup="Proposed Powerhouse Point", icon=folium.Icon(color="red", icon="bolt")).add_to(m)
    
    if "intake_lat" in sd and "ph_lat" in sd:
        folium.PolyLine([[sd["intake_lat"], sd["intake_lng"]], [sd["ph_lat"], sd["ph_lng"]]], color="#3b82f6", weight=5, opacity=0.9, dash_array="6, 6").add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    
    # Capture Map & Object Clicks
    map_output = st_folium(m, width="100%", height=680, key="prospector_map")
    
    if map_output:
        obj_click = map_output.get("last_object_clicked")
        map_click = map_output.get("last_clicked")
        
        # 1. Did the user click an existing pin? (Enter Relocation Mode)
        if obj_click and st.session_state.get("last_obj_click") != obj_click:
            st.session_state["last_obj_click"] = obj_click
            c_lat = obj_click["lat"]
            
            if "intake_lat" in sd and abs(sd["intake_lat"] - c_lat) < 0.005:
                push_state_to_history()
                st.session_state["workflow_state"] = "RELOCATE_INTAKE"
                st.rerun()
            elif "ph_lat" in sd and abs(sd["ph_lat"] - c_lat) < 0.005:
                push_state_to_history()
                st.session_state["workflow_state"] = "RELOCATE_POWERHOUSE"
                st.rerun()
                
        # 2. Did the user click the map? (Drop the pin)
        elif map_click and st.session_state.get("last_processed_click") != map_click:
            st.session_state["last_processed_click"] = map_click
            click_lat, click_lng = map_click["lat"], map_click["lng"]
            
            # Dropping an Intake Pin
            if st.session_state["workflow_state"] in ["AWAITING_INTAKE", "CONFIRM_INTAKE", "RELOCATE_INTAKE"]:
                push_state_to_history()
                if process_intake(click_lat, click_lng):
                    if "ph_lat" in sd:
                        res = process_calculations(click_lat, click_lng, st.session_state["site_data"]["catchment_km2"], sd["ph_lat"], sd["ph_lng"])
                        if "error" not in res:
                            st.session_state["site_data"] = res
                            st.session_state["workflow_state"] = "COMPLETE"
                    st.rerun()
                    
            # Dropping a Powerhouse Pin
            elif st.session_state["workflow_state"] in ["AWAITING_POWERHOUSE", "RELOCATE_POWERHOUSE"]:
                push_state_to_history()
                res = process_calculations(sd["intake_lat"], sd["intake_lng"], sd["catchment_km2"], click_lat, click_lng)
                if "error" in res: st.error(res["error"])
                else:
                    st.session_state["site_data"] = res
                    st.session_state["workflow_state"] = "COMPLETE"
                    st.rerun()

with col_dash:
    st.markdown("<h3 style='margin-bottom: 15px;'>Evaluation Engine</h3>", unsafe_allow_html=True)
    
    # Dynamic Contextual Instructions
    if st.session_state["workflow_state"] == "AWAITING_INTAKE":
        st.info("🎯 **Action Required:** Click the map to plot the Intake, OR enter precise coordinates below.")
        
    elif st.session_state["workflow_state"] == "CONFIRM_INTAKE":
        st.success(f"**Intake pinpointed.** Extracted Catchment: **{sd['catchment_km2']:.1f} sq km**.")
        st.info("⚠️ Is the weir site accurately placed on the map? If not, click the pin to unlock it, then click the map to relocate. Once satisfied, click Lock.")
        if st.button("Lock Intake Location & Proceed", type="primary", use_container_width=True):
            push_state_to_history()
            st.session_state["workflow_state"] = "AWAITING_POWERHOUSE"
            st.rerun()
            
    elif st.session_state["workflow_state"] == "AWAITING_POWERHOUSE":
        st.success("🔒 **Intake locked.**")
        st.info("🎯 **Action Required:** Click the map to drop the Powerhouse pin, OR enter precise coordinates below.")
        
    elif st.session_state["workflow_state"] == "RELOCATE_INTAKE":
        st.warning("📍 **RELOCATION MODE:** The map cursor is a crosshair. Click anywhere on the map to drop the INTAKE pin at its new location.")
        
    elif st.session_state["workflow_state"] == "RELOCATE_POWERHOUSE":
        st.warning("📍 **RELOCATION MODE:** The map cursor is a crosshair. Click anywhere on the map to drop the POWERHOUSE pin at its new location.")

    # Manual Input Box
    with st.expander("⚙️ Manual Coordinate Overrides", expanded=(st.session_state["workflow_state"] in ["AWAITING_INTAKE", "AWAITING_POWERHOUSE"])):
        in_coord_val = st.text_input("Intake Coordinates", value=format_combined_dms(sd.get("intake_lat"), sd.get("intake_lng")), placeholder="27°54'16.27\"N 94°06'07.18\"E")
        
        ph_coord_val = ""
        if st.session_state["workflow_state"] in ["AWAITING_POWERHOUSE", "COMPLETE", "RELOCATE_POWERHOUSE", "RELOCATE_INTAKE"]:
            ph_coord_val = st.text_input("Powerhouse Coordinates", value=format_combined_dms(sd.get("ph_lat"), sd.get("ph_lng")), placeholder="27°50'12.00\"N 94°08'05.00\"E")
        
        if st.button("Execute Manual Coordinates"):
            push_state_to_history()
            p_in_lat, p_in_lng = parse_combined_coords(in_coord_val)
            p_ph_lat, p_ph_lng = parse_combined_coords(ph_coord_val) if ph_coord_val else (None, None)
            
            if p_in_lat and p_in_lng:
                if not (p_ph_lat and p_ph_lng):
                    if process_intake(p_in_lat, p_in_lng): st.rerun()
                else:
                    if "catchment_km2" not in sd: process_intake(p_in_lat, p_in_lng)
                    res = process_calculations(p_in_lat, p_in_lng, st.session_state["site_data"].get("catchment_km2", 100), p_ph_lat, p_ph_lng)
                    if "error" in res: st.error(res["error"])
                    else:
                        st.session_state["site_data"] = res
                        st.session_state["workflow_state"] = "COMPLETE"
                        st.rerun()
            else:
                st.error("Invalid coordinate string provided. Ensure it matches format 27°54'16\"N 94°06'07\"E")

    if st.session_state["workflow_state"] != "AWAITING_INTAKE":
        rc1, rc2 = st.columns(2)
        if rc1.button("Reset Intake Node"):
            push_state_to_history()
            st.session_state["site_data"].pop("intake_lat", None)
            st.session_state["site_data"].pop("intake_lng", None)
            st.session_state["site_data"].pop("catchment_km2", None)
            st.session_state["workflow_state"] = "AWAITING_INTAKE"
            st.rerun()
        if rc2.button("Reset Powerhouse Node"):
            push_state_to_history()
            st.session_state["site_data"].pop("ph_lat", None)
            st.session_state["site_data"].pop("ph_lng", None)
            st.session_state["workflow_state"] = "AWAITING_POWERHOUSE"
            st.rerun()

    if st.session_state["workflow_state"] == "COMPLETE":
        st.markdown("<br>", unsafe_allow_html=True)
        f_class = "status-green" if sd["flag"] == "GREEN" else ("status-amber" if sd["flag"] == "AMBER" else "status-red")
        st.markdown(f"""
        <div class="status-banner {f_class}">
            { "✅" if sd["flag"] == "GREEN" else "⚠️" } TRIAGE STATUS: {sd['flag']} — {sd['flag_reason']}
        </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Capacity", f"{sd['capacity_MW']:.2f} MW")
        m2.metric("Annual Gen", f"{sd['annual_MWh']:.0f} MWh")
        m3.metric("CAPEX / MW", f"₹{sd['capex']['per_mw']:.2f} Cr")
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("#### Engineering Architecture")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Catchment Area:** {sd['catchment_km2']:.2f} sq km")
                st.write(f"**Gross Head:** {sd['gross_head_m']:.1f} m")
                st.write(f"**Net Design Head:** {sd['net_head_m']:.1f} m")
            with col_b:
                st.write(f"**Design Flow (Q):** {sd['q_design_cumecs']:.2f} m³/s")
                st.write(f"**Penstock:** {sd['penstock_m']:.0f} m (Dia: {sd['penstock_dia_mm']} mm)")
                st.write(f"**Prime Mover:** {sd['num_units']}x {sd['turbine_type']}")
            
            st.divider()
            st.write(f"🔌 **Grid Connection:** {sd['nearest_substation']} Substation at {sd['substation_km']:.1f} km ({sd['voltage_kv']} kV)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        river_input = st.text_input("Stream/Project Identification Name")
        action_col1, action_col2 = st.columns(2)
        
        if action_col1.button("Synthesize Feasibility Report", type="primary", use_container_width=True):
            with st.spinner("Compiling Desktop Pre-Feasibility Note via Gemini Engine..."):
                model = genai.GenerativeModel(cfg_gemini_model)
                site_context_payload = {
                    "topography": {"catchment_sq_km": sd["catchment_km2"], "gross_head_m": sd["gross_head_m"], "net_head_m": sd["net_head_m"]},
                    "hydrology": {"design_flow_cumecs": sd["q_design_cumecs"], "specific_yield": sd["specific_yield"], "projected_annual_generation_mwh": sd["annual_MWh"]},
                    "electro_mechanical": {"capacity_mw": sd["capacity_MW"], "turbine_type": sd["turbine_type"], "unit_count": sd["num_units"]},
                    "infrastructure": {"penstock_length_m": sd["penstock_m"], "nearest_substation": sd["nearest_substation"], "interconnection_distance_km": sd["substation_km"]},
                    "financials": {"total_capex_cr": sd['capex']["total"], "capex_per_mw_cr": sd['capex']["per_mw"], "mnre_subsidy_eligibility_cr": min(30.0, 3.6 * sd["capacity_MW"])}
                }
                
                prompt = f"""
                You are a senior technical lead engineer. Construct an exhaustive, professional Desktop Pre-Feasibility Note based strictly upon the provided raw engineering calculations payload below. 
                RAW CALCULATIONS PAYLOAD:
                {json.dumps(site_context_payload, indent=2)}
                
                Explicitly label the header note exactly as follows:
                # DESKTOP PRE-FEASIBILITY NOTE
                ## ⚠️ UNVERIFIED DESKTOP ANALYSIS — NOT A FORMAL SALIENT FEATURES DOCUMENT
                **Estimates based on remote sensing and parametric models. Field validation required before any commitment.**
                
                Compile 8 explicit sections containing deep analytical prose based on the data:
                   - SECTION 1: LOCATION & GEOSPATIAL BOUNDARIES
                   - SECTION 2: HYDROLOGICAL PROFILE & INFRASTRUCTURE CONSTRAINTS
                   - SECTION 3: CIVIL WORKS CONFIGURATION PARAMETERS
                   - SECTION 4: POWER GENERATION SCHEMATIC & EQUIPMENT DESIGN SELECTION
                   - SECTION 5: GRID INTERCONNECTION & POWER EVACUATION ANALYSIS
                   - SECTION 6: ECONOMIC FEASIBILITY & DEVELOPMENT ROADMAP
                   - SECTION 7: FIELD VALIDATION CRITICAL CHECKLIST
                   - SECTION 8: CAVEATS & TECHNICAL LIMITATIONS
                """
                response = model.generate_content(prompt)
                st.session_state["feasibility_note"] = response.text
                
        if "feasibility_note" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["feasibility_note"])
            st.download_button("Download Markdown Report", data=st.session_state["feasibility_note"], file_name=f"Sahaj_Urja_PFN_{river_input or 'Unassigned'}.md", use_container_width=True)

# HTML/JS Injection for Ctrl+Z Global Shortcut
components.html(
    """
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key.toLowerCase() === 'z') {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const undoBtn = buttons.find(b => b.innerText.includes('Undo Last Action'));
            if (undoBtn) undoBtn.click();
        }
    });
    </script>
    """,
    height=0, width=0
)
