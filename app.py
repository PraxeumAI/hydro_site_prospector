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
import os
from math import radians, sin, cos, sqrt, atan2, pi

# 1. Page Configuration & CSS Fixes (Native Background)
st.set_page_config(layout="wide", page_title="Sahaj Urja Site Prospector", initial_sidebar_state="collapsed")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    white-space: nowrap !important;
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Sahaj Urja Site Prospector — Arunachal Pradesh SHP Triage")

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
if st.sidebar.button("↩️ Undo Last Action (Ctrl + Z)", use_container_width=True):
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
        if len(parts) >= 2: return float(parts[0]), float(parts[1])
    except: pass
    return None, None

def format_dms(dd, is_lat):
    if dd is None: return ""
    direction = ('N' if dd >= 0 else 'S') if is_lat else ('E' if dd >= 0 else 'W')
    dd = abs(dd)
    d, m = int(dd), int((dd - int(dd)) * 60)
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
    return {'Civil Works': civil, 'Electro-Mechanical': em, 'Penstock': penstock, 
            'Transmission': transmission, 'Access Roads': road, 'Land Acquisition': land, 
            'Pre-Dev / Clearances': predev, 'Contingency': contingency,
            'Total': total, 'Per_MW': total / mw if mw > 0 else 0}

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
                max_upa = max_dict.get('upa')
                max_pixel_img = clipped.where(clipped.eq(max_upa), 1).selfMask()
                vectors = max_pixel_img.reduceToVectors(geometry=buffer, scale=90, geometryType='centroid')
                snapped_coords = vectors.first().geometry().coordinates().getInfo()
                click_lng, click_lat = snapped_coords[0], snapped_coords[1]
                upa_val = max_upa
                st.toast("Pin snapped to nearest high-flow channel.")
        
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
        
        effective_plf = cfg_default_plf * 0.4 if (mean_lean_daily and (mean_lean_daily * 30.0) < 40.0) else cfg_default_plf
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
        project_line = ee.Geometry.LineString([[intake_lng, intake_lat], [ph_lng, ph_lat]])
        within_wdpa = wdpa.filterBounds(project_line).size().getInfo() > 0
        
        if within_wdpa: flag, reason = "RED", "WDPA Protected Area Exclusion (Alignment intersects reserve)"
        elif p_mw < cfg_min_mw: flag, reason = "RED", f"Below Policy Target Threshold ({p_mw:.2f} MW < {cfg_min_mw} MW)"
        elif p_mw > cfg_max_mw: flag, reason = "AMBER", f"Exceeds Target Ceiling Profile ({p_mw:.2f} MW > {cfg_max_mw} MW)"
        elif near_dist > cfg_substation_limit: flag, reason = "AMBER", f"Grid Isolation Warning ({near_dist:.1f} km > {cfg_substation_limit} km)"
        else: flag, reason = "GREEN", "Optimal Hydrological & Civil Target Envelope"
            
        capex_data = compute_capex(p_mw, penstock_m, near_dist, cfg_road_km, cfg_is_apst)
        if capex_data['Per_MW'] > 13.0 and flag == "GREEN": flag, reason = "AMBER", f"CAPEX Outlier Projections (₹{capex_data['Per_MW']:.2f} Cr/MW)"

        mnre_cfa = min(30.0, 3.6 * p_mw)
        dscr_est = (annual_mwh * 4.50 / 10000000) / ((capex_data['Total'] * 0.70) * 0.0975 / (1 - (1.0975)**-15))

        return {"intake_lat": intake_lat, "intake_lng": intake_lng, "ph_lat": ph_lat, "ph_lng": ph_lng,
                "elev_intake": elev_in, "elev_ph": elev_p, "catchment_km2": catchment_km2,
                "gross_head_m": gross_h, "net_head_m": net_h, "q_design_cumecs": q_design, "capacity_MW": p_mw,
                "annual_MWh": annual_mwh, "plf": effective_plf * 100, "penstock_m": penstock_m,
                "penstock_dia_mm": penstock_dia_mm, "turbine_type": turbine_type, "num_units": num_units,
                "nearest_substation": near_sub, "substation_km": near_dist, "voltage_kv": voltage,
                "within_wdpa": within_wdpa, "flag": flag, "flag_reason": reason, "capex": capex_data,
                "specific_yield": specific_yield, "mnre_cfa_cr": mnre_cfa, "dscr": dscr_est}
    except Exception as ex:
        return {"error": str(ex)}

# 5. Initialization
if "workflow_state" not in st.session_state:
    st.session_state["workflow_state"] = "AWAITING_INTAKE"
    st.session_state["site_data"] = {}

sd = st.session_state["site_data"]

# 6. Map State Logic (Stable Centering)
map_center = [28.0, 94.5]
map_zoom = 7

if "intake_lat" in sd and "ph_lat" not in sd: 
    map_center, map_zoom = [sd["intake_lat"], sd["intake_lng"]], 16
elif "ph_lat" in sd and "intake_lat" not in sd: 
    map_center, map_zoom = [sd["ph_lat"], sd["ph_lng"]], 16
elif "intake_lat" in sd and "ph_lat" in sd: 
    map_center, map_zoom = [(sd["intake_lat"] + sd["ph_lat"]) / 2, (sd["intake_lng"] + sd["ph_lng"]) / 2], 14

# 7. User Interface Layout
col_map, col_dash = st.columns([0.55, 0.45])

with col_map:
    st.subheader("Interactive Basin Triage Map")
    
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles=None)
    
    folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", attr="Google", name="High-Res Satellite Hybrid", show=True).add_to(m)
    folium.TileLayer("OpenTopoMap", name="Topographic Relief Map", show=False).add_to(m)
    
    if st.session_state["workflow_state"] in ["RELOCATE_INTAKE", "RELOCATE_POWERHOUSE"]:
        m.get_root().html.add_child(folium.Element("<style>.leaflet-container { cursor: crosshair !important; }</style>"))
    
    substation_group = folium.FeatureGroup(name="18 Verified Substation Radii", show=False).add_to(m)
    for name, s_lat, s_lng in SUBSTATIONS:
        folium.Marker([s_lat, s_lng], tooltip=f"Substation: {name}", icon=folium.Icon(color="orange", icon="flash")).add_to(substation_group)
        folium.Circle([s_lat, s_lng], radius=cfg_substation_limit * 1000, color="orange", fill=False, dash_array="5, 5").add_to(substation_group)
        
    if "intake_lat" in sd: folium.Marker([sd["intake_lat"], sd["intake_lng"]], tooltip="Click to Relocate Intake Pin", popup="Proposed Weir Intake", icon=folium.Icon(color="green", icon="cloud")).add_to(m)
    if "ph_lat" in sd: folium.Marker([sd["ph_lat"], sd["ph_lng"]], tooltip="Click to Relocate Powerhouse Pin", popup="Proposed Powerhouse", icon=folium.Icon(color="red", icon="bolt")).add_to(m)
    if "intake_lat" in sd and "ph_lat" in sd: folium.PolyLine([[sd["intake_lat"], sd["intake_lng"]], [sd["ph_lat"], sd["ph_lng"]]], color="blue", weight=4, opacity=0.8, dash_array="6, 6").add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    
    # We only request clicks from Streamlit-Folium to prevent bounce loops
    map_output = st_folium(m, width="100%", height=650, key="prospector_map", returned_objects=["last_clicked", "last_object_clicked"])
    
    if map_output:
        obj_click = map_output.get("last_object_clicked")
        map_click = map_output.get("last_clicked")
        
        if obj_click and st.session_state.get("last_obj_click") != obj_click:
            st.session_state["last_obj_click"] = obj_click
            c_lat, c_lng = obj_click["lat"], obj_click["lng"]
            
            if "intake_lat" in sd and abs(sd["intake_lat"] - c_lat) < 0.005 and abs(sd["intake_lng"] - c_lng) < 0.005:
                push_state_to_history()
                st.session_state["workflow_state"] = "RELOCATE_INTAKE"
                st.rerun()
            elif "ph_lat" in sd and abs(sd["ph_lat"] - c_lat) < 0.005 and abs(sd["ph_lng"] - c_lng) < 0.005:
                push_state_to_history()
                st.session_state["workflow_state"] = "RELOCATE_POWERHOUSE"
                st.rerun()
                
        elif map_click and st.session_state.get("last_processed_click") != map_click:
            st.session_state["last_processed_click"] = map_click
            click_lat, click_lng = map_click["lat"], map_click["lng"]
            
            if st.session_state["workflow_state"] in ["AWAITING_INTAKE", "CONFIRM_INTAKE", "RELOCATE_INTAKE"]:
                push_state_to_history()
                if process_intake(click_lat, click_lng):
                    if "ph_lat" in sd:
                        res = process_calculations(st.session_state["site_data"]["intake_lat"], st.session_state["site_data"]["intake_lng"], st.session_state["site_data"]["catchment_km2"], sd["ph_lat"], sd["ph_lng"])
                        if "error" not in res:
                            st.session_state["site_data"] = res
                            st.session_state["workflow_state"] = "COMPLETE"
                    st.rerun()
                    
            elif st.session_state["workflow_state"] in ["AWAITING_POWERHOUSE", "RELOCATE_POWERHOUSE"]:
                push_state_to_history()
                res = process_calculations(sd["intake_lat"], sd["intake_lng"], sd["catchment_km2"], click_lat, click_lng)
                if "error" in res: 
                    st.error(res["error"])
                else:
                    st.session_state["site_data"] = res
                    st.session_state["workflow_state"] = "COMPLETE"
                    st.rerun()

with col_dash:
    st.subheader("Coordinate Engine & Evaluation Module")
    
    if st.session_state["workflow_state"] == "AWAITING_INTAKE":
        st.info("Action Required: Click the map to plot the Intake, OR enter precise coordinates below.")
        
    elif st.session_state["workflow_state"] == "CONFIRM_INTAKE":
        st.warning(f"Intake pinpointed. Extracted Catchment: **{sd['catchment_km2']:.1f} sq km**.")
        st.info("⚠️ Is the weir site accurately placed on the map? If not, click the pin to unlock it, then click the map to relocate. Once satisfied, click Lock.")
        if st.button("✅ Lock Intake Location & Proceed", type="primary", use_container_width=True):
            push_state_to_history()
            st.session_state["workflow_state"] = "AWAITING_POWERHOUSE"
            st.rerun()
            
    elif st.session_state["workflow_state"] == "AWAITING_POWERHOUSE":
        st.success("Intake locked.")
        st.info("Action Required: Click the map to drop the Powerhouse pin, OR enter precise coordinates below.")
        
    elif st.session_state["workflow_state"] == "RELOCATE_INTAKE":
        st.warning("📍 RELOCATION MODE: Click anywhere on the map to drop the INTAKE pin.")
        
    elif st.session_state["workflow_state"] == "RELOCATE_POWERHOUSE":
        st.warning("📍 RELOCATION MODE: Click anywhere on the map to drop the POWERHOUSE pin.")

    with st.expander("Manual Coordinate Overrides (DMS or Decimal)", expanded=(st.session_state["workflow_state"] in ["AWAITING_INTAKE", "AWAITING_POWERHOUSE"])):
        in_coord_val = st.text_input("Intake Coordinates", value=format_combined_dms(sd.get("intake_lat"), sd.get("intake_lng")), placeholder="27°54'16.27\"N 94°06'07.18\"E")
        
        ph_coord_val = ""
        if st.session_state["workflow_state"] in ["AWAITING_POWERHOUSE", "COMPLETE", "RELOCATE_POWERHOUSE", "RELOCATE_INTAKE"]:
            ph_coord_val = st.text_input("Powerhouse Coordinates", value=format_combined_dms(sd.get("ph_lat"), sd.get("ph_lng")), placeholder="27°50'12.00\"N 94°08'05.00\"E")
        
        if st.button("Apply Manual Coordinates Engine"):
            push_state_to_history()
            p_in_lat, p_in_lng = parse_combined_coords(in_coord_val)
            p_ph_lat, p_ph_lng = parse_combined_coords(ph_coord_val) if ph_coord_val else (None, None)
            
            if p_in_lat and p_in_lng:
                if not (p_ph_lat and p_ph_lng):
                    if process_intake(p_in_lat, p_in_lng): st.rerun()
                else:
                    if "catchment_km2" not in sd: process_intake(p_in_lat, p_in_lng)
                    res = process_calculations(p_in_lat, p_in_lng, st.session_state["site_data"].get("catchment_km2", 100), p_ph_lat, p_ph_lng)
                    if "error" in res: 
                        st.error(res["error"])
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
            st.session_state["site_data"].pop("intake_lat", None); st.session_state["site_data"].pop("intake_lng", None); st.session_state["site_data"].pop("catchment_km2", None)
            st.session_state["workflow_state"] = "AWAITING_INTAKE"
            st.rerun()
        if rc2.button("Reset Powerhouse Node"):
            push_state_to_history()
            st.session_state["site_data"].pop("ph_lat", None); st.session_state["site_data"].pop("ph_lng", None)
            st.session_state["workflow_state"] = "AWAITING_POWERHOUSE"
            st.rerun()

    st.markdown("---")
    if st.session_state["workflow_state"] == "COMPLETE":
        f_color = "green" if sd["flag"] == "GREEN" else ("orange" if sd["flag"] == "AMBER" else "red")
        st.markdown(f"""
        <div style="background-color: {f_color}; padding: 12px; border-radius: 4px; color: white; font-weight: bold; text-align: center; margin-bottom: 15px;">
            TRIAGE STATUS: {sd['flag']} — {sd['flag_reason']}
        </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Capacity", f"{sd['capacity_MW']:.2f} MW")
        m2.metric("Annual Gen", f"{sd['annual_MWh']:.0f} MWh")
        m3.metric("CAPEX / MW", f"₹{sd['capex']['Per_MW']:.2f} Cr")
        
        st.write(f"**MNRE Subsidy Estimate:** ₹{sd['mnre_cfa_cr']:.2f} Cr  |  **Est. DSCR (15yr):** {sd['dscr']:.2f}x")
        
        with st.expander("Show CAPEX Breakdown", expanded=False):
            capex_df = pd.DataFrame(list(sd['capex'].items()), columns=["Category", "Amount (₹ Cr)"]).set_index("Category")
            st.dataframe(capex_df.style.format("₹ {:.2f} Cr"), use_container_width=True)

        river_input = st.text_input("River / Project Name")
        notes_input = st.text_area("Field Notes / Risks")
        
        action_col1, action_col2 = st.columns(2)
        
        if action_col1.button("Save Entry to Log Database", use_container_width=True):
            log_row = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "Project_Name": river_input,
                "Intake_Coords": format_combined_dms(sd["intake_lat"], sd["intake_lng"]), 
                "PH_Coords": format_combined_dms(sd["ph_lat"], sd["ph_lng"]),
                "Catchment_km2": round(sd["catchment_km2"], 1), "Gross_Head_m": round(sd["gross_head_m"], 1),
                "Capacity_MW": round(sd["capacity_MW"], 2), "Annual_MWh": round(sd["annual_MWh"], 0),
                "CAPEX_Per_MW_Cr": round(sd['capex']["Per_MW"], 2), "MNRE_CFA_Cr": round(sd["mnre_cfa_cr"], 2),
                "Flag": sd["flag"], "Notes": notes_input, "LLM_Note_Attached": "feasibility_note" in st.session_state
            }
            file_exists = os.path.exists("site_log.csv")
            pd.DataFrame([log_row]).to_csv("site_log.csv", mode='a', header=not file_exists, index=False)
            st.success("Record appended successfully to site_log.csv.")

        if action_col2.button("Synthesize Full Report", use_container_width=True):
            with st.spinner("Compiling Note via Gemini API..."):
                model = genai.GenerativeModel(cfg_gemini_model)
                site_context_payload = {
                    "topography": {"catchment_sq_km": sd["catchment_km2"], "gross_head_m": sd["gross_head_m"], "net_head_m": sd["net_head_m"], "wdpa_intersection": sd["within_wdpa"]},
                    "hydrology": {"design_flow_cumecs": sd["q_design_cumecs"], "specific_yield": sd["specific_yield"], "projected_annual_generation_mwh": sd["annual_MWh"]},
                    "electro_mechanical": {"capacity_mw": sd["capacity_MW"], "turbine_type": sd["turbine_type"], "unit_count": sd["num_units"]},
                    "infrastructure": {"penstock_length_m": sd["penstock_m"], "nearest_substation": sd["nearest_substation"], "interconnection_distance_km": sd["substation_km"], "voltage_kv": sd["voltage_kv"]},
                    "financials": sd['capex'], "triage_flag": sd['flag']
                }
                
                prompt = f"""
                You are a senior technical lead engineer. Construct an exhaustive, professional Desktop Pre-Feasibility Note based strictly upon the provided raw engineering payload below. 
                RAW CALCULATIONS PAYLOAD:
                {json.dumps(site_context_payload, indent=2)}
                
                Explicitly label the header exactly as follows:
                # DESKTOP PRE-FEASIBILITY NOTE
                ## ⚠️ UNVERIFIED DESKTOP ANALYSIS — NOT A FORMAL SALIENT FEATURES DOCUMENT
                **Estimates based on remote sensing and parametric models. Field validation required.**
                
                Compile 8 explicit sections containing analytical prose based on the data:
                   - SECTION 1: LOCATION & BOUNDARIES
                   - SECTION 2: HYDROLOGICAL PROFILE & CONSTRAINTS
                   - SECTION 3: CIVIL WORKS CONFIGURATION
                   - SECTION 4: POWER GENERATION SCHEMATIC
                   - SECTION 5: GRID INTERCONNECTION ANALYSIS
                   - SECTION 6: ECONOMIC FEASIBILITY
                   - SECTION 7: FIELD VALIDATION CRITICAL CHECKLIST (CRITICAL: You must explicitly list these 12 validation points: Catchment verification by current meter, head verification via DGPS, sediment/bedload assessment, customary clan land rights, APEDA allotment status, cumulative basin allocation, forest classification check, geotechnical assessment, grid reliability, road access, and local community sentiment).
                   - SECTION 8: CAVEATS
                DO NOT FABRICATE COORDINATES OR DATA NOT PROVIDED.
                """
                response = model.generate_content(prompt)
                st.session_state["feasibility_note"] = response.text
                
        if "feasibility_note" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["feasibility_note"])
            st.download_button("Download MD", data=st.session_state["feasibility_note"], file_name=f"Sahaj_Urja_PFN_{river_input or 'Unassigned'}.md")

# 8. Site Log Viewer
with st.expander("View Local Site Log Database (site_log.csv)", expanded=False):
    if os.path.exists("site_log.csv"):
        log_df = pd.read_csv("site_log.csv")
        st.dataframe(log_df, use_container_width=True)
        st.download_button("Export CSV", data=log_df.to_csv(index=False), file_name="sahaj_urja_site_log.csv", mime="text/csv")
    else:
        st.info("No sites saved yet.")

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
