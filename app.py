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

# 1. Page Configuration & CSS Fixes for Cut-off Numbers
st.set_page_config(layout="wide", page_title="Sahaj Urja Site Prospector")

# Inject CSS to prevent metrics from overflowing/cutting off
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
def parse_dms(coord_str):
    coord_str = str(coord_str).strip()
    try:
        return float(coord_str)
    except ValueError:
        pass
    cleaned = coord_str.replace("''", '"').replace('°', ' ').replace("'", ' ').replace('"', ' ').strip()
    match = re.search(r'(\d+)\s+(\d+)\s+([\d\.]+)\s*([NSEWnsew]?)', cleaned)
    if match:
        d, m, s, direction = float(match.group(1)), float(match.group(2)), float(match.group(3)), match.group(4).upper()
        dd = d + (m / 60.0) + (s / 3600.0)
        return -dd if direction in ['S', 'W'] else dd
    return None

def format_dms(dd, is_lat):
    if dd is None: return ""
    direction = ('N' if dd >= 0 else 'S') if is_lat else ('E' if dd >= 0 else 'W')
    dd = abs(dd)
    d = int(dd)
    m = int((dd - d) * 60)
    s = (dd - d - m/60) * 3600
    return f"{d}°{m}'{s:.1f}\"{direction}"

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
elif "intake_lat" in sd and "ph_lat" in sd:
    map_center, map_zoom = [(sd["intake_lat"] + sd["ph_lat"]) / 2, (sd["intake_lng"] + sd["ph_lng"]) / 2], 12

# 6. User Interface Layout
col_map, col_dash = st.columns([0.55, 0.45])

with col_map:
    st.subheader("Interactive Basin Triage Map")
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles=None)
    
    # Layer Overrides (Satellite is now Default)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="High-Res Satellite Imagery", show=True
    ).add_to(m)
    folium.TileLayer("OpenTopoMap", name="Topographic Relief Map", show=False).add_to(m)
    folium.TileLayer("openstreetmap", name="Standard Road Overlay Map", show=False).add_to(m)
    
    # Substations off by default to reduce clutter
    substation_group = folium.FeatureGroup(name="18 Verified Substation Radii", show=False).add_to(m)
    for name, s_lat, s_lng in SUBSTATIONS:
        folium.Marker([s_lat, s_lng], tooltip=f"Substation: {name}", icon=folium.Icon(color="orange", icon="flash")).add_to(substation_group)
        folium.Circle([s_lat, s_lng], radius=cfg_substation_limit * 1000, color="orange", fill=False, dash_array="5, 5").add_to(substation_group)
        
    if "intake_lat" in sd:
        folium.Marker([sd["intake_lat"], sd["intake_lng"]], popup="Proposed Weir Intake", icon=folium.Icon(color="green", icon="cloud")).add_to(m)
    if "ph_lat" in sd:
        folium.Marker([sd["ph_lat"], sd["ph_lng"]], popup="Proposed Powerhouse Point", icon=folium.Icon(color="red", icon="bolt")).add_to(m)
        folium.PolyLine([[sd["intake_lat"], sd["intake_lng"]], [sd["ph_lat"], sd["ph_lng"]]], color="blue", weight=4, opacity=0.8, dash_array="6, 6").add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    map_output = st_folium(m, width="100%", height=650, key="prospector_map")
    
    # Click Processing with History Push
    if map_output and map_output.get("last_clicked"):
        click_lat, click_lng = map_output["last_clicked"]["lat"], map_output["last_clicked"]["lng"]
        current_click = (click_lat, click_lng)
        
        # Prevent re-trigger on unchanged clicks
        if st.session_state.get("last_processed_click") != current_click:
            push_state_to_history()
            st.session_state["last_processed_click"] = current_click
            
            if st.session_state["workflow_state"] in ["AWAITING_INTAKE", "CONFIRM_INTAKE"]:
                if process_intake(click_lat, click_lng):
                    st.rerun()

            elif st.session_state["workflow_state"] == "AWAITING_POWERHOUSE":
                res = process_calculations(sd["intake_lat"], sd["intake_lng"], sd["catchment_km2"], click_lat, click_lng)
                if "error" in res: st.error(res["error"])
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
        st.info("⚠️ Is the weir site accurately placed on the map? If not, click the map again to adjust the pin. Once satisfied, click Lock.")
        if st.button("✅ Lock Intake Location & Proceed", type="primary", use_container_width=True):
            push_state_to_history()
            st.session_state["workflow_state"] = "AWAITING_POWERHOUSE"
            st.rerun()
            
    elif st.session_state["workflow_state"] == "AWAITING_POWERHOUSE":
        st.success("Intake locked.")
        st.info("Action Required: Click the map to drop the Powerhouse pin, OR enter precise coordinates below.")

    with st.expander("Manual Coordinate Overrides (DMS or Decimal)", expanded=(st.session_state["workflow_state"] == "AWAITING_INTAKE")):
        c1, c2 = st.columns(2)
        in_lat_val = c1.text_input("Intake Latitude", value=format_dms(sd.get("intake_lat"), True), placeholder="28°53'09.2\"N")
        in_lng_val = c2.text_input("Intake Longitude", value=format_dms(sd.get("intake_lng"), False), placeholder="76°38'13.6\"E")
        
        c3, c4 = st.columns(2)
        ph_lat_val = c3.text_input("Powerhouse Latitude", value=format_dms(sd.get("ph_lat"), True))
        ph_lng_val = c4.text_input("Powerhouse Longitude", value=format_dms(sd.get("ph_lng"), False))
        
        if st.button("Apply Manual Coordinates Engine"):
            push_state_to_history()
            p_in_lat, p_in_lng = parse_dms(in_lat_val), parse_dms(in_lng_val)
            p_ph_lat, p_ph_lng = parse_dms(ph_lat_val), parse_dms(ph_lng_val)
            
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
                st.error("Invalid coordinate string provided.")

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
        m3.metric("CAPEX / MW", f"₹{sd['capex']['per_mw']:.2f} Cr")
        
        st.markdown("### Engineering Parameters")
        st.write(f"**Catchment Area:** {sd['catchment_km2']:.2f} sq km  |  **Design Flow (Q):** {sd['q_design_cumecs']:.2f} m³/s")
        st.write(f"**Gross Head:** {sd['gross_head_m']:.1f} m  |  **Net Design Head:** {sd['net_head_m']:.1f} m")
        st.write(f"**Penstock Profile:** {sd['penstock_m']:.0f} m (Dia: {sd['penstock_dia_mm']} mm) | Type: {sd['num_units']}x {sd['turbine_type']}")
        st.write(f"**Grid Connection:** {sd['nearest_substation']} Substation at {sd['substation_km']:.1f} km ({sd['voltage_kv']} kV)")
        
        river_input = st.text_input("River / Stream Identification Mapping Name")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        if action_col2.button("Synthesize Report", use_container_width=True):
            with st.spinner("Compiling Desktop Pre-Feasibility Note via Gemini API..."):
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
            st.download_button("Download MD", data=st.session_state["feasibility_note"], file_name=f"Sahaj_Urja_PFN_{river_input or 'Unassigned'}.md")

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
