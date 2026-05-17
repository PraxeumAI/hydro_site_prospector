import streamlit as st
import folium
from streamlit_folium import st_folium
import earthengine_api as ee
import google.generativeai as genai
import pandas as pd
import numpy as np
import json
import datetime
from math import radians, sin, cos, sqrt, atan2, pi

# 1. Page Configuration & Setup
st.set_page_config(layout="wide", page_title="Sahaj Urja Site Prospector")
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

# 3. Sidebar Configuration Panel (Editable Defaults)
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

# 4. Helper Math & Mathematical Engines
def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
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
    return {
        'civil': civil, 'em': em, 'penstock': penstock, 'transmission': transmission,
        'road': road, 'land': land, 'predev': predev, 'contingency': contingency,
        'total': total, 'per_mw': total / mw if mw > 0 else 0
    }

# 5. Workflow State Machine Initialization
if "workflow_state" not in st.session_state:
    st.session_state["workflow_state"] = "AWAITING_INTAKE"
    st.session_state["site_data"] = {}

# 6. Twin-Panel Interface Execution
col_map, col_dash = st.columns([0.6, 0.4])

with col_map:
    st.subheader("Interactive Basin Triage Map")
    
    # Initialize base Folium map focused on regional corridors
    m = folium.Map(location=[28.0, 94.5], zoom_start=7, tiles="OpenTopoMap")
    
    # Draw Substation Radii
    for name, s_lat, s_lng in SUBSTATIONS:
        folium.Marker([s_lat, s_lng], tooltip=name, icon=folium.Icon(color="orange", icon="flash")).add_to(m)
        folium.Circle([s_lat, s_lng], radius=cfg_substation_limit * 1000, color="orange", fill=False, dash_array="5, 5").add_to(m)
        
    # Render Workflow Elements visually
    if st.session_state["workflow_state"] in ["AWAITING_POWERHOUSE", "COMPLETE"]:
        folium.Marker([st.session_state["site_data"]["intake_lat"], st.session_state["site_data"]["intake_lng"]], 
                      popup="Proposed Intake", icon=folium.Icon(color="green")).add_to(m)
    if st.session_state["workflow_state"] == "COMPLETE":
        folium.Marker([st.session_state["site_data"]["ph_lat"], st.session_state["site_data"]["ph_lng"]], 
                      popup="Proposed Powerhouse", icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine([[st.session_state["site_data"]["intake_lat"], st.session_state["site_data"]["intake_lng"]],
                         [st.session_state["site_data"]["ph_lat"], st.session_state["site_data"]["ph_lng"]]], 
                        color="blue", weight=3, dash_array="5, 5").add_to(m)

    # Capture map clicks
    map_output = st_folium(m, width="100%", height=600, key="prospector_map")
    
    if map_output and map_output.get("last_clicked"):
        click_lat = map_output["last_clicked"]["lat"]
        click_lng = map_output["last_clicked"]["lng"]
        
        if st.session_state["workflow_state"] == "AWAITING_INTAKE":
            st.warning(f"Processing Intake coordinates... Lat: {click_lat:.4f}, Lng: {click_lng:.4f}")
            
            # GEE Catchment & Stream Snapping Engine
            pt = ee.Geometry.Point([click_lng, click_lat])
            merit_hydro = ee.Image('MERIT/Hydro/v1_0_1')
            
            try:
                upa_val = merit_hydro.select('upa').sample(pt, scale=90).first().get('upa').getInfo()
                
                # Snapping Override Loop if clicked outside explicit core streams
                if upa_val is None or upa_val < 10.0:
                    buffer = pt.buffer(250)
                    clipped = merit_hydro.select('upa').clip(buffer)
                    max_dict = clipped.reduceRegion(ee.Reducer.max(), buffer, 90).getInfo()
                    max_upa = max_dict.get('upa')
                    
                    if max_upa and max_upa >= 10.0:
                        upa_val = max_upa
                        st.info("Point snapped to adjacent stream channel within 250m buffer boundary.")
                
                if upa_val and upa_val >= 10.0:
                    st.session_state["site_data"]["intake_lat"] = click_lat
                    st.session_state["site_data"]["intake_lng"] = click_lng
                    st.session_state["site_data"]["catchment_km2"] = upa_val
                    st.session_state["workflow_state"] = "AWAITING_POWERHOUSE"
                    st.rerun()
                else:
                    st.error("Catchment area falls below 10 sq km baseline. Re-click nearer a verified stream reach.")
            except Exception as ex:
                st.error(f"Spatial Server connection timeout or processing failure: {str(ex)}")

        elif st.session_state["workflow_state"] == "AWAITING_POWERHOUSE":
            st.warning("Processing Powerhouse coordinates and generating hydro-mechanic profile...")
            
            intake_lat = st.session_state["site_data"]["intake_lat"]
            intake_lng = st.session_state["site_data"]["intake_lng"]
            catchment_km2 = st.session_state["site_data"]["catchment_km2"]
            
            # Query Elevations from Copernicus 30m Grid
            try:
                dem = ee.Image('MERIT/DEM/v1_0_3').select('dem')
                elev_in = dem.sample(ee.Geometry.Point([intake_lng, intake_lat]), scale=90).first().get('dem').getInfo()
                elev_p = dem.sample(ee.Geometry.Point([click_lng, click_lat]), scale=90).first().get('dem').getInfo()
                
                gross_h = elev_in - elev_p
                if gross_h <= 0:
                    st.error(f"Hydraulic violation: Powerhouse elevation ({elev_p}m) equal to or higher than Intake ({elev_in}m). Aborting.")
                else:
                    # Execute Complete Calculation Pipeline
                    net_h = gross_h * (1.0 - cfg_head_loss)
                    specific_yield = cfg_yield_west if intake_lng < 94.0 else cfg_yield_east
                    q_design = catchment_km2 * specific_yield
                    p_mw = q_design * net_h * 9.81 * cfg_efficiency / 1000.0
                    
                    # Core Lean Season Rainfall Attenuation Check via CHIRPS Time-Series
                    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')\
                        .filterDate('2000-01-01', '2024-12-31')\
                        .filter(ee.Filter.calendarRange(1, 3, 'month'))
                    mean_lean_daily = chirps.mean().reduceRegion(
                        ee.Reducer.mean(), ee.Geometry.Point([intake_lng, intake_lat]).buffer(5000), 5500
                    ).get('precipitation').getInfo()
                    
                    effective_plf = cfg_default_plf
                    if mean_lean_daily and (mean_lean_daily * 30.0) < 40.0:
                        effective_plf = cfg_default_plf * 0.5  # Apply 50% lean season penalty
                    
                    annual_mwh = p_mw * 8760.0 * effective_plf
                    straight_line_km = haversine_km(intake_lat, intake_lng, click_lat, click_lng)
                    penstock_m = straight_line_km * 1000.0 * cfg_terrain_mult
                    
                    penstock_dia_mm = round(sqrt(4.0 * q_design / (pi * 4.0)) * 1000.0 / 50.0) * 50
                    
                    turbine_type = "Pelton (Vertical)" if net_h > 200.0 else ("Francis (Horizontal)" if net_h > 30.0 else "Kaplan / Crossflow")
                    num_units = 2 if p_mw >= 4.0 else 1
                    
                    distances = [(name, haversine_km(click_lat, click_lng, s_lat, s_lng)) for name, s_lat, s_lng in SUBSTATIONS]
                    near_sub, near_dist = min(distances, key=lambda x: x[1])
                    voltage = 33 if p_mw <= 10.0 else 132
                    
                    wdpa = ee.FeatureCollection('WCMC/WDPA/current/polygons')
                    within_wdpa = wdpa.filterBounds(ee.Geometry.Point([intake_lng, intake_lat])).size().getInfo() > 0
                    
                    # Deterministic Flag Allocator
                    if within_wdpa:
                        flag, reason = "RED", "WDPA Protected Area Exclusion Area"
                    elif p_mw < cfg_min_mw:
                        flag, reason = "RED", f"Below Policy Target Threshold ({p_mw:.2f} MW < {cfg_min_mw} MW)"
                    elif p_mw > cfg_max_mw:
                        flag, reason = "AMBER", f"Exceeds Target Ceiling Profile ({p_mw:.2f} MW > {cfg_max_mw} MW)"
                    elif near_dist > cfg_substation_limit:
                        flag, reason = "AMBER", f"Grid Isolation Warning ({near_dist:.1f} km > {cfg_substation_limit} km)"
                    else:
                        flag, reason = "GREEN", "Optimal Hydrological & Civil Target Envelope"
                        
                    capex_data = compute_capex(p_mw, penstock_m, near_dist, cfg_road_km, cfg_is_apst)
                    if capex_data['per_mw'] > 13.0 and flag == "GREEN":
                        flag, reason = "AMBER", f"CAPEX Outlier Projections (₹{capex_data['per_mw']:.2f} Cr/MW)"

                    # Update session data container
                    st.session_state["site_data"].update({
                        "ph_lat": click_lat, "ph_lng": click_lng, "elev_intake": elev_in, "elev_ph": elev_p,
                        "gross_head_m": gross_h, "net_head_m": net_h, "q_design_cumecs": q_design, "capacity_MW": p_mw,
                        "annual_MWh": annual_mwh, "plf": effective_plf * 100, "penstock_m": penstock_m,
                        "penstock_dia_mm": penstock_dia_mm, "turbine_type": turbine_type, "num_units": num_units,
                        "nearest_substation": near_sub, "substation_km": near_dist, "voltage_kv": voltage,
                        "within_wdpa": within_wdpa, "flag": flag, "flag_reason": reason, "capex": capex_data,
                        "specific_yield": specific_yield
                    })
                    st.session_state["workflow_state"] = "COMPLETE"
                    st.rerun()
            except Exception as ex:
                st.error(f"Error evaluating hydraulic delta: {str(ex)}")

with col_dash:
    st.subheader("Hydraulic & Economic Evaluation Dashboard")
    
    if st.session_state["workflow_state"] == "AWAITING_INTAKE":
        st.info("Action Required: Identify potential upstream weir location and click directly on the river channel matrix.")
    elif st.session_state["workflow_state"] == "AWAITING_POWERHOUSE":
        st.success(f"Intake locked successfully. Catchment Area traced: {st.session_state['site_data']['catchment_km2']:.1f} sq km.")
        st.info("Action Required: Click downstream on the channel profile to specify intended Powerhouse center line.")
        if st.button("Reset Workflow"):
            st.session_state["workflow_state"] = "AWAITING_INTAKE"
            st.session_state["site_data"] = {}
            st.rerun()
            
    elif st.session_state["workflow_state"] == "COMPLETE":
        sd = st.session_state["site_data"]
        
        # Color Code Flag Cards Dynamically
        f_color = "green" if sd["flag"] == "GREEN" else ("orange" if sd["flag"] == "AMBER" else "red")
        st.markdown(f"""
        <div style="background-color: {f_color}; padding: 15px; border-radius: 5px; color: white; font-weight: bold; text-align: center; margin-bottom: 20px;">
            TRIAGE STATUS: {sd['flag']} — {sd['flag_reason']}
        </div>
        """, unsafe_allow_html=True)
        
        # Display Core Performance Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Calculated Capacity", f"{sd['capacity_MW']:.2f} MW")
        m2.metric("Annual Generation Output", f"{sd['annual_MWh']:.0f} MWh")
        m3.metric("Projected CAPEX / MW", f"₹{sd['capex']['per_mw']:.2f} Cr")
        
        # Technical Specification Grid
        st.markdown("### Technical Metrics Profile")
        st.write(f"**Catchment Boundary Surface:** {sd['catchment_km2']:.2f} sq km")
        st.write(f"**Design Inflow (Q):** {sd['q_design_cumecs']:.2f} m³/s (Yield Factor: {sd['specific_yield']:.3f})")
        st.write(f"**Gross Hydraulic Head:** {sd['gross_head_m']:.1f} m  |  **Net Design Head:** {sd['net_head_m']:.1f} m")
        st.write(f"**Estimated Penstock Route Length:** {sd['penstock_m']:.0f} m (Diameter: {sd['penstock_dia_mm']} mm)")
        st.write(f"**Prime Mover Array:** {sd['num_units']} x {sd['turbine_type']}")
        st.write(f"**Grid Evacuation Base:** {sd['nearest_substation']} Substation at {sd['substation_km']:.1f} km ({sd['voltage_kv']} kV Link)")
        
        # CAPEX Structural Array
        st.markdown("### Parametric Cost Breakdown (₹ Crore)")
        cx = sd['capex']
        capex_df = pd.DataFrame({
            "Civil Structures": [cx['civil']], "Electro-Mechanical": [cx['em']], "Penstock Steel Base": [cx['penstock']],
            "Transmission Interface": [cx['transmission']], "Road Construction": [cx['road']], "Land Acquisition Module": [cx['land']],
            "Pre-Development & Clearances": [cx['predev']], "Unforeseen Contingency": [cx['contingency']], "Total Project Value": [cx['total']]
        }).T.rename(columns={0: "Projected Cost"})
        st.table(capex_df.style.format("₹ {:.2f} Cr"))
        
        # Database Logger Module & Action Arrays
        river_input = st.text_input("Assign River / Stream Vector Descriptor Identification Name")
        notes_input = st.text_area("Field Scouting Observations / Risk Notes")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        if action_col1.button("Commit to Site Registry"):
            log_row = {
                "timestamp": datetime.datetime.now().isoformat(), "river_name": river_input,
                "intake_lat": sd["intake_lat"], "intake_lng": sd["intake_lng"], "ph_lat": sd["ph_lat"], "ph_lng": sd["ph_lng"],
                "catchment_km2": sd["catchment_km2"], "gross_head_m": sd["gross_head_m"], "net_head_m": sd["net_head_m"],
                "penstock_m": sd["penstock_m"], "q_design_cumecs": sd["q_design_cumecs"], "capacity_MW": sd["capacity_MW"],
                "annual_MWh": sd["annual_MWh"], "plf": sd["plf"], "nearest_substation": sd["nearest_substation"],
                "substation_km": sd["substation_km"], "voltage_kv": sd["voltage_kv"], "capex_total_cr": cx["total"],
                "capex_per_mw": cx["per_mw"], "flag": sd["flag"], "flag_reason": sd["flag_reason"], "notes": notes_input
            }
            try:
                df_existing = pd.read_csv("site_log.csv")
                df_new = pd.concat([df_existing, pd.DataFrame([log_row])], ignore_index=True)
            except FileNotFoundError:
                df_new = pd.DataFrame([log_row])
            df_new.to_csv("site_log.csv", index=False)
            st.success("Data successfully locked to persistent site_log.csv container.")

        if action_col2.button("Run Pre-Feasibility Synthesis"):
            with st.spinner("Invoking Gemini Processing Core to compile structural Pre-Feasibility Assessment..."):
                model = genai.GenerativeModel(cfg_gemini_model)
                site_context_payload = {
                    "topography": {"catchment_sq_km": sd["catchment_km2"], "gross_head_m": sd["gross_head_m"], "net_head_m": sd["net_head_m"]},
                    "hydrology": {"design_flow_cumecs": sd["q_design_cumecs"], "specific_yield": sd["specific_yield"], "projected_annual_generation_mwh": sd["annual_MWh"], "plf_percentage": sd["plf"]},
                    "electro_mechanical": {"capacity_mw": sd["capacity_MW"], "turbine_type": sd["turbine_type"], "unit_count": sd["num_units"]},
                    "infrastructure": {"penstock_length_m": sd["penstock_m"], "penstock_diameter_mm": sd["penstock_dia_mm"], "nearest_substation": sd["nearest_substation"], "interconnection_distance_km": sd["substation_km"], "evacuation_voltage_kv": sd["voltage_kv"]},
                    "financials": {"total_capex_cr": cx["total"], "capex_per_mw_cr": cx["per_mw"], "civil_works_cr": cx["civil"], "e_m_package_cr": cx["em"], "penstock_cost_cr": cx["penstock"], "land_acquisition_cost_cr": cx["land"], "mnre_subsidy_eligibility_cr": min(30.0, 3.6 * sd["capacity_MW"])}
                }
                
                prompt = f"""
                You are a senior technical lead engineer specializing in Himalayan run-of-river small hydropower developments. Construct an exhaustive, highly professional Desktop Pre-Feasibility Note based strictly upon the provided raw engineering calculations payload below. 
                
                RAW CALCULATIONS PAYLOAD:
                {json.dumps(site_context_payload, indent=2)}
                
                Strict Structural Execution Rules:
                1. Render the text inline as a direct professional brief. 
                2. Explicitly label the header note exactly as follows:
                # DESKTOP PRE-FEASIBILITY NOTE
                ## ⚠️ UNVERIFIED DESKTOP ANALYSIS — NOT A FORMAL SALIENT FEATURES DOCUMENT
                **Estimates based on remote sensing and parametric models. Field validation required before any commitment, application, or capital expenditure.**
                
                3. Compile 8 explicit sections containing deep analytical prose based on the data:
                   - SECTION 1: LOCATION & GEOSPATIAL BOUNDARIES
                   - SECTION 2: HYDROLOGICAL PROFILE & INFRASTRUCTURE CONSTRAINTS
                   - SECTION 3: CIVIL WORKS CONFIGURATION PARAMETERS (Address Trench Weir scaling, Desilting Basin retention requirements, Forebay capacities, and Steel Penstock configuration)
                   - SECTION 4: POWER GENERATION SCHEMATIC & EQUIPMENT DESIGN SELECTION
                   - SECTION 5: GRID INTERCONNECTION & POWER EVACUATION ANALYSIS
                   - SECTION 6: ECONOMIC FEASIBILITY & DEVELOPMENT ROADMAP (Detail explicit CERC capital frameworks and MNRE CFA calculations)
                   - SECTION 7: FIELD VALIDATION CRITICAL CHECKLIST (Construct a highly prominent checklist covering DGPS verification, silt/bedload realities, customary land tenure complexities, and regulatory alignment)
                   - SECTION 8: CAVEATS & TECHNICAL LIMITATIONS
                
                Deliver data accurately without excessive rounding, maintaining a direct, objective engineering perspective.
                """
                response = model.generate_content(prompt)
                st.session_state["feasibility_note"] = response.text
                
        if "feasibility_note" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["feasibility_note"])
            st.download_button("Download Report File", data=st.session_state["feasibility_note"], file_name=f"Sahaj_Urja_PFN_{river_input or 'Unassigned'}.md", mime="text/markdown")

        if action_col3.button("Reset Tracking Loop"):
            st.session_state["workflow_state"] = "AWAITING_INTAKE"
            st.session_state["site_data"] = {}
            if "feasibility_note" in st.session_state:
                del st.session_state["feasibility_note"]
            st.rerun()

# 7. Persistent Log Management Interface Layer
st.markdown("---")
with st.expander("Review Local Site Assessment Log Database"):
    try:
        df_log = pd.read_csv("site_log.csv")
        st.dataframe(df_log)
        if st.button("Purge Persistent Log History"):
            pd.DataFrame(columns=df_log.columns).to_csv("site_log.csv", index=False)
            st.success("Log records truncated completely.")
            st.rerun()
    except FileNotFoundError:
        st.info("No registered records located in the active environment.")