"""
Supply Chain Optimization — Self-Service Application
=====================================================
Upload your data, run the full pipeline, and explore interactive results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import STORE_COORDINATES, STORE_NAMES, DEPT_NAMES
from src.pipeline_runner import (
    run_full_pipeline, get_template_dataframe, validate_upload, DEFAULT_CONFIG,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Optimization System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    [data-testid="stMetric"] {
        background: #1e1e2d; border: 1px solid rgba(255,255,255,0.05);
        padding: 1.2rem; border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important; font-size: 0.9rem !important; font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 1.8rem !important; font-weight: 600 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 16px; background: transparent; padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px; padding: 8px 16px; font-weight: 500; color: #a0aec0;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important; border-bottom-color: #667eea !important;
    }

    [data-testid="stSidebar"] {
        background: #151521; border-right: 1px solid rgba(255,255,255,0.05);
    }
    .sidebar-title {
        font-size: 1.25rem; font-weight: 600; color: #ffffff;
        margin-bottom: 1.5rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #ffffff; margin: 2rem 0 1rem 0;
    }
    .main-header { padding: 1rem 0 1.5rem 0; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .main-header h1 { font-size: 1.8rem; font-weight: 600; color: #ffffff; margin: 0; }
    .main-header p { color: #a0aec0; font-size: 1rem; margin: 0.5rem 0 0 0; }
    .block-container { padding-top: 2rem !important; }

    .upload-box {
        background: #1e1e2d; border: 2px dashed rgba(102,126,234,0.4);
        border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .step-badge {
        display: inline-block; background: #667eea; color: white;
        border-radius: 50%; width: 28px; height: 28px; line-height: 28px;
        text-align: center; font-weight: 600; font-size: 0.85rem; margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper: label builders ──────────────────────────────────────────────────
def store_label(sid):
    name = STORE_NAMES.get(int(sid), "")
    return f"Store {int(sid)} — {name}" if name else f"Store {int(sid)}"

def dept_label(did):
    name = DEPT_NAMES.get(int(did), "")
    return f"Dept {int(did)} — {name}" if name else f"Dept {int(did)}"


# ─── Load demo data ──────────────────────────────────────────────────────────
@st.cache_data
def load_demo_results():
    """Try to load pre-computed outputs for instant demo mode."""
    try:
        preds = pd.read_csv("outputs/predictions.csv")
        inv = pd.read_csv("outputs/inventory_plan.csv")
        routes = pd.read_csv("outputs/optimized_routes.csv")
        with open("outputs/model_metrics.json") as f:
            metrics = json.load(f)
        with open("outputs/simulation_results.json") as f:
            sim = json.load(f)
        hist = pd.read_csv("data/walmart/walmart_merged.csv", parse_dates=["Date"])
        with open("config.json") as f:
            cfg = json.load(f)
        return {
            "predictions": preds, "inventory_plan": inv, "routes": routes,
            "metrics": metrics, "simulation": sim, "historical_data": hist,
            "inventory_baseline_cost": 49350302.77,
            "routes_baseline_dist": 62021.9, "config": cfg,
        }
    except Exception:
        return None


# ─── Session State Init ──────────────────────────────────────────────────────
if "results" not in st.session_state:
    demo = load_demo_results()
    if demo:
        st.session_state["results"] = demo
        st.session_state["data_source"] = "demo"
    else:
        st.session_state["results"] = None
        st.session_state["data_source"] = None


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Supply Chain Optimization System</h1>
    <p>Demand Forecasting  •  Inventory Optimization  •  Route Planning  •  Risk Simulation</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
results = st.session_state.get("results")

with st.sidebar:
    st.markdown('<div class="sidebar-title">Control Panel</div>', unsafe_allow_html=True)

    # Data source indicator
    src = st.session_state.get("data_source")
    if src == "demo":
        st.caption("Using built-in demo data")
    elif src == "upload":
        st.caption("Using your uploaded data")
    else:
        st.caption("No data loaded yet")

    st.divider()

    # Store / Department selection
    if results and not results["predictions"].empty:
        preds = results["predictions"]

        store_ids = sorted(preds["Store"].unique().tolist())
        store_options = {store_label(s): s for s in store_ids}
        selected_store_label = st.selectbox("Select Store", list(store_options.keys()), index=0)
        selected_store = store_options[selected_store_label]

        dept_ids = sorted(preds[preds["Store"] == selected_store]["Dept"].unique().tolist())
        dept_options = {dept_label(d): d for d in dept_ids}
        selected_dept_label = st.selectbox("Select Department", list(dept_options.keys()), index=0)
        selected_dept = dept_options[selected_dept_label]
    else:
        selected_store = 1
        selected_dept = 1
        st.info("Upload data to enable store selection")

    st.divider()

    st.markdown("**Simulation Scenarios**")
    sim_scenario = st.radio(
        "scenario_radio",
        ["Baseline (No Simulation)", "Demand Spike", "Supply Delay", "Warehouse Failure"],
        index=0, label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Data Export**")
    if results:
        if not results["predictions"].empty:
            st.download_button("Download Predictions",
                               results["predictions"].to_csv(index=False).encode(),
                               "predictions.csv", "text/csv", use_container_width=True)
        if isinstance(results.get("inventory_plan"), pd.DataFrame) and not results["inventory_plan"].empty:
            st.download_button("Download Inventory Plan",
                               results["inventory_plan"].to_csv(index=False).encode(),
                               "inventory_plan.csv", "text/csv", use_container_width=True)
        if isinstance(results.get("routes"), pd.DataFrame) and not results["routes"].empty:
            st.download_button("Download Routes",
                               results["routes"].to_csv(index=False).encode(),
                               "routes.csv", "text/csv", use_container_width=True)


# ─── Convenience references ──────────────────────────────────────────────────
predictions = results["predictions"] if results else pd.DataFrame()
inventory_plan = results["inventory_plan"] if results else pd.DataFrame()
routes = results["routes"] if results else pd.DataFrame()
metrics = results["metrics"] if results else {}
sim_results = results["simulation"] if results else {}
historical = results.get("historical_data", pd.DataFrame()) if results else pd.DataFrame()
config = results.get("config", DEFAULT_CONFIG) if results else DEFAULT_CONFIG
baseline_cost = results.get("inventory_baseline_cost", 0) if results else 0
baseline_dist = results.get("routes_baseline_dist", 0) if results else 0

if isinstance(inventory_plan, dict):
    inventory_plan = pd.DataFrame()
if isinstance(routes, dict):
    routes = pd.DataFrame()


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload Data", "Overview", "Demand Forecast", "Inventory", "Routes", "Simulation",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0: Upload Data
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.markdown('<div class="section-header">Get Started</div>', unsafe_allow_html=True)

    st.markdown("""
    Upload your own supply chain sales data or use the built-in demo dataset to explore 
    the system's capabilities. The pipeline will automatically run forecasting, inventory 
    optimization, route planning, and disruption simulation on your data.
    """)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("**Upload Your Data**")
        st.markdown("""
        Your CSV file must contain the following columns:
        
        | Column | Type | Example |
        |:-------|:-----|:--------|
        | `Store` | int | 1 |
        | `Dept` | int | 1 |
        | `Date` | date | 2024-01-05 |
        | `Weekly_Sales` | float | 24924.50 |
        | `IsHoliday` | bool | FALSE |
        | `Temperature` | float | 42.31 |
        | `Fuel_Price` | float | 2.572 |
        | `CPI` | float | 211.09 |
        | `Unemployment` | float | 8.106 |
        | `Type` | str | A |
        | `Size` | int | 151315 |
        """)

        # Download template
        template_csv = get_template_dataframe().to_csv(index=False).encode()
        st.download_button("Download CSV Template", template_csv, "supply_chain_template.csv",
                           "text/csv", use_container_width=True)

        uploaded = st.file_uploader("Upload your CSV file", type=["csv"], label_visibility="collapsed")

        if uploaded:
            try:
                user_df = pd.read_csv(uploaded)
                ok, err = validate_upload(user_df)
                if not ok:
                    st.error(err)
                else:
                    st.success(f"File loaded — {len(user_df):,} rows, {user_df['Store'].nunique()} stores, {user_df['Dept'].nunique()} departments")

                    if st.button("Run Pipeline", type="primary", use_container_width=True):
                        status = st.status("Running supply chain pipeline...", expanded=True)
                        progress = st.progress(0)

                        def _cb(name, step, total):
                            progress.progress(step / total, text=name)
                            status.update(label=name)

                        with status:
                            res = run_full_pipeline(user_df, progress_callback=_cb)

                        st.session_state["results"] = res
                        st.session_state["data_source"] = "upload"
                        progress.progress(1.0, text="Pipeline complete")
                        status.update(label="Pipeline complete", state="complete")
                        st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with col_r:
        st.markdown("**Or Use Demo Data**")
        st.markdown("""
        Instantly explore the dashboard using pre-computed results from the 
        Walmart Retail Sales dataset (421K rows, 45 stores, 81 departments).
        """)

        if st.button("Load Demo Data", use_container_width=True):
            demo = load_demo_results()
            if demo:
                st.session_state["results"] = demo
                st.session_state["data_source"] = "demo"
                st.rerun()
            else:
                st.warning("Demo data files not found. Please upload your own data.")

        if results:
            st.markdown("---")
            st.markdown("**Current Data Summary**")
            n_stores = predictions["Store"].nunique() if not predictions.empty else 0
            n_depts = predictions["Dept"].nunique() if not predictions.empty else 0
            n_rows = len(historical) if not historical.empty else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Stores", n_stores)
            c2.metric("Departments", n_depts)
            c3.metric("Historical Rows", f"{n_rows:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not results:
        st.info("Upload data or load the demo dataset to view results.")
    else:
        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast MAE", f"${metrics.get('val_mae', 0):,.0f}" if metrics else "N/A")
        c2.metric("Forecast R²", f"{metrics.get('val_r2', 0):.3f}" if metrics else "N/A")
        inv_cost = inventory_plan["Total_Cost"].sum() if not inventory_plan.empty else 0
        c3.metric("Total Inventory Cost", f"${inv_cost:,.0f}" if inv_cost else "N/A")
        rt_dist = routes.groupby("Vehicle")["Cumulative_Distance_km"].max().sum() if not routes.empty else 0
        c4.metric("Total Route Distance", f"{rt_dist:,.0f} km" if rt_dist else "N/A")

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown('<div class="section-header">Cost Savings Analysis</div>', unsafe_allow_html=True)
            if not inventory_plan.empty:
                optimized_cost = inventory_plan["Total_Cost"].sum()
                savings = max(0, baseline_cost - optimized_cost)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=["Naive (Weekly Order)"], y=[baseline_cost],
                                     marker_color="#4a5568", text=[f"${baseline_cost:,.0f}"], textposition="auto"))
                fig.add_trace(go.Bar(x=["Optimized (LP)"], y=[optimized_cost],
                                     marker_color="#667eea", text=[f"${optimized_cost:,.0f}"], textposition="auto"))
                fig.update_layout(yaxis_title="Total Cost ($)", template="plotly_dark", height=350,
                                  showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Estimated Savings", f"${savings:,.0f}")

        with col_right:
            st.markdown('<div class="section-header">Delivery Efficiency</div>', unsafe_allow_html=True)
            if not routes.empty:
                opt_dist = routes.groupby("Vehicle")["Cumulative_Distance_km"].max().sum()
                dsav = max(0, baseline_dist - opt_dist)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=["Individual Trips"], y=[baseline_dist],
                                     marker_color="#4a5568", text=[f"{baseline_dist:,.0f} km"], textposition="auto"))
                fig.add_trace(go.Bar(x=["Optimized (VRP)"], y=[opt_dist],
                                     marker_color="#667eea", text=[f"{opt_dist:,.0f} km"], textposition="auto"))
                fig.update_layout(yaxis_title="Total Distance (km)", template="plotly_dark", height=350,
                                  showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Distance Saved", f"{dsav:,.0f} km")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Demand Forecast
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not results:
        st.info("Upload data or load the demo dataset to view results.")
    else:
        st.markdown(f'<div class="section-header">Forecast: {store_label(selected_store)}, {dept_label(selected_dept)}</div>',
                    unsafe_allow_html=True)
        col_l, col_r = st.columns([2, 1], gap="large")

        with col_l:
            st.markdown("**Historical vs Predicted Sales**")
            if not historical.empty and not predictions.empty:
                hist = historical[(historical["Store"] == selected_store) & (historical["Dept"] == selected_dept)].sort_values("Date")
                pred = predictions[(predictions["Store"] == selected_store) & (predictions["Dept"] == selected_dept)].copy()
                if not pred.empty:
                    pred["Date"] = pd.to_datetime(pred["Date"])
                    pred = pred.sort_values("Date")
                fig = go.Figure()
                if not hist.empty:
                    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Weekly_Sales"], mode="lines",
                                             name="Historical", line=dict(color="#a0aec0", width=1.5)))
                if not pred.empty:
                    fig.add_trace(go.Scatter(x=pred["Date"], y=pred["Predicted_Weekly_Sales"],
                                             mode="lines+markers", name="Predicted",
                                             line=dict(color="#667eea", width=2, dash="dot"), marker=dict(size=6)))
                fig.update_layout(xaxis_title="Date", yaxis_title="Weekly Sales ($)", template="plotly_dark",
                                  height=400, margin=dict(l=0, r=0, t=20, b=0),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**Feature Importance**")
            if metrics and "feature_importance" in metrics:
                fi = metrics["feature_importance"]
                fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
                fi_df = fi_df.sort_values("Importance", ascending=True).tail(12)
                fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                             template="plotly_dark", color_discrete_sequence=["#667eea"])
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                fig.update_yaxes(title=""); fig.update_xaxes(title="")
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Inventory
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not results:
        st.info("Upload data or load the demo dataset to view results.")
    elif inventory_plan.empty:
        st.info("No inventory plan available.")
    else:
        st.markdown(f'<div class="section-header">Inventory Plan: {store_label(selected_store)}</div>',
                    unsafe_allow_html=True)
        si = inventory_plan[inventory_plan["Store"] == selected_store]
        if si.empty:
            st.warning(f"No inventory data for {store_label(selected_store)}.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Orders Placed", f'{(si["Optimal_Order"] > 0).sum()}')
            c2.metric("Avg Inventory", f'{si["Inventory_Level"].mean():,.0f}')
            c3.metric("Total Store Cost", f'${si["Total_Cost"].sum():,.0f}')
            order_rows = si[si["Optimal_Order"] > 0]
            c4.metric("Avg Order Size", f'{order_rows["Optimal_Order"].mean():,.0f}' if not order_rows.empty else "0")

            st.markdown("<br>", unsafe_allow_html=True)
            cl, cr = st.columns(2, gap="large")
            with cl:
                st.markdown("**Demand vs Inventory Level**")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=si["Week"], y=si["Demand"], name="Demand",
                                     marker_color="#4a5568", opacity=0.8), secondary_y=False)
                fig.add_trace(go.Scatter(x=si["Week"], y=si["Inventory_Level"], name="Inventory Level",
                                         mode="lines+markers", line=dict(color="#667eea", width=2),
                                         marker=dict(size=6)), secondary_y=True)
                fig.update_layout(template="plotly_dark", height=380, margin=dict(l=0, r=0, t=20, b=0),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
            with cr:
                st.markdown("**Cost Breakdown**")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=si["Week"], y=si["Holding_Cost"], name="Holding Cost", marker_color="#a0aec0"))
                fig.add_trace(go.Bar(x=si["Week"], y=si["Ordering_Cost"], name="Ordering Cost", marker_color="#667eea"))
                fig.update_layout(template="plotly_dark", height=380, barmode="stack",
                                  margin=dict(l=0, r=0, t=20, b=0),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Detailed Schedule**")
            st.dataframe(si.style.format({
                "Demand": "{:,.0f}", "Optimal_Order": "{:,.0f}", "Inventory_Level": "{:,.0f}",
                "Holding_Cost": "${:,.2f}", "Ordering_Cost": "${:,.2f}", "Total_Cost": "${:,.2f}",
            }), use_container_width=True, height=250)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Routes
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not results:
        st.info("Upload data or load the demo dataset to view results.")
    elif routes.empty:
        st.info("No route data available.")
    else:
        st.markdown('<div class="section-header">Optimized Delivery Routes</div>', unsafe_allow_html=True)
        vehicles = routes["Vehicle"].unique()
        vs = routes.groupby("Vehicle").agg(
            Stops=("Stop_Order", "max"), Total_Distance=("Cumulative_Distance_km", "max"),
            Total_Load=("Cumulative_Load", "max")).reset_index()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Vehicles Used", len(vehicles))
        c2.metric("Total Fleet Distance", f"{vs['Total_Distance'].sum():,.0f} km")
        c3.metric("Total Completed Stops", f"{vs['Stops'].sum():,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**System Geographic Map**")
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="CartoDB dark_matter")
        colors = ["#667eea", "#764ba2", "#5aa897", "#f6a5c0", "#f9d56e", "#6b5b95", "#d64161", "#ff7b25"]

        routing_cfg = config.get("routing", DEFAULT_CONFIG["routing"])
        for depot in routing_cfg.get("depots", []):
            folium.Marker([depot["lat"], depot["lon"]],
                          popup=f"<b>{depot['name']}</b><br>Depot",
                          icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

        for i, v in enumerate(vehicles):
            vd = routes[routes["Vehicle"] == v].sort_values("Stop_Order")
            c = colors[i % len(colors)]
            coords = list(zip(vd["Lat"], vd["Lon"]))
            if len(coords) > 1:
                folium.PolyLine(coords, color=c, weight=3, opacity=0.8).add_to(m)
            for _, row in vd.iterrows():
                if not row.get("Is_Depot", False):
                    folium.CircleMarker([row["Lat"], row["Lon"]], radius=5, color=c, fill=True,
                                        fillColor=c, fillOpacity=0.9,
                                        popup=f"Location: {row['Location']}<br>Vehicle: {v}").add_to(m)

        with st.container():
            st_folium(m, width=None, height=450, use_container_width=True, returned_objects=[])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Simulation
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    if not results:
        st.info("Upload data or load the demo dataset to view results.")
    elif not sim_results:
        st.info("No simulation results available.")
    else:
        st.markdown('<div class="section-header">Disruption Event Simulation</div>', unsafe_allow_html=True)
        ds = sim_results.get("demand_spike", {})
        sd = sim_results.get("supply_delay", {})
        wf = sim_results.get("warehouse_failure", {})

        if sim_scenario == "Demand Spike":
            st.markdown("### Scenario: Demand Spike (+20%)")
            st.markdown("Measuring system resilience against sudden surges in product popularity.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Demand Increase", f'{ds.get("demand_increase_pct", 0):.0f}%')
            c2.metric("Cost Increase", f'${ds.get("cost_increase", 0):,.0f}')
            c3.metric("Extra Vehicles Required", ds.get("additional_vehicles_needed", 0))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Baseline Cost", "Post-Spike Cost"],
                                 y=[ds.get("baseline_inventory_cost", 0), ds.get("new_inventory_cost", 0)],
                                 marker_color=["#4a5568", "#667eea"],
                                 text=[f'${ds.get("baseline_inventory_cost", 0):,.0f}',
                                       f'${ds.get("new_inventory_cost", 0):,.0f}'], textposition="auto"))
            fig.update_layout(yaxis_title="Total Cost ($)", template="plotly_dark", height=400,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        elif sim_scenario == "Supply Delay":
            st.markdown("### Scenario: Global Supply Delay")
            st.markdown("Measuring impact of inbound shipping delays enforcing larger safety stocks.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Lead Time Delay", f'{sd.get("delay_days", 0)} days')
            c2.metric("Required Safety Stock", f'{sd.get("extra_safety_stock_total", 0):,.0f} units')
            c3.metric("Cost Increase", f'${sd.get("cost_increase", 0):,.0f}')

        elif sim_scenario == "Warehouse Failure":
            st.markdown("### Scenario: Structural Warehouse Failure")
            st.markdown("Measuring routing fallback when a primary distribution center goes offline.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Failed Infrastructure", wf.get("failed_depot", "N/A"))
            c2.metric("Stores Reassigned", wf.get("stores_reassigned", 0))
            c3.metric("Distance Increase", f'{wf.get("distance_increase_km", 0):,.0f} km')

        else:
            st.info("Select a scenario from the sidebar to view simulation impacts.")
