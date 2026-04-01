"""
Supply Chain Optimization Dashboard
=====================================
Interactive Streamlit application for visualizing and interacting
with the supply chain optimization system.
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

from src.data_processing import STORE_COORDINATES

# -- Page Config --
st.set_page_config(
    page_title="Supply Chain Optimization System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS for premium look --
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main > div {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1e1e2d;
        border: 1px solid rgba(255,255,255,0.05);
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: transparent;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        color: #a0aec0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #667eea !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #151521;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
    }

    /* Header */
    .main-header {
        padding: 1rem 0 2rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }

    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Container spacing */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# -- Load Data --
@st.cache_data
def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("config.json not found! Please run the pipeline first.")
        return {}

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv("outputs/predictions.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_validation():
    try:
        return pd.read_csv("outputs/validation_predictions.csv", parse_dates=["Date"])
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_inventory_plan():
    try:
        return pd.read_csv("outputs/inventory_plan.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_routes():
    try:
        return pd.read_csv("outputs/optimized_routes.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_metrics():
    try:
        with open("outputs/model_metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_simulation_results():
    try:
        with open("outputs/simulation_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_walmart_merged():
    try:
        return pd.read_csv("data/walmart/walmart_merged.csv", parse_dates=["Date"])
    except FileNotFoundError:
        return pd.DataFrame()


# -- Load everything --
config = load_config()
predictions = load_predictions()
validation = load_validation()
inventory_plan = load_inventory_plan()
routes = load_routes()
metrics = load_metrics()
sim_results = load_simulation_results()
walmart_data = load_walmart_merged()


# -- Header --
st.markdown("""
<div class="main-header">
    <h1>Supply Chain Optimization System</h1>
    <p>Demand Forecasting • Inventory Optimization • Route Planning • Risk Simulation</p>
</div>
""", unsafe_allow_html=True)


# -- Sidebar --
with st.sidebar:
    st.markdown('<div class="sidebar-title">Control Panel</div>', unsafe_allow_html=True)

    st.markdown("**Data Selection**")
    available_stores = sorted(predictions["Store"].unique().tolist()) if not predictions.empty else list(range(1, 46))
    selected_store = st.selectbox("Select Store", available_stores, index=0)

    if not predictions.empty:
        store_depts = sorted(predictions[predictions["Store"] == selected_store]["Dept"].unique().tolist())
        if store_depts:
            selected_dept = st.selectbox("Select Department", store_depts, index=0)
        else:
            selected_dept = 1
    else:
        selected_dept = st.number_input("Department", min_value=1, max_value=99, value=1)

    st.divider()

    st.markdown("**Simulation Scenarios**")
    sim_scenario = st.radio(
        "Select Scenario to view",
        ["Baseline (No Simulation)", "Demand Spike", "Supply Delay", "Warehouse Failure"],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**Data Export**")
    if not predictions.empty:
        csv_pred = predictions.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv_pred, "predictions.csv", "text/csv", use_container_width=True)

    if not inventory_plan.empty:
        csv_inv = inventory_plan.to_csv(index=False).encode("utf-8")
        st.download_button("Download Inventory Plan", csv_inv, "inventory_plan.csv", "text/csv", use_container_width=True)

    if not routes.empty:
        csv_routes = routes.to_csv(index=False).encode("utf-8")
        st.download_button("Download Routes", csv_routes, "optimized_routes.csv", "text/csv", use_container_width=True)


# -- Main Tabs --
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Demand Forecast",
    "Inventory",
    "Routes",
    "Simulation"
])


# -----------------------------------------------------------------------------
# TAB 1: Overview
# -----------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Forecast MAE", f"${metrics.get('val_mae', 0):,.0f}" if metrics else "N/A")
    with col2:
        st.metric("Forecast R²", f"{metrics.get('val_r2', 0):.3f}" if metrics else "N/A")
    with col3:
        inv_cost = inventory_plan["Total_Cost"].sum() if not inventory_plan.empty else 0
        st.metric("Total Inventory Cost", f"${inv_cost:,.0f}" if inv_cost else "N/A")
    with col4:
        rt_dist = routes.groupby("Vehicle")["Cumulative_Distance_km"].max().sum() if not routes.empty else 0
        st.metric("Total Route Distance", f"{rt_dist:,.0f} km" if rt_dist else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-header">Cost Savings Analysis</div>', unsafe_allow_html=True)

        if not inventory_plan.empty:
            inv_config = config.get("inventory", {})
            baseline_cost = 49350302.77 # Hardcoded from previous run for demo consistency
            optimized_cost = inventory_plan["Total_Cost"].sum()
            savings = max(0, baseline_cost - optimized_cost)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Naive (Weekly Order)"], y=[baseline_cost],
                marker_color="#4a5568", name="Baseline",
                text=[f"${baseline_cost:,.0f}"], textposition="auto",
            ))
            fig.add_trace(go.Bar(
                x=["Optimized (LP)"], y=[optimized_cost],
                marker_color="#667eea", name="Optimized",
                text=[f"${optimized_cost:,.0f}"], textposition="auto",
            ))
            fig.update_layout(
                yaxis_title="Total Cost ($)",
                template="plotly_dark",
                height=350,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Estimated Inventory Savings", f"${savings:,.0f}")
        else:
            st.info("Run the pipeline to see cost analysis.")

    with col_right:
        st.markdown('<div class="section-header">Delivery Efficiency</div>', unsafe_allow_html=True)

        if not routes.empty:
            baseline_dist = 62021.9 # Hardcoded from previous run for demo consistency
            optimized_dist = routes.groupby("Vehicle")["Cumulative_Distance_km"].max().sum()
            dist_savings = max(0, baseline_dist - optimized_dist)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Individual Trips"], y=[baseline_dist],
                marker_color="#4a5568", name="Baseline",
                text=[f"{baseline_dist:,.0f} km"], textposition="auto",
            ))
            fig.add_trace(go.Bar(
                x=["Optimized (VRP)"], y=[optimized_dist],
                marker_color="#667eea", name="Optimized",
                text=[f"{optimized_dist:,.0f} km"], textposition="auto",
            ))
            fig.update_layout(
                yaxis_title="Total Distance (km)",
                template="plotly_dark",
                height=350,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Distance Saved", f"{dist_savings:,.0f} km")
        else:
            st.info("Run the pipeline to see routing analysis.")


# -----------------------------------------------------------------------------
# TAB 2: Demand Forecast
# -----------------------------------------------------------------------------
with tab2:
    st.markdown(f'<div class="section-header">Forecast Analysis: Store {selected_store}, Dept {selected_dept}</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown("**Historical vs Predicted Sales**")
        if not walmart_data.empty and not predictions.empty:
            hist = walmart_data[
                (walmart_data["Store"] == selected_store) &
                (walmart_data["Dept"] == selected_dept)
            ].sort_values("Date")

            pred = predictions[
                (predictions["Store"] == selected_store) &
                (predictions["Dept"] == selected_dept)
            ].copy()
            if not pred.empty:
                pred["Date"] = pd.to_datetime(pred["Date"])
                pred = pred.sort_values("Date")

            fig = go.Figure()

            if not hist.empty:
                fig.add_trace(go.Scatter(
                    x=hist["Date"], y=hist["Weekly_Sales"],
                    mode="lines", name="Historical",
                    line=dict(color="#a0aec0", width=1.5),
                ))

            if not pred.empty:
                fig.add_trace(go.Scatter(
                    x=pred["Date"], y=pred["Predicted_Weekly_Sales"],
                    mode="lines+markers", name="Predicted",
                    line=dict(color="#667eea", width=2, dash="dot"),
                    marker=dict(size=6),
                ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Weekly Sales ($)",
                template="plotly_dark",
                height=400,
                margin=dict(l=0, r=0, t=20, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available.")

    with col_right:
        st.markdown("**Feature Importance**")
        if metrics and "feature_importance" in metrics:
            fi = metrics["feature_importance"]
            fi_df = pd.DataFrame({
                "Feature": list(fi.keys()),
                "Importance": list(fi.values()),
            }).sort_values("Importance", ascending=True).tail(12)

            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         template="plotly_dark", color_discrete_sequence=["#667eea"])
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=20, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_yaxes(title="")
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Inventory
# -----------------------------------------------------------------------------
with tab3:
    st.markdown(f'<div class="section-header">Inventory Plan: Store {selected_store}</div>', unsafe_allow_html=True)

    if not inventory_plan.empty:
        store_inv = inventory_plan[inventory_plan["Store"] == selected_store]

        if not store_inv.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Orders Placed", f'{(store_inv["Optimal_Order"] > 0).sum()}')
            with col2:
                st.metric("Avg Inventory", f'{store_inv["Inventory_Level"].mean():,.0f}')
            with col3:
                st.metric("Total Store Cost", f'${store_inv["Total_Cost"].sum():,.0f}')
            with col4:
                st.metric("Avg Order Size", f'{store_inv[store_inv["Optimal_Order"] > 0]["Optimal_Order"].mean():,.0f}')

            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns(2, gap="large")

            with col_left:
                st.markdown("**Demand vs Inventory Level**")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(
                        x=store_inv["Week"], y=store_inv["Demand"],
                        name="Demand", marker_color="#4a5568", opacity=0.8
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=store_inv["Week"], y=store_inv["Inventory_Level"],
                        name="Inventory Level", mode="lines+markers",
                        line=dict(color="#667eea", width=2),
                        marker=dict(size=6),
                    ),
                    secondary_y=True,
                )
                fig.update_layout(
                    template="plotly_dark", height=380,
                    margin=dict(l=0, r=0, t=20, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.markdown("**Cost Breakdown**")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=store_inv["Week"], y=store_inv["Holding_Cost"],
                    name="Holding Cost", marker_color="#a0aec0"
                ))
                fig.add_trace(go.Bar(
                    x=store_inv["Week"], y=store_inv["Ordering_Cost"],
                    name="Ordering Cost", marker_color="#667eea"
                ))
                fig.update_layout(
                    template="plotly_dark", height=380, barmode="stack",
                    margin=dict(l=0, r=0, t=20, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Detailed Schedule**")
            st.dataframe(store_inv.style.format({
                "Demand": "{:,.0f}",
                "Optimal_Order": "{:,.0f}",
                "Inventory_Level": "{:,.0f}",
                "Holding_Cost": "${:,.2f}",
                "Ordering_Cost": "${:,.2f}",
                "Total_Cost": "${:,.2f}",
            }), use_container_width=True, height=250)
        else:
            st.warning(f"No inventory data for Store {selected_store}.")
    else:
        st.info("No inventory plan available.")


# -----------------------------------------------------------------------------
# TAB 4: Routes
# -----------------------------------------------------------------------------
with tab4:
    st.markdown('<div class="section-header">Optimized Delivery Routes</div>', unsafe_allow_html=True)

    if not routes.empty:
        vehicles = routes["Vehicle"].unique()
        vehicle_stats = routes.groupby("Vehicle").agg(
            Stops=("Stop_Order", "max"),
            Total_Distance=("Cumulative_Distance_km", "max"),
            Total_Load=("Cumulative_Load", "max"),
        ).reset_index()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vehicles Used", len(vehicles))
        with col2:
            st.metric("Total Fleet Distance", f"{vehicle_stats['Total_Distance'].sum():,.0f} km")
        with col3:
            st.metric("Total Completed Stops", f"{vehicle_stats['Stops'].sum():,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # We use st_folium but need to constraint it nicely
        st.markdown("**System Geographic Map**")
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="CartoDB dark_matter")

        colors = ["#667eea", "#764ba2", "#5aa897", "#f6a5c0", "#f9d56e", "#6b5b95", "#d64161", "#ff7b25"]

        if config.get("routing", {}).get("depots"):
            for depot in config["routing"]["depots"]:
                folium.Marker(
                    [depot["lat"], depot["lon"]],
                    popup=f"<b>{depot['name']}</b><br>Depot",
                    icon=folium.Icon(color="red", icon="info-sign"),
                ).add_to(m)

        for i, vehicle in enumerate(vehicles):
            vdata = routes[routes["Vehicle"] == vehicle].sort_values("Stop_Order")
            color = colors[i % len(colors)]

            coords = list(zip(vdata["Lat"], vdata["Lon"]))
            if len(coords) > 1:
                folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m)

            for _, row in vdata.iterrows():
                if not row.get("Is_Depot", False):
                    folium.CircleMarker(
                        [row["Lat"], row["Lon"]],
                        radius=5, color=color, fill=True,
                        fillColor=color, fillOpacity=0.9,
                        popup=f"Location: {row['Location']}<br>Vehicle: {vehicle}",
                    ).add_to(m)

        # Place folium in a container
        with st.container():
            st_folium(m, width=None, height=450, use_container_width=True, returned_objects=[])

    else:
        st.info("No route data available. Run the pipeline first.")


# -----------------------------------------------------------------------------
# TAB 5: Simulation
# -----------------------------------------------------------------------------
with tab5:
    st.markdown('<div class="section-header">Disruption Event Simulation</div>', unsafe_allow_html=True)

    if sim_results:
        ds = sim_results.get("demand_spike", {})
        sd = sim_results.get("supply_delay", {})
        wf = sim_results.get("warehouse_failure", {})

        if sim_scenario == "Demand Spike":
            st.markdown("### Scenaro: Demand Spike (+20%)")
            st.markdown("Measuring system resilience against sudden surges in product popularity.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Demand Increase", f'{ds.get("demand_increase_pct", 0):.0f}%')
            with c2:
                st.metric("Cost Increase", f'${ds.get("cost_increase", 0):,.0f}')
            with c3:
                st.metric("Extra Vehicles Required", ds.get("additional_vehicles_needed", 0))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Baseline Cost", "Post-Spike Cost"],
                y=[ds.get("baseline_inventory_cost", 0), ds.get("new_inventory_cost", 0)],
                marker_color=["#4a5568", "#667eea"],
                text=[f'${ds.get("baseline_inventory_cost", 0):,.0f}',
                      f'${ds.get("new_inventory_cost", 0):,.0f}'],
                textposition="auto",
            ))
            fig.update_layout(
                yaxis_title="Total Cost ($)",
                template="plotly_dark", height=400,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif sim_scenario == "Supply Delay":
            st.markdown("### Scenaro: Global Supply Delay")
            st.markdown("Measuring impact of inbound shipping delays enforcing larger safety stocks.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Lead Time Delay", f'{sd.get("delay_days", 0)} days')
            with c2:
                st.metric("Required Safety Stock", f'{sd.get("extra_safety_stock_total", 0):,.0f} units')
            with c3:
                st.metric("Cost Increase", f'${sd.get("cost_increase", 0):,.0f}')

        elif sim_scenario == "Warehouse Failure":
            st.markdown("### Scenaro: Structural Warehouse Failure")
            st.markdown("Measuring routing fallback when a primary distribution center goes offline.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Failed Infrastructure", wf.get("failed_depot", "N/A"))
            with c2:
                st.metric("Stores Reassigned", wf.get("stores_reassigned", 0))
            with c3:
                st.metric("Distance Increase", f'{wf.get("distance_increase_km", 0):,.0f} km')

        else:
            st.info("Select a scenario from the sidebar to view simulation impacts.")

    else:
        st.info("No simulation results available.")
