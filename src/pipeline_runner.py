"""
Pipeline Runner Module
=======================
Wraps the entire supply chain pipeline into a single callable function
for use within the Streamlit dashboard. No disk I/O — everything stays
in memory via return values.
"""

import pandas as pd
import numpy as np
import json
import logging
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pulp import LpMinimize, LpProblem, LpVariable, LpStatus, lpSum, value

from src.data_processing import STORE_COORDINATES, STORE_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config used when user doesn't provide their own
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "inventory": {
        "holding_cost_per_unit_per_week": 0.05,
        "ordering_cost_per_order": 5000,
        "safety_stock_weeks": 1,
        "max_order_quantity": 100000000,
    },
    "routing": {
        "num_vehicles": 5,
        "vehicle_capacity": 50000000,
        "depots": [
            {"name": "Central Depot", "lat": 39.8283, "lon": -98.5795},
            {"name": "East Depot", "lat": 35.2271, "lon": -80.8431},
            {"name": "West Depot", "lat": 34.0522, "lon": -118.2437},
        ],
    },
    "forecasting": {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "test_size_weeks": 13,
        "lag_weeks": [1, 2, 4, 8, 12],
    },
    "simulation": {
        "demand_spike_factor": 1.2,
        "supply_delay_days": 7,
        "failed_depot_index": 0,
    },
}

FEATURE_COLS = [
    "Store", "Dept", "Type_encoded", "Size",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "IsHoliday", "Year", "Month", "Week", "DayOfYear", "Quarter",
]


def get_template_dataframe():
    """Return an empty DataFrame with the required column schema."""
    return pd.DataFrame(columns=[
        "Store", "Dept", "Date", "Weekly_Sales", "IsHoliday",
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "Type", "Size",
    ])


def validate_upload(df):
    """Validate that an uploaded DataFrame has the required columns. Returns (ok, error_msg)."""
    required = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday",
                 "Temperature", "Fuel_Price", "CPI", "Unemployment",
                 "Type", "Size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if len(df) < 100:
        return False, "Dataset too small. Need at least 100 rows for meaningful analysis."
    return True, ""


# ---------------------------------------------------------------------------
# Step 1 : Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    """Add time features, encodings, and coordinates to raw data."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    # Time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"] = df["Date"].dt.quarter

    # Encode categoricals
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    type_map = {"A": 0, "B": 1, "C": 2}
    df["Type_encoded"] = df["Type"].map(type_map).fillna(1)

    # Fill NaN in numeric columns
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # MarkDown columns (optional)
    for i in range(1, 6):
        md = f"MarkDown{i}"
        if md in df.columns:
            df[md] = df[md].fillna(0)
        else:
            df[md] = 0

    # Coordinates
    df["Latitude"] = df["Store"].map(lambda s: STORE_COORDINATES.get(int(s), (39.8, -98.5))[0])
    df["Longitude"] = df["Store"].map(lambda s: STORE_COORDINATES.get(int(s), (39.8, -98.5))[1])

    return df


# ---------------------------------------------------------------------------
# Step 2 : Forecasting
# ---------------------------------------------------------------------------
def create_lag_features(df, lag_weeks):
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    lag_cols = []
    for lag in lag_weeks:
        col_name = f"Sales_Lag_{lag}"
        df[col_name] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)
        lag_cols.append(col_name)

    df["Sales_RollingMean_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    lag_cols.append("Sales_RollingMean_4")

    df["Sales_RollingMean_12"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    lag_cols.append("Sales_RollingMean_12")

    df["Sales_RollingStd_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).std())
    df["Sales_RollingStd_4"] = df["Sales_RollingStd_4"].fillna(0)
    lag_cols.append("Sales_RollingStd_4")

    return df, lag_cols


def run_forecasting(df, config):
    """Train model, evaluate, and generate 13-week predictions. Returns dict of results."""
    fc = config["forecasting"]
    lag_weeks = fc["lag_weeks"]

    df, lag_cols = create_lag_features(df, lag_weeks)
    feature_cols = FEATURE_COLS + lag_cols

    df_model = df.dropna(subset=lag_cols).copy()

    # Time split
    test_weeks = fc["test_size_weeks"]
    max_date = df_model["Date"].max()
    split_date = max_date - pd.Timedelta(weeks=test_weeks)

    train_df = df_model[df_model["Date"] <= split_date]
    val_df = df_model[df_model["Date"] > split_date]

    X_train = train_df[feature_cols].values
    y_train = train_df["Weekly_Sales"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["Weekly_Sales"].values

    model = GradientBoostingRegressor(
        n_estimators=fc["n_estimators"], max_depth=fc["max_depth"],
        learning_rate=fc["learning_rate"], subsample=0.8,
        min_samples_split=10, min_samples_leaf=5, random_state=42, verbose=0,
    )
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    metrics = {
        "val_mae": float(mean_absolute_error(y_val, y_pred_val)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        "val_r2": float(r2_score(y_val, y_pred_val)),
        "feature_importance": dict(zip(feature_cols, model.feature_importances_.tolist())),
        "feature_columns": feature_cols,
    }

    # Validation predictions
    val_results = val_df[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    val_results["Predicted_Sales"] = y_pred_val

    # Generate future predictions
    future_weeks = 13
    last_date = df["Date"].max()
    all_predictions = []

    for store in df["Store"].unique():
        store_data = df[df["Store"] == store]
        for dept in store_data["Dept"].unique():
            dept_data = store_data[store_data["Dept"] == dept].sort_values("Date")
            if len(dept_data) < 13:
                continue
            last_row_dict = dept_data.iloc[-1].to_dict()
            recent_sales = dept_data["Weekly_Sales"].values.copy()

            for w in range(1, future_weeks + 1):
                future_date = last_date + pd.Timedelta(weeks=w)
                row = last_row_dict.copy()
                row["Date"] = future_date
                row["Year"] = future_date.year
                row["Month"] = future_date.month
                row["Week"] = future_date.isocalendar()[1]
                row["DayOfYear"] = future_date.timetuple().tm_yday
                row["Quarter"] = (future_date.month - 1) // 3 + 1

                for lag_w in lag_weeks:
                    col = f"Sales_Lag_{lag_w}"
                    idx = len(recent_sales) - lag_w
                    row[col] = recent_sales[idx] if idx >= 0 else recent_sales[0]
                row["Sales_RollingMean_4"] = recent_sales[-4:].mean()
                row["Sales_RollingMean_12"] = recent_sales[-12:].mean()
                row["Sales_RollingStd_4"] = recent_sales[-4:].std() if len(recent_sales) >= 4 else 0

                X_list = [row.get(c, 0) for c in feature_cols]
                X = np.array([X_list])
                pred = max(0, model.predict(X)[0])
                recent_sales = np.append(recent_sales, pred)

                all_predictions.append({
                    "Store": int(row["Store"]),
                    "Dept": int(row["Dept"]),
                    "Date": future_date.strftime("%Y-%m-%d"),
                    "Predicted_Weekly_Sales": round(pred, 2),
                })

    predictions_df = pd.DataFrame(all_predictions)
    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions_df,
        "validation": val_results,
        "historical_data": df,
    }


# ---------------------------------------------------------------------------
# Step 3 : Inventory Optimization
# ---------------------------------------------------------------------------
def run_inventory(predictions_df, config):
    """Run LP-based inventory optimization. Returns inventory plan DataFrame and baseline cost."""
    inv_cfg = config["inventory"]
    h = inv_cfg["holding_cost_per_unit_per_week"]
    o = inv_cfg["ordering_cost_per_order"]
    sw = inv_cfg["safety_stock_weeks"]
    max_q = inv_cfg["max_order_quantity"]

    store_demand = predictions_df.groupby(["Store", "Date"]).agg(
        Total_Demand=("Predicted_Weekly_Sales", "sum")).reset_index()
    store_demand = store_demand.sort_values(["Store", "Date"])

    all_results = []
    baseline_cost = 0

    for store_id in store_demand["Store"].unique():
        sd = store_demand[store_demand["Store"] == store_id]
        demands = sd["Total_Demand"].values
        T = len(demands)
        if T < 2:
            continue

        avg_d = np.mean(demands)
        safety = avg_d * sw
        init_inv = safety * 1.5

        prob = LpProblem(f"Inv_{store_id}", LpMinimize)
        order = [LpVariable(f"o_{t}", 0, max_q) for t in range(T)]
        inv = [LpVariable(f"i_{t}", 0) for t in range(T)]
        is_ord = [LpVariable(f"b_{t}", cat="Binary") for t in range(T)]
        M = max_q + 1

        prob += lpSum([h * inv[t] for t in range(T)]) + lpSum([o * is_ord[t] for t in range(T)])
        for t in range(T):
            if t == 0:
                prob += inv[t] == init_inv + order[t] - demands[t]
            else:
                prob += inv[t] == inv[t - 1] + order[t] - demands[t]
            prob += inv[t] >= safety
            prob += order[t] <= M * is_ord[t]

        prob.solve()
        status = LpStatus[prob.status]

        for t in range(T):
            if status == "Optimal":
                oq = value(order[t])
                il = value(inv[t])
            else:
                needed = max(0, demands[t] + safety - (init_inv if t == 0 else all_results[-1]["Inventory_Level"]))
                oq = needed
                il = (init_inv if t == 0 else all_results[-1]["Inventory_Level"]) + oq - demands[t]

            all_results.append({
                "Store": store_id, "Week": t + 1,
                "Demand": round(demands[t], 2),
                "Optimal_Order": round(oq, 2),
                "Inventory_Level": round(il, 2),
                "Holding_Cost": round(h * il, 2),
                "Ordering_Cost": round(o if oq > 0.01 else 0, 2),
                "Total_Cost": round(h * il + (o if oq > 0.01 else 0), 2),
            })

        # Baseline for this store
        for d in demands:
            baseline_cost += o + h * (safety + d * 0.5)

    inv_plan = pd.DataFrame(all_results)
    return inv_plan, baseline_cost


# ---------------------------------------------------------------------------
# Step 4 : Route Optimization
# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def run_routing(predictions_df, config):
    """Run VRP nearest-neighbor routing. Returns routes DataFrame and baseline distance."""
    rc = config["routing"]
    depots = rc["depots"]
    num_v = rc["num_vehicles"]
    cap = rc["vehicle_capacity"]

    store_demands = predictions_df.groupby("Store")["Predicted_Weekly_Sales"].sum().to_dict()

    # Build locations list
    locations, names, demands_list = [], [], []
    for dep in depots:
        locations.append((dep["lat"], dep["lon"]))
        names.append(dep["name"])
        demands_list.append(0)

    store_ids = sorted(set(predictions_df["Store"].unique().tolist()))
    for sid in store_ids:
        lat, lon = STORE_COORDINATES.get(int(sid), (39.8, -98.5))
        locations.append((lat, lon))
        names.append(STORE_NAMES.get(int(sid), f"Store {sid}"))
        demands_list.append(store_demands.get(sid, 5000))

    n = len(locations)
    nd = len(depots)

    # Distance matrix
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dm[i][j] = _haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])

    # Assign to nearest depot
    depot_assign = {d: [] for d in range(nd)}
    for s in range(nd, n):
        nearest = min(range(nd), key=lambda d: dm[d][s])
        depot_assign[nearest].append(s)

    # Solve per depot
    all_stops = []
    vpd = max(1, num_v // nd)

    for di, store_idxs in depot_assign.items():
        if not store_idxs:
            continue
        unvisited = set(store_idxs)
        vid = 0
        while unvisited and vid < vpd:
            current = di
            load = 0
            dist = 0
            stops = [{"Vehicle": f"V{di+1}_{vid+1}", "Stop_Order": 0,
                       "Location": names[di], "Lat": locations[di][0], "Lon": locations[di][1],
                       "Demand": 0, "Cumulative_Load": 0, "Distance_km": 0,
                       "Cumulative_Distance_km": 0, "Is_Depot": True}]
            so = 1
            while unvisited:
                best, bd = None, float("inf")
                for s in unvisited:
                    d = dm[current][s]
                    if d < bd and load + demands_list[s] <= cap:
                        best, bd = s, d
                if best is None:
                    break
                unvisited.remove(best)
                load += demands_list[best]
                dist += bd
                stops.append({"Vehicle": f"V{di+1}_{vid+1}", "Stop_Order": so,
                               "Location": names[best], "Lat": locations[best][0], "Lon": locations[best][1],
                               "Demand": round(demands_list[best], 2), "Cumulative_Load": round(load, 2),
                               "Distance_km": round(bd, 2), "Cumulative_Distance_km": round(dist, 2),
                               "Is_Depot": False})
                current = best
                so += 1

            if len(stops) > 1:
                rd = dm[current][di]
                dist += rd
                stops.append({"Vehicle": f"V{di+1}_{vid+1}", "Stop_Order": so,
                               "Location": f"{names[di]} (return)", "Lat": locations[di][0], "Lon": locations[di][1],
                               "Demand": 0, "Cumulative_Load": round(load, 2),
                               "Distance_km": round(rd, 2), "Cumulative_Distance_km": round(dist, 2),
                               "Is_Depot": True})
            all_stops.extend(stops)
            vid += 1

    routes_df = pd.DataFrame(all_stops)

    # Baseline distance (individual round trips)
    baseline_dist = 0
    for sid in store_ids:
        lat, lon = STORE_COORDINATES.get(int(sid), (39.8, -98.5))
        md = min(_haversine(lat, lon, d["lat"], d["lon"]) for d in depots)
        baseline_dist += md * 2

    return routes_df, baseline_dist


# ---------------------------------------------------------------------------
# Step 5 : Simulation
# ---------------------------------------------------------------------------
def run_simulation(predictions_df, inv_plan, routes_df, config):
    """Run the three disruption scenarios. Returns metrics dict."""
    sim = config["simulation"]
    inv_cfg = config["inventory"]
    baseline_inv_cost = inv_plan["Total_Cost"].sum() if not inv_plan.empty else 0
    baseline_dist = routes_df.groupby("Vehicle")["Cumulative_Distance_km"].max().sum() if not routes_df.empty else 0

    # -- Demand Spike --
    sf = sim["demand_spike_factor"]
    avg_before = predictions_df["Predicted_Weekly_Sales"].mean()
    avg_after = avg_before * sf
    new_inv_cost = baseline_inv_cost * sf * 1.1
    ds = {
        "baseline_avg_demand": float(avg_before), "new_avg_demand": float(avg_after),
        "demand_increase_pct": float((sf - 1) * 100),
        "baseline_inventory_cost": float(baseline_inv_cost), "new_inventory_cost": float(new_inv_cost),
        "cost_increase": float(new_inv_cost - baseline_inv_cost),
        "cost_increase_pct": float((new_inv_cost / max(baseline_inv_cost, 1) - 1) * 100),
        "additional_vehicles_needed": max(0, int(np.ceil((sf - 1) * config["routing"]["num_vehicles"]))),
    }

    # -- Supply Delay --
    dd = sim["supply_delay_days"]
    h = inv_cfg["holding_cost_per_unit_per_week"]
    avg_weekly = predictions_df.groupby("Store")["Predicted_Weekly_Sales"].mean()
    extra_safety = (avg_weekly / 7 * dd).sum()
    add_hold = extra_safety * h * (dd / 7)
    sd = {
        "delay_days": dd, "extra_safety_stock_total": float(extra_safety),
        "additional_holding_cost": float(add_hold),
        "baseline_inventory_cost": float(baseline_inv_cost),
        "new_inventory_cost": float(baseline_inv_cost + add_hold),
        "cost_increase": float(add_hold),
        "cost_increase_pct": float(add_hold / max(baseline_inv_cost, 1) * 100),
        "stores_at_stockout_risk": 0, "stockout_risk_pct": 0.0,
    }

    # -- Warehouse Failure --
    depots = config["routing"]["depots"]
    fi = sim["failed_depot_index"]
    failed = depots[fi] if fi < len(depots) else depots[0]
    remaining = [d for i, d in enumerate(depots) if i != fi]
    new_dist = 0
    reassigned = 0
    for sid, (lat, lon) in STORE_COORDINATES.items():
        md = min(_haversine(lat, lon, d["lat"], d["lon"]) for d in remaining)
        new_dist += md * 2
        fd = _haversine(lat, lon, failed["lat"], failed["lon"])
        od = min(_haversine(lat, lon, d["lat"], d["lon"]) for d in remaining)
        if fd <= od:
            reassigned += 1

    wf = {
        "failed_depot": failed["name"], "stores_reassigned": reassigned,
        "baseline_distance_km": float(baseline_dist), "new_distance_km": float(new_dist),
        "distance_increase_km": float(new_dist - baseline_dist),
        "distance_increase_pct": float((new_dist / max(baseline_dist, 1) - 1) * 100),
        "additional_fuel_cost_estimate": float((new_dist - baseline_dist) * 1.5),
    }

    return {"demand_spike": ds, "supply_delay": sd, "warehouse_failure": wf}


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------
def run_full_pipeline(df, config=None, progress_callback=None):
    """
    Run the complete pipeline on an in-memory DataFrame.
    progress_callback(step_name, step_number, total_steps) is called after each stage.
    Returns a dict with all results.
    """
    if config is None:
        config = DEFAULT_CONFIG

    total = 5

    def _progress(name, step):
        if progress_callback:
            progress_callback(name, step, total)

    # 1. Feature engineering
    _progress("Engineering features...", 1)
    df = engineer_features(df)

    # 2. Forecasting
    _progress("Training demand forecasting model...", 2)
    fc_results = run_forecasting(df, config)

    # 3. Inventory
    _progress("Optimizing inventory levels...", 3)
    inv_plan, baseline_cost = run_inventory(fc_results["predictions"], config)

    # 4. Routing
    _progress("Optimizing delivery routes...", 4)
    routes_df, baseline_dist = run_routing(fc_results["predictions"], config)

    # 5. Simulation
    _progress("Running disruption simulations...", 5)
    sim_results = run_simulation(fc_results["predictions"], inv_plan, routes_df, config)

    return {
        "historical_data": fc_results["historical_data"],
        "predictions": fc_results["predictions"],
        "validation": fc_results["validation"],
        "metrics": fc_results["metrics"],
        "inventory_plan": inv_plan,
        "inventory_baseline_cost": baseline_cost,
        "routes": routes_df,
        "routes_baseline_dist": baseline_dist,
        "simulation": sim_results,
        "config": config,
    }
