"""
Simulation Module
==================
Simulates supply-chain disruption scenarios and measures system adaptation.
Scenarios: demand spike, supply delay, warehouse failure.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import copy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def load_baseline_results(config):
    """Load baseline outputs from previous pipeline runs."""
    results = {}

    # Predictions
    pred_path = config["output_paths"]["predictions"]
    if os.path.exists(pred_path):
        results["predictions"] = pd.read_csv(pred_path)
    else:
        logger.warning(f"Predictions not found at {pred_path}")
        results["predictions"] = pd.DataFrame()

    # Inventory plan
    inv_path = config["output_paths"]["inventory_plan"]
    if os.path.exists(inv_path):
        results["inventory_plan"] = pd.read_csv(inv_path)
    else:
        logger.warning(f"Inventory plan not found at {inv_path}")
        results["inventory_plan"] = pd.DataFrame()

    # Routes
    route_path = config["output_paths"]["optimized_routes"]
    if os.path.exists(route_path):
        results["routes"] = pd.read_csv(route_path)
    else:
        logger.warning(f"Routes not found at {route_path}")
        results["routes"] = pd.DataFrame()

    return results


def simulate_demand_spike(config, baseline, spike_factor=None, affected_stores=None):
    """
    Scenario 1: Demand Spike
    - Increase predicted demand by spike_factor (default: +20%)
    - Re-compute inventory costs
    - Show impact on routes (higher loads)
    """
    sim_config = config["simulation"]
    if spike_factor is None:
        spike_factor = sim_config["demand_spike_factor"]

    predictions = baseline["predictions"].copy()
    inventory_plan = baseline["inventory_plan"].copy()

    if affected_stores is None:
        # Affect all stores
        predictions["Predicted_Weekly_Sales"] *= spike_factor
    else:
        mask = predictions["Store"].isin(affected_stores)
        predictions.loc[mask, "Predicted_Weekly_Sales"] *= spike_factor

    # Re-compute inventory impact
    inv_config = config["inventory"]
    holding_cost = inv_config["holding_cost_per_unit_per_week"]
    ordering_cost = inv_config["ordering_cost_per_order"]

    # Simulate new inventory costs
    new_demand_by_store = predictions.groupby("Store")["Predicted_Weekly_Sales"].mean()
    baseline_demand_by_store = baseline["predictions"].groupby("Store")["Predicted_Weekly_Sales"].mean()

    baseline_total_cost = baseline["inventory_plan"]["Total_Cost"].sum() if not baseline["inventory_plan"].empty else 0

    # Estimate new cost: proportionally scale based on demand increase
    demand_ratio = new_demand_by_store.sum() / max(baseline_demand_by_store.sum(), 1)
    new_total_cost = baseline_total_cost * demand_ratio * 1.1  # 10% penalty for rush orders

    # Route impact: higher loads
    routes = baseline["routes"].copy() if not baseline["routes"].empty else pd.DataFrame()
    if not routes.empty:
        routes["Demand"] *= spike_factor
        routes["Cumulative_Load"] *= spike_factor

    return {
        "scenario": "Demand Spike",
        "spike_factor": spike_factor,
        "predictions": predictions,
        "routes": routes,
        "metrics": {
            "baseline_avg_demand": float(baseline_demand_by_store.mean()),
            "new_avg_demand": float(new_demand_by_store.mean()),
            "demand_increase_pct": float((spike_factor - 1) * 100),
            "baseline_inventory_cost": float(baseline_total_cost),
            "new_inventory_cost": float(new_total_cost),
            "cost_increase": float(new_total_cost - baseline_total_cost),
            "cost_increase_pct": float((new_total_cost / max(baseline_total_cost, 1) - 1) * 100),
            "additional_vehicles_needed": max(0, int(np.ceil((spike_factor - 1) * config["routing"]["num_vehicles"]))),
        },
    }


def simulate_supply_delay(config, baseline, delay_days=None):
    """
    Scenario 2: Supply Delay
    - Assume orders are delayed by N days
    - Stockout risk increases
    - Extra holding cost for safety buffer
    """
    sim_config = config["simulation"]
    if delay_days is None:
        delay_days = sim_config["supply_delay_days"]

    inv_config = config["inventory"]
    holding_cost = inv_config["holding_cost_per_unit_per_week"]

    inventory_plan = baseline["inventory_plan"].copy()
    predictions = baseline["predictions"].copy()

    if inventory_plan.empty:
        return {
            "scenario": "Supply Delay",
            "delay_days": delay_days,
            "metrics": {"error": "No baseline inventory plan available"},
        }

    # With delay, we need extra safety stock = avg_daily_demand * delay_days
    avg_weekly_demand = predictions.groupby("Store")["Predicted_Weekly_Sales"].mean()
    extra_safety_per_store = avg_weekly_demand / 7 * delay_days

    # Additional holding cost
    additional_holding = extra_safety_per_store.sum() * holding_cost * (delay_days / 7)

    baseline_cost = inventory_plan["Total_Cost"].sum()
    new_cost = baseline_cost + additional_holding

    # Stockout risk: stores where current inventory < extra safety needed
    stores_at_risk = 0
    for store_id in inventory_plan["Store"].unique():
        store_inv = inventory_plan[inventory_plan["Store"] == store_id]["Inventory_Level"].min()
        extra_needed = extra_safety_per_store.get(store_id, 0)
        if store_inv < extra_needed:
            stores_at_risk += 1

    return {
        "scenario": "Supply Delay",
        "delay_days": delay_days,
        "predictions": predictions,
        "metrics": {
            "delay_days": delay_days,
            "extra_safety_stock_total": float(extra_safety_per_store.sum()),
            "additional_holding_cost": float(additional_holding),
            "baseline_inventory_cost": float(baseline_cost),
            "new_inventory_cost": float(new_cost),
            "cost_increase": float(new_cost - baseline_cost),
            "cost_increase_pct": float((new_cost / max(baseline_cost, 1) - 1) * 100),
            "stores_at_stockout_risk": int(stores_at_risk),
            "stockout_risk_pct": float(stores_at_risk / max(len(inventory_plan["Store"].unique()), 1) * 100),
        },
    }


def simulate_warehouse_failure(config, baseline, failed_depot_index=None):
    """
    Scenario 3: Warehouse/Depot Failure
    - Remove one depot
    - Re-route all stores to remaining depots
    - Measure increase in distance and cost
    """
    sim_config = config["simulation"]
    if failed_depot_index is None:
        failed_depot_index = sim_config["failed_depot_index"]

    route_config = config["routing"]
    depots = route_config["depots"]

    if failed_depot_index >= len(depots):
        return {
            "scenario": "Warehouse Failure",
            "metrics": {"error": "Invalid depot index"},
        }

    failed_depot = depots[failed_depot_index]
    remaining_depots = [d for i, d in enumerate(depots) if i != failed_depot_index]

    # Simulate re-routing
    from src.data_processing import STORE_COORDINATES
    from src.routing import haversine_distance

    # Calculate new total distance with remaining depots
    new_total_dist = 0
    stores_reassigned = 0
    for sid, (lat, lon) in STORE_COORDINATES.items():
        min_dist = float("inf")
        for depot in remaining_depots:
            d = haversine_distance(lat, lon, depot["lat"], depot["lon"])
            min_dist = min(min_dist, d)
        new_total_dist += min_dist * 2  # Round trip

        # Check if this store was originally closest to failed depot
        orig_dist_failed = haversine_distance(lat, lon, failed_depot["lat"], failed_depot["lon"])
        orig_min_other = min(
            haversine_distance(lat, lon, d["lat"], d["lon"])
            for d in remaining_depots
        )
        if orig_dist_failed <= orig_min_other:
            stores_reassigned += 1

    # Baseline distance
    routes = baseline["routes"]
    if not routes.empty:
        baseline_dist = routes.groupby("Vehicle")["Cumulative_Distance_km"].max().sum()
    else:
        baseline_dist = 0
        for sid, (lat, lon) in STORE_COORDINATES.items():
            min_dist = min(
                haversine_distance(lat, lon, d["lat"], d["lon"])
                for d in depots
            )
            baseline_dist += min_dist * 2

    return {
        "scenario": "Warehouse Failure",
        "failed_depot": failed_depot["name"],
        "metrics": {
            "failed_depot": failed_depot["name"],
            "remaining_depots": [d["name"] for d in remaining_depots],
            "stores_reassigned": stores_reassigned,
            "baseline_distance_km": float(baseline_dist),
            "new_distance_km": float(new_total_dist),
            "distance_increase_km": float(new_total_dist - baseline_dist),
            "distance_increase_pct": float((new_total_dist / max(baseline_dist, 1) - 1) * 100),
            "additional_fuel_cost_estimate": float((new_total_dist - baseline_dist) * 1.5),  # $1.50/km
        },
    }


def run_all_scenarios(config, baseline=None):
    """Run all simulation scenarios and return combined results."""
    if baseline is None:
        baseline = load_baseline_results(config)

    results = {}

    logger.info("Running Scenario 1: Demand Spike...")
    results["demand_spike"] = simulate_demand_spike(config, baseline)

    logger.info("Running Scenario 2: Supply Delay...")
    results["supply_delay"] = simulate_supply_delay(config, baseline)

    logger.info("Running Scenario 3: Warehouse Failure...")
    results["warehouse_failure"] = simulate_warehouse_failure(config, baseline)

    # Save metrics summary
    metrics_summary = {}
    for scenario_key, scenario_result in results.items():
        metrics_summary[scenario_key] = scenario_result.get("metrics", {})

    output_path = config["output_paths"]["simulation_results"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    logger.info(f"Simulation results saved to {output_path}")

    return results


def run_all(config_path="config.json"):
    """Run complete simulation pipeline."""
    config = load_config(config_path)
    logger.info("=" * 60)
    logger.info("STARTING SIMULATION PIPELINE")
    logger.info("=" * 60)

    baseline = load_baseline_results(config)
    results = run_all_scenarios(config, baseline)

    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE — Summary:")
    for key, res in results.items():
        metrics = res.get("metrics", {})
        logger.info(f"  [{key}] {json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=0)}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_all()
