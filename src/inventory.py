"""
Inventory Optimization Module
==============================
Uses PuLP linear programming to optimize inventory levels per store.
Minimizes total cost (holding + ordering) subject to demand satisfaction and safety stock.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from pulp import (
    LpMinimize, LpProblem, LpVariable, LpStatus, lpSum, value
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def aggregate_store_demand(predictions_df):
    """
    Aggregate predicted demand per store per week (across departments).
    Returns a DataFrame with Store, Date, Total_Demand.
    """
    agg = predictions_df.groupby(["Store", "Date"]).agg(
        Total_Demand=("Predicted_Weekly_Sales", "sum")
    ).reset_index()
    agg = agg.sort_values(["Store", "Date"]).reset_index(drop=True)
    return agg


def optimize_inventory_for_store(store_id, demand_series, config):
    """
    Solve inventory optimization for a single store using LP.

    Decision variables:
        - order[t]: quantity ordered in period t
        - inventory[t]: inventory level at end of period t

    Objective: Minimize holding_cost * sum(inventory) + ordering_cost * num_orders

    Constraints:
        - inventory[t] = inventory[t-1] + order[t] - demand[t]
        - inventory[t] >= safety_stock
        - order[t] >= 0
    """
    inv_config = config["inventory"]
    holding_cost = inv_config["holding_cost_per_unit_per_week"]
    ordering_cost = inv_config["ordering_cost_per_order"]
    safety_weeks = inv_config["safety_stock_weeks"]
    max_order = inv_config["max_order_quantity"]

    T = len(demand_series)
    demands = demand_series.values

    # Safety stock = average demand * safety_weeks
    avg_demand = np.mean(demands)
    safety_stock = avg_demand * safety_weeks

    # Create LP problem
    prob = LpProblem(f"Inventory_Store_{store_id}", LpMinimize)

    # Decision variables
    order = [LpVariable(f"order_{t}", lowBound=0, upBound=max_order) for t in range(T)]
    inventory = [LpVariable(f"inv_{t}", lowBound=0) for t in range(T)]
    is_ordering = [LpVariable(f"is_order_{t}", cat="Binary") for t in range(T)]

    # Big M for binary ordering indicator
    M = max_order + 1

    # Objective: minimize holding cost + fixed ordering cost
    prob += (
        lpSum([holding_cost * inventory[t] for t in range(T)]) +
        lpSum([ordering_cost * is_ordering[t] for t in range(T)])
    )

    # Constraints
    initial_inventory = safety_stock * 1.5  # Start with comfortable stock

    for t in range(T):
        # Inventory balance
        if t == 0:
            prob += inventory[t] == initial_inventory + order[t] - demands[t]
        else:
            prob += inventory[t] == inventory[t - 1] + order[t] - demands[t]

        # Safety stock
        prob += inventory[t] >= safety_stock

        # Link ordering indicator to order quantity
        prob += order[t] <= M * is_ordering[t]

    # Solve
    prob.solve()
    status = LpStatus[prob.status]

    if status != "Optimal":
        logger.warning(f"  Store {store_id}: LP status = {status} (using fallback)")
        # Fallback: order demand + safety stock gap
        results = []
        current_inv = initial_inventory
        for t in range(T):
            needed = max(0, demands[t] + safety_stock - current_inv)
            current_inv = current_inv + needed - demands[t]
            results.append({
                "Store": store_id,
                "Week": t + 1,
                "Demand": round(demands[t], 2),
                "Optimal_Order": round(needed, 2),
                "Inventory_Level": round(current_inv, 2),
                "Holding_Cost": round(holding_cost * current_inv, 2),
                "Ordering_Cost": round(ordering_cost if needed > 0 else 0, 2),
                "Total_Cost": round(holding_cost * current_inv + (ordering_cost if needed > 0 else 0), 2),
            })
        return pd.DataFrame(results)

    # Extract results
    results = []
    total_cost = value(prob.objective)
    for t in range(T):
        ord_qty = value(order[t])
        inv_level = value(inventory[t])
        results.append({
            "Store": store_id,
            "Week": t + 1,
            "Demand": round(demands[t], 2),
            "Optimal_Order": round(ord_qty, 2),
            "Inventory_Level": round(inv_level, 2),
            "Holding_Cost": round(holding_cost * inv_level, 2),
            "Ordering_Cost": round(ordering_cost if ord_qty > 0.01 else 0, 2),
            "Total_Cost": round(holding_cost * inv_level + (ordering_cost if ord_qty > 0.01 else 0), 2),
        })

    return pd.DataFrame(results)


def optimize_inventory(config, predictions_df=None):
    """
    Run inventory optimization for all stores.
    """
    if predictions_df is None:
        pred_path = config["output_paths"]["predictions"]
        predictions_df = pd.read_csv(pred_path)
        logger.info(f"Loaded predictions from {pred_path}")

    # Aggregate store-level demand
    store_demand = aggregate_store_demand(predictions_df)
    stores = store_demand["Store"].unique()

    logger.info(f"Optimizing inventory for {len(stores)} stores...")

    all_results = []
    for store_id in stores:
        store_data = store_demand[store_demand["Store"] == store_id]
        demand_series = store_data["Total_Demand"]

        if len(demand_series) < 2:
            continue

        result = optimize_inventory_for_store(store_id, demand_series, config)
        all_results.append(result)

    if not all_results:
        logger.warning("No inventory optimization results generated!")
        return pd.DataFrame()

    inventory_plan = pd.concat(all_results, ignore_index=True)

    # Save
    output_path = config["output_paths"]["inventory_plan"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    inventory_plan.to_csv(output_path, index=False)
    logger.info(f"Inventory plan saved to {output_path} — shape: {inventory_plan.shape}")

    # Summary stats
    total_cost = inventory_plan["Total_Cost"].sum()
    total_orders = (inventory_plan["Optimal_Order"] > 0).sum()
    avg_inventory = inventory_plan["Inventory_Level"].mean()

    logger.info(f"  Total cost: ${total_cost:,.2f}")
    logger.info(f"  Total orders placed: {total_orders}")
    logger.info(f"  Average inventory level: {avg_inventory:,.2f}")

    return inventory_plan


def compute_baseline_cost(config, predictions_df=None):
    """
    Compute naive baseline cost (order exactly what you need each week, no optimization).
    Used for before/after comparison.
    """
    inv_config = config["inventory"]
    holding_cost = inv_config["holding_cost_per_unit_per_week"]
    ordering_cost = inv_config["ordering_cost_per_order"]

    if predictions_df is None:
        predictions_df = pd.read_csv(config["output_paths"]["predictions"])

    store_demand = aggregate_store_demand(predictions_df)
    stores = store_demand["Store"].unique()

    total_baseline_cost = 0
    for store_id in stores:
        store_data = store_demand[store_demand["Store"] == store_id]
        demands = store_data["Total_Demand"].values
        # Naive: order every week, hold safety stock buffer
        safety_stock = np.mean(demands) * inv_config["safety_stock_weeks"]
        for d in demands:
            total_baseline_cost += ordering_cost  # Order every week
            total_baseline_cost += holding_cost * (safety_stock + d * 0.5)  # Average daily holding

    return total_baseline_cost


def run_all(config_path="config.json"):
    """Run complete inventory optimization pipeline."""
    config = load_config(config_path)
    logger.info("=" * 60)
    logger.info("STARTING INVENTORY OPTIMIZATION PIPELINE")
    logger.info("=" * 60)

    inventory_plan = optimize_inventory(config)

    # Baseline comparison
    baseline = compute_baseline_cost(config)
    optimized = inventory_plan["Total_Cost"].sum()
    savings = baseline - optimized
    savings_pct = (savings / baseline * 100) if baseline > 0 else 0

    logger.info("=" * 60)
    logger.info("INVENTORY OPTIMIZATION COMPLETE")
    logger.info(f"  Baseline cost: ${baseline:,.2f}")
    logger.info(f"  Optimized cost: ${optimized:,.2f}")
    logger.info(f"  Savings: ${savings:,.2f} ({savings_pct:.1f}%)")
    logger.info("=" * 60)

    return inventory_plan


if __name__ == "__main__":
    run_all()
