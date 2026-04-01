"""
Demand Forecasting Module
=========================
Trains a Gradient Boosting model to predict weekly sales per store/department.
Uses lag features, time features, and external factors (weather, CPI, etc.).
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


FEATURE_COLS = [
    "Store", "Dept", "Type_encoded", "Size",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "IsHoliday", "Year", "Month", "Week", "DayOfYear", "Quarter",
]

LAG_FEATURE_NAMES = []  # Populated dynamically


def create_lag_features(df, lag_weeks):
    """
    Create lag features for Weekly_Sales grouped by Store and Dept.
    Also creates rolling mean features.
    """
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    lag_cols = []

    for lag in lag_weeks:
        col_name = f"Sales_Lag_{lag}"
        df[col_name] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)
        lag_cols.append(col_name)

    # Rolling mean features
    df["Sales_RollingMean_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
    )
    lag_cols.append("Sales_RollingMean_4")

    df["Sales_RollingMean_12"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda x: x.shift(1).rolling(window=12, min_periods=1).mean()
    )
    lag_cols.append("Sales_RollingMean_12")

    # Rolling std for volatility
    df["Sales_RollingStd_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).std()
    )
    df["Sales_RollingStd_4"] = df["Sales_RollingStd_4"].fillna(0)
    lag_cols.append("Sales_RollingStd_4")

    return df, lag_cols


def train_model(config):
    """
    Train the demand forecasting model.

    Steps:
        1. Load merged Walmart data
        2. Create lag features
        3. Split by time (last N weeks for validation)
        4. Train GradientBoostingRegressor
        5. Evaluate and save
    """
    fc_config = config["forecasting"]
    data_path = config["output_paths"]["walmart_merged"]

    logger.info("Loading merged Walmart data...")
    df = pd.read_csv(data_path, parse_dates=["Date"])
    logger.info(f"  Data shape: {df.shape}")

    # Create lag features
    lag_weeks = fc_config["lag_weeks"]
    df, lag_cols = create_lag_features(df, lag_weeks)

    # Feature columns
    feature_cols = FEATURE_COLS + lag_cols

    # Drop rows with NaN from lag features (early weeks)
    df_model = df.dropna(subset=lag_cols).copy()
    logger.info(f"  After dropping NaN lag rows: {df_model.shape}")

    # Time-based split
    test_weeks = fc_config["test_size_weeks"]
    max_date = df_model["Date"].max()
    split_date = max_date - pd.Timedelta(weeks=test_weeks)

    train_df = df_model[df_model["Date"] <= split_date]
    val_df = df_model[df_model["Date"] > split_date]

    logger.info(f"  Train period: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} rows)")
    logger.info(f"  Val period:   {val_df['Date'].min()} to {val_df['Date'].max()} ({len(val_df)} rows)")

    X_train = train_df[feature_cols].values
    y_train = train_df["Weekly_Sales"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["Weekly_Sales"].values

    # Train model
    logger.info("Training Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=fc_config["n_estimators"],
        max_depth=fc_config["max_depth"],
        learning_rate=fc_config["learning_rate"],
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "val_mae": float(mean_absolute_error(y_val, y_pred_val)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        "val_r2": float(r2_score(y_val, y_pred_val)),
        "feature_importance": dict(zip(feature_cols, model.feature_importances_.tolist())),
        "feature_columns": feature_cols,
        "lag_columns": lag_cols,
    }

    logger.info(f"  Train MAE: {metrics['train_mae']:.2f}, RMSE: {metrics['train_rmse']:.2f}, R²: {metrics['train_r2']:.4f}")
    logger.info(f"  Val   MAE: {metrics['val_mae']:.2f}, RMSE: {metrics['val_rmse']:.2f}, R²: {metrics['val_r2']:.4f}")

    # Save model
    model_path = config["output_paths"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"  Model saved to {model_path}")

    # Save metrics
    metrics_path = config["output_paths"]["metrics_path"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved to {metrics_path}")

    # Save validation predictions for analysis
    val_results = val_df[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    val_results["Predicted_Sales"] = y_pred_val
    val_results.to_csv("outputs/validation_predictions.csv", index=False)

    return model, metrics, feature_cols, lag_cols


def generate_predictions(config, model=None, feature_cols=None, lag_cols=None):
    """
    Generate predictions for test data or future periods.
    Uses the trained model to predict Weekly_Sales.
    """
    if model is None:
        model_path = config["output_paths"]["model_path"]
        model = joblib.load(model_path)
        logger.info(f"  Model loaded from {model_path}")

    if feature_cols is None or lag_cols is None:
        metrics_path = config["output_paths"]["metrics_path"]
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        feature_cols = metrics["feature_columns"]
        lag_cols = metrics["lag_columns"]

    # Load the full training data and test data
    data_path = config["output_paths"]["walmart_merged"]
    df_train = pd.read_csv(data_path, parse_dates=["Date"])

    # For test predictions, we simulate using the last known data
    # Aggregate store-level predictions for the next weeks
    logger.info("Generating predictions for future weeks...")

    # Get the last known date in training data
    last_date = df_train["Date"].max()

    # Create future predictions using the most recent data as base
    future_weeks = 13  # Predict 13 weeks ahead
    all_predictions = []

    for store in df_train["Store"].unique():
        store_data = df_train[df_train["Store"] == store].copy()
        for dept in store_data["Dept"].unique():
            dept_data = store_data[store_data["Dept"] == dept].sort_values("Date").copy()
            if len(dept_data) < 13:
                continue

            # Use the last row as a template for features
            last_row_dict = dept_data.iloc[-1].to_dict()

            for w in range(1, future_weeks + 1):
                future_date = last_date + pd.Timedelta(weeks=w)
                row = last_row_dict.copy()
                row["Date"] = future_date
                row["Year"] = future_date.year
                row["Month"] = future_date.month
                row["Week"] = future_date.isocalendar()[1]
                row["DayOfYear"] = future_date.timetuple().tm_yday
                row["Quarter"] = (future_date.month - 1) // 3 + 1

                # Lag features use most recent known sales
                recent_sales = dept_data["Weekly_Sales"].values
                for lag_w in config["forecasting"]["lag_weeks"]:
                    col = f"Sales_Lag_{lag_w}"
                    idx = len(recent_sales) - lag_w
                    row[col] = recent_sales[idx] if idx >= 0 else recent_sales[0]

                row["Sales_RollingMean_4"] = recent_sales[-4:].mean()
                row["Sales_RollingMean_12"] = recent_sales[-12:].mean()
                row["Sales_RollingStd_4"] = recent_sales[-4:].std() if len(recent_sales) >= 4 else 0

                # Extract features in correct order
                X_list = [row.get(c, 0) for c in feature_cols]
                X = np.array([X_list])
                pred = model.predict(X)[0]
                pred = max(0, pred)  # No negative sales

                # Append the new prediction to recent_sales to use for the next step's rolling features
                recent_sales = np.append(recent_sales, pred)

                all_predictions.append({
                    "Store": int(row["Store"]),
                    "Dept": int(row["Dept"]),
                    "Date": future_date.strftime("%Y-%m-%d"),
                    "Predicted_Weekly_Sales": round(pred, 2),
                })

    predictions_df = pd.DataFrame(all_predictions)

    # Save
    output_path = config["output_paths"]["predictions"]
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path} — shape: {predictions_df.shape}")

    return predictions_df


def run_all(config_path="config.json"):
    """Run complete forecasting pipeline."""
    config = load_config(config_path)
    logger.info("=" * 60)
    logger.info("STARTING DEMAND FORECASTING PIPELINE")
    logger.info("=" * 60)

    model, metrics, feature_cols, lag_cols = train_model(config)
    predictions = generate_predictions(config, model, feature_cols, lag_cols)

    logger.info("=" * 60)
    logger.info("FORECASTING COMPLETE")
    logger.info(f"  Val MAE: {metrics['val_mae']:.2f}")
    logger.info(f"  Val RMSE: {metrics['val_rmse']:.2f}")
    logger.info(f"  Val R²: {metrics['val_r2']:.4f}")
    logger.info(f"  Predictions generated: {len(predictions)}")
    logger.info("=" * 60)

    return model, metrics, predictions


if __name__ == "__main__":
    run_all()
