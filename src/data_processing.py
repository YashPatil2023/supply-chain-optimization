"""
Data Processing Module
======================
Handles loading, cleaning, merging, and feature engineering for both datasets.
Outputs clean CSV files to data/ directory.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Simulated coordinates for 45 Walmart stores (realistic US locations)
# ---------------------------------------------------------------------------
STORE_COORDINATES = {
    1: (33.4484, -112.0740),   # Phoenix, AZ
    2: (34.0522, -118.2437),   # Los Angeles, CA
    3: (41.8781, -87.6298),    # Chicago, IL
    4: (29.7604, -95.3698),    # Houston, TX
    5: (40.7128, -74.0060),    # New York, NY
    6: (33.7490, -84.3880),    # Atlanta, GA
    7: (39.7392, -104.9903),   # Denver, CO
    8: (47.6062, -122.3321),   # Seattle, WA
    9: (25.7617, -80.1918),    # Miami, FL
    10: (32.7767, -96.7970),   # Dallas, TX
    11: (42.3601, -71.0589),   # Boston, MA
    12: (36.1627, -86.7816),   # Nashville, TN
    13: (38.9072, -77.0369),   # Washington, DC
    14: (39.9526, -75.1652),   # Philadelphia, PA
    15: (35.2271, -80.8431),   # Charlotte, NC
    16: (30.2672, -97.7431),   # Austin, TX
    17: (37.7749, -122.4194),  # San Francisco, CA
    18: (44.9778, -93.2650),   # Minneapolis, MN
    19: (29.9511, -90.0715),   # New Orleans, LA
    20: (43.0389, -87.9065),   # Milwaukee, WI
    21: (35.1495, -90.0490),   # Memphis, TN
    22: (38.2527, -85.7585),   # Louisville, KY
    23: (39.1031, -84.5120),   # Cincinnati, OH
    24: (36.1699, -115.1398),  # Las Vegas, NV
    25: (32.7157, -117.1611),  # San Diego, CA
    26: (45.5152, -122.6784),  # Portland, OR
    27: (41.2565, -95.9345),   # Omaha, NE
    28: (35.4676, -97.5164),   # Oklahoma City, OK
    29: (27.9506, -82.4572),   # Tampa, FL
    30: (26.1224, -80.1373),   # Fort Lauderdale, FL
    31: (31.7619, -106.4850),  # El Paso, TX
    32: (33.5207, -86.8025),   # Birmingham, AL
    33: (37.3382, -121.8863),  # San Jose, CA
    34: (40.4406, -79.9959),   # Pittsburgh, PA
    35: (41.4993, -81.6944),   # Cleveland, OH
    36: (34.7465, -92.2896),   # Little Rock, AR
    37: (39.7684, -86.1581),   # Indianapolis, IN
    38: (38.6270, -90.1994),   # St. Louis, MO
    39: (42.3314, -83.0458),   # Detroit, MI
    40: (30.3322, -81.6557),   # Jacksonville, FL
    41: (37.5407, -77.4360),   # Richmond, VA
    42: (28.5383, -81.3792),   # Orlando, FL
    43: (35.7796, -78.6382),   # Raleigh, NC
    44: (36.7468, -119.7726),  # Fresno, CA
    45: (32.2226, -110.9747),  # Tucson, AZ
}

# Human-readable store names (city-based)
STORE_NAMES = {
    1: "Phoenix, AZ", 2: "Los Angeles, CA", 3: "Chicago, IL",
    4: "Houston, TX", 5: "New York, NY", 6: "Atlanta, GA",
    7: "Denver, CO", 8: "Seattle, WA", 9: "Miami, FL",
    10: "Dallas, TX", 11: "Boston, MA", 12: "Nashville, TN",
    13: "Washington, DC", 14: "Philadelphia, PA", 15: "Charlotte, NC",
    16: "Austin, TX", 17: "San Francisco, CA", 18: "Minneapolis, MN",
    19: "New Orleans, LA", 20: "Milwaukee, WI", 21: "Memphis, TN",
    22: "Louisville, KY", 23: "Cincinnati, OH", 24: "Las Vegas, NV",
    25: "San Diego, CA", 26: "Portland, OR", 27: "Omaha, NE",
    28: "Oklahoma City, OK", 29: "Tampa, FL", 30: "Fort Lauderdale, FL",
    31: "El Paso, TX", 32: "Birmingham, AL", 33: "San Jose, CA",
    34: "Pittsburgh, PA", 35: "Cleveland, OH", 36: "Little Rock, AR",
    37: "Indianapolis, IN", 38: "St. Louis, MO", 39: "Detroit, MI",
    40: "Jacksonville, FL", 41: "Richmond, VA", 42: "Orlando, FL",
    43: "Raleigh, NC", 44: "Fresno, CA", 45: "Tucson, AZ",
}

# Human-readable department categories
DEPT_NAMES = {
    1: "Grocery", 2: "Apparel", 3: "Electronics", 4: "Home & Garden",
    5: "Pharmacy", 6: "Toys & Games", 7: "Automotive", 8: "Sporting Goods",
    9: "Health & Beauty", 10: "Frozen Foods", 11: "Dairy", 12: "Bakery",
    13: "Deli & Prepared", 14: "Beverages", 15: "Snacks & Candy",
    16: "Pet Supplies", 17: "Baby Products", 18: "Office Supplies",
    19: "Seasonal", 20: "Furniture", 21: "Jewelry", 22: "Shoes",
    23: "Stationery", 24: "Hardware", 25: "Garden Center",
    26: "Kitchen & Dining", 27: "Bedding & Bath", 28: "Cleaning Supplies",
    29: "Paper Goods", 30: "Meat & Seafood", 31: "Produce",
    32: "Wine & Spirits", 33: "Men's Clothing", 34: "Women's Clothing",
    35: "Kids' Clothing", 36: "Accessories", 37: "Footwear",
    38: "Home Décor", 39: "Luggage", 40: "Books & Media",
    41: "Craft Supplies", 42: "Party Supplies", 43: "Outdoor Living",
    44: "Small Appliances", 45: "Major Appliances", 46: "Flooring",
    47: "Paint", 48: "Plumbing", 49: "Electrical", 50: "Lighting",
    51: "Storage", 52: "Fabric & Sewing", 53: "Musical Instruments",
    54: "Cameras", 55: "Cell Phones", 56: "Video Games",
    57: "Movies & TV", 58: "Music", 59: "Tobacco", 60: "Optical",
    61: "Wireless", 62: "Photo Center", 63: "Fine Jewelry",
    64: "Watches", 65: "Celebration", 67: "Personal Care",
    71: "Swimwear", 72: "Intimates", 77: "Collectibles",
    78: "Community", 79: "Seasonal Décor", 80: "Impulse Items",
    81: "Floral", 82: "Celebrations", 83: "Uniforms",
    85: "Custom Orders", 87: "Fuel Station", 90: "Services",
    91: "Layaway", 92: "Tire Center", 93: "Auto Services",
    94: "Pharmacy Rx", 95: "Clinic", 96: "Optical Services",
    97: "Financial", 98: "Returns", 99: "General Merch",
}


def process_walmart_data(config):
    """
    Load, merge, clean, and feature-engineer the Walmart dataset.

    Steps:
        1. Load train.csv, stores.csv, features.csv
        2. Merge train + stores on Store
        3. Merge result + features on (Store, Date, IsHoliday)
        4. Parse dates and create time features
        5. Handle NaN in MarkDown columns
        6. Add simulated coordinates
        7. Save clean CSV
    """
    paths = config["data_paths"]

    logger.info("Loading Walmart train data...")
    train = pd.read_csv(paths["walmart_train"])
    logger.info(f"  Train shape: {train.shape}")

    logger.info("Loading Walmart stores data...")
    stores = pd.read_csv(paths["walmart_stores"])
    logger.info(f"  Stores shape: {stores.shape}")

    logger.info("Loading Walmart features data...")
    features = pd.read_csv(paths["walmart_features"])
    logger.info(f"  Features shape: {features.shape}")

    # Merge train + stores on Store
    logger.info("Merging train with stores...")
    merged = train.merge(stores, on="Store", how="left")

    # Merge with features on (Store, Date, IsHoliday)
    logger.info("Merging with features...")
    merged = merged.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

    # Parse dates
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    # Time features
    merged["Year"] = merged["Date"].dt.year
    merged["Month"] = merged["Date"].dt.month
    merged["Week"] = merged["Date"].dt.isocalendar().week.astype(int)
    merged["DayOfYear"] = merged["Date"].dt.dayofyear
    merged["Quarter"] = merged["Date"].dt.quarter

    # Handle NaN in MarkDown columns (markdowns not available before Nov 2011)
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for col in markdown_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Fill remaining NaNs
    merged["CPI"] = merged["CPI"].fillna(merged["CPI"].median())
    merged["Unemployment"] = merged["Unemployment"].fillna(merged["Unemployment"].median())
    merged["Temperature"] = merged["Temperature"].fillna(merged["Temperature"].median())
    merged["Fuel_Price"] = merged["Fuel_Price"].fillna(merged["Fuel_Price"].median())

    # Add store coordinates
    merged["Latitude"] = merged["Store"].map(lambda s: STORE_COORDINATES.get(s, (39.8, -98.5))[0])
    merged["Longitude"] = merged["Store"].map(lambda s: STORE_COORDINATES.get(s, (39.8, -98.5))[1])

    # Convert IsHoliday to int
    merged["IsHoliday"] = merged["IsHoliday"].astype(int)

    # Convert Type to numeric
    type_map = {"A": 0, "B": 1, "C": 2}
    merged["Type_encoded"] = merged["Type"].map(type_map)

    # Save
    output_path = config["output_paths"]["walmart_merged"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(f"Walmart merged data saved to {output_path} — shape: {merged.shape}")

    return merged


def process_walmart_test(config):
    """
    Process test.csv the same way as train for prediction.
    """
    paths = config["data_paths"]
    stores = pd.read_csv(paths["walmart_stores"])
    features = pd.read_csv(paths["walmart_features"])
    test = pd.read_csv(paths["walmart_test"])

    # Merge
    merged = test.merge(stores, on="Store", how="left")
    merged = merged.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

    # Parse dates
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    # Time features
    merged["Year"] = merged["Date"].dt.year
    merged["Month"] = merged["Date"].dt.month
    merged["Week"] = merged["Date"].dt.isocalendar().week.astype(int)
    merged["DayOfYear"] = merged["Date"].dt.dayofyear
    merged["Quarter"] = merged["Date"].dt.quarter

    # Handle NaN
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for col in markdown_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged["CPI"] = merged["CPI"].fillna(merged["CPI"].median())
    merged["Unemployment"] = merged["Unemployment"].fillna(merged["Unemployment"].median())
    merged["Temperature"] = merged["Temperature"].fillna(merged["Temperature"].median())
    merged["Fuel_Price"] = merged["Fuel_Price"].fillna(merged["Fuel_Price"].median())

    merged["IsHoliday"] = merged["IsHoliday"].astype(int)
    type_map = {"A": 0, "B": 1, "C": 2}
    merged["Type_encoded"] = merged["Type"].map(type_map)

    merged["Latitude"] = merged["Store"].map(lambda s: STORE_COORDINATES.get(s, (39.8, -98.5))[0])
    merged["Longitude"] = merged["Store"].map(lambda s: STORE_COORDINATES.get(s, (39.8, -98.5))[1])

    return merged


def process_retail_data(config):
    """
    Load, clean, and feature-engineer the Retail Supply Chain dataset.

    Steps:
        1. Load xlsx
        2. Parse dates, compute lead time
        3. Clean Returned column
        4. Assign simulated lat/long per city
        5. Save clean CSV
    """
    paths = config["data_paths"]

    logger.info("Loading Retail Supply Chain data...")
    df = pd.read_excel(paths["retail_supply_chain"])
    logger.info(f"  Retail shape: {df.shape}")

    # Parse dates
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # Compute lead time (days)
    df["Lead_Time_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # Clean Returned column
    df["Returned"] = df["Returned"].fillna("No")
    df["Returned"] = df["Returned"].map(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # Ship Mode encoding
    ship_mode_map = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 3}
    df["Ship_Mode_encoded"] = df["Ship Mode"].map(ship_mode_map).fillna(3)

    # Assign simulated coordinates per unique City+State using hash
    def city_to_coords(city, state):
        """Generate deterministic realistic US coordinates for a city."""
        seed = hashlib.md5(f"{city}_{state}".encode()).hexdigest()
        seed_int = int(seed[:8], 16)
        np.random.seed(seed_int % (2**31))
        lat = np.random.uniform(25.0, 48.0)
        lon = np.random.uniform(-124.0, -71.0)
        return lat, lon

    coords = df.apply(lambda row: city_to_coords(row.get("City", ""), row.get("State", "")), axis=1)
    df["Latitude"] = coords.apply(lambda x: x[0])
    df["Longitude"] = coords.apply(lambda x: x[1])

    # Save
    output_path = config["output_paths"]["retail_clean"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Retail data saved to {output_path} — shape: {df.shape}")

    return df


def run_all(config_path="config.json"):
    """Run the complete data processing pipeline."""
    config = load_config(config_path)
    logger.info("=" * 60)
    logger.info("STARTING DATA PROCESSING PIPELINE")
    logger.info("=" * 60)

    walmart_df = process_walmart_data(config)
    retail_df = process_retail_data(config)

    logger.info("=" * 60)
    logger.info("DATA PROCESSING COMPLETE")
    logger.info(f"  Walmart merged: {walmart_df.shape}")
    logger.info(f"  Retail cleaned: {retail_df.shape}")
    logger.info("=" * 60)

    return walmart_df, retail_df


if __name__ == "__main__":
    run_all()
