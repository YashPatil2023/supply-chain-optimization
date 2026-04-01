# 🚚 Supply Chain Optimization System

> An intelligent end-to-end supply chain optimization platform that leverages Machine Learning, Linear Programming, and Vehicle Routing to forecast demand, optimize inventory, plan delivery routes, and simulate real-world disruptions.

**Built for Odoo Hackathon 2026**

---

## 🎯 Features

| Module | Technology | Description |
|--------|-----------|-------------|
| **Demand Forecasting** | Gradient Boosting (scikit-learn) | Predicts weekly sales per store/department using historical data, external factors (CPI, temperature, fuel price), and lag features |
| **Inventory Optimization** | PuLP (Linear Programming) | Minimizes holding + ordering costs while maintaining safety stock levels |
| **Route Optimization** | VRP Heuristic + Folium | Multi-depot vehicle routing with capacity constraints and interactive map visualization |
| **Disruption Simulation** | Custom Engine | Simulates demand spikes, supply delays, and warehouse failures — shows system adaptation |
| **Interactive Dashboard** | Streamlit + Plotly | Real-time visualization with store/department selectors, simulation controls, and CSV exports |

---

## 📁 Project Structure

```
Project/
├── data/                          # Cleaned processed data
│   ├── walmart/
│   └── supply_chain/
├── Dataset/                       # Raw datasets (input)
│   ├── Walmart Sales Forecast/
│   └── Retail Supply Chain Sales Dataset/
├── notebooks/                     # EDA & experimentation
│   ├── data_cleaning.ipynb
│   └── forecasting.ipynb
├── src/                           # Core modules
│   ├── data_processing.py         # Data ingestion & cleaning
│   ├── forecasting.py             # Demand forecasting (ML)
│   ├── inventory.py               # Inventory optimization (LP)
│   ├── routing.py                 # Route optimization (VRP)
│   └── simulation.py              # Disruption simulation
├── app/
│   └── streamlit_app.py           # Interactive dashboard
├── models/                        # Saved trained models
├── outputs/                       # Generated results
├── config.json                    # Dynamic configuration
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Step 1: Process and clean data
python -m src.data_processing

# Step 2: Train forecasting model & generate predictions
python -m src.forecasting

# Step 3: Optimize inventory levels
python -m src.inventory

# Step 4: Optimize delivery routes
python -m src.routing

# Step 5: Run disruption simulations
python -m src.simulation
```

### 3. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Datasets

### Walmart Sales Forecasting (Primary)
- **421K+ rows** of weekly sales data across **45 stores** and **81 departments**
- Includes external factors: Temperature, Fuel Price, CPI, Unemployment, Holiday flags
- Date range: Feb 2010 – Oct 2012

### Retail Supply Chain Dataset (Secondary)
- **10K+ individual orders** with full logistics details
- Includes: Ship Mode, Lead Time, City/State, Returns, Profit margins
- Used for logistics analysis and route planning

---

## ⚙️ Configuration

All system parameters are configurable via `config.json`:

```json
{
    "inventory": {
        "holding_cost_per_unit_per_week": 0.5,
        "ordering_cost_per_order": 50,
        "safety_stock_weeks": 2
    },
    "routing": {
        "num_vehicles": 5,
        "vehicle_capacity": 50000,
        "depots": [...]
    },
    "forecasting": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1
    }
}
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE (Mean Absolute Error) | Average prediction error in dollars |
| RMSE (Root Mean Squared Error) | Penalizes large errors |
| R² Score | Model explanatory power |
| Cost Reduction % | Before vs after inventory optimization |
| Distance Reduction % | Before vs after route optimization |

---

## 🎮 Simulation Scenarios

| Scenario | What Changes | Impact Measured |
|----------|-------------|-----------------|
| **Demand Spike (+20%)** | Predicted demand increases | Inventory cost spike, extra vehicles needed |
| **Supply Delay (7 days)** | Orders arrive late | Stockout risk, extra safety stock cost |
| **Warehouse Failure** | One depot goes offline | Route re-assignment, distance increase, fuel cost |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas / NumPy** — Data processing
- **scikit-learn** — Machine Learning (GradientBoostingRegressor)
- **PuLP** — Linear Programming for inventory optimization
- **Folium** — Interactive map visualization
- **Plotly** — Interactive charts
- **Streamlit** — Web dashboard framework

*Built with ❤️ using Python, Machine Learning, and Optimization*