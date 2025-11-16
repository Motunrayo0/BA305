import streamlit as st
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# 1. CONFIG

DATA_PATH = "rideshare_demand_final.csv"  
st.set_page_config(
    page_title="Rideshare Driver Helper",
    page_icon="🚗",
    layout="centered"
)

st.title("Where Should a Driver Go Next?")
st.write(
    "This app now uses an XGBoost model to predict **demand** (how many rides are requested and price) "
    "in different areas, given the time and conditions.\n\n"
    "Higher predicted demand → Where driver should be place to wait."
)

# -----------------------------
# 2. LOAD & PREP DATA
# -----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Normalize column names: lowercase, strip
    df.columns = [c.strip().lower() for c in df.columns]

    st.write("Columns in CSV:", df.columns.tolist())

    if "demand" not in df.columns:
        st.error("I couldn't find a 'demand' column in your CSV.")
        st.stop()

    # Drop rows with missing demand
    df = df[df["demand"].notna()].copy()

    return df

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`. "
        "Make sure the CSV is in the same folder as this app."
    )
    st.stop()


# 3. BASIC CHECKS & SETUP

# Identify src_* columns = locations
src_cols = [c for c in df.columns if c.startswith("src_")]
if not src_cols:
    st.error("I couldn't find any columns starting with 'src_' (location indicators).")
    st.stop()

required_cols = ["month", "day", "hour", "weekday", "is_weekend", "demand"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {missing}")
    st.stop()

# Sample to keep training fast
max_rows = 50000
if len(df) > max_rows:
    df_sample = df.sample(n=max_rows, random_state=42)
else:
    df_sample = df.copy()


# 4. TRAIN MODEL (XGBoost, TARGET = DEMAND)

@st.cache_resource
def train_model(df_sample):
    # Target = demand
    y = df_sample["demand"]

    # Features: all except demand and avg_price (to avoid using price to predict demand)
    feature_cols = [
        c for c in df_sample.columns
        if c not in ["demand", "avg_price"]
    ]
    X = df_sample[feature_cols]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(
        steps=[
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    return pipe, feature_cols

with st.spinner("Training XGBoost model (target = demand)..."):
    model, feature_cols = train_model(df_sample)


st.sidebar.header("Driver Settings")

# Month options from data
months = sorted(df_sample["month"].dropna().unique().tolist())
month = st.sidebar.selectbox("Month", months)

# Day of month from data
days = sorted(df_sample["day"].dropna().unique().tolist())
day = st.sidebar.selectbox("Day of month", days)

# Hour
hour = st.sidebar.slider("Hour of day (24h)", 0, 23, 17)

# Weekday (0–6) – if your data uses that scheme
weekday_vals = sorted(df_sample["weekday"].dropna().unique().tolist())
weekday = st.sidebar.selectbox("Weekday (0=Mon ... 6=Sun)", weekday_vals)

# is_weekend
is_weekend = 1 if weekday in [5, 6] else 0
st.sidebar.write(f"`is_weekend` will be set to: **{is_weekend}**")

# Weather “scenario” knobs – simple sliders
avg_temperature = st.sidebar.slider(
    "Temperature (°F)", 
    float(df_sample["avg_temperature"].min()),
    float(df_sample["avg_temperature"].max()),
    float(df_sample["avg_temperature"].median())
)

avg_precip_intensity = st.sidebar.slider(
    "Precip intensity", 
    float(df_sample["avg_precip_intensity"].min()),
    float(df_sample["avg_precip_intensity"].max()),
    float(df_sample["avg_precip_intensity"].median())
)

st.write("### Step 1: You choose time & conditions in the sidebar.")
st.write(
    f"You chose **month {month}**, **day {day}**, **hour {hour}**, "
    f"weekday = **{weekday}**, weekend flag = **{is_weekend}**, "
    f"temperature ≈ **{avg_temperature:.1f}°F**."
)

# -----------------------------
# 6. BUILD CANDIDATE ROWS (ONE PER AREA)
# -----------------------------
# Map human-readable area names to src_ columns
area_map = {
    "Beacon Hill": "src_beacon_hill",
    "Boston University": "src_boston_university",
    "Fenway": "src_fenway",
    "Financial District": "src_financial_district",
    "Haymarket Square": "src_haymarket_square",
    "North End": "src_north_end",
    "North Station": "src_north_station",
    "Northeastern University": "src_northeastern_university",
    "South Station": "src_south_station",
    "Theatre District": "src_theatre_district",
    "West End": "src_west_end",
}

# Keep only areas that exist in this dataset
candidate_areas = {
    name: col for name, col in area_map.items() if col in src_cols
}

if not candidate_areas:
    st.error("None of the expected src_ location columns were found in the dataset.")
    st.stop()

# Use medians for all non-location, non-target columns by default
median_vals = df_sample.median(numeric_only=True)

rows = []
for area_name, src_col in candidate_areas.items():
    row = {}

    for col in feature_cols:
        if col == "month":
            row[col] = month
        elif col == "day":
            row[col] = day
        elif col == "hour":
            row[col] = hour
        elif col == "weekday":
            row[col] = weekday
        elif col == "is_weekend":
            row[col] = is_weekend
        elif col == "avg_temperature":
            row[col] = avg_temperature
        elif col == "avg_precip_intensity":
            row[col] = avg_precip_intensity
        elif col in src_cols:
            row[col] = 0  # all areas off by default
        else:
            # Fallback: use median if available, else 0
            row[col] = float(median_vals.get(col, 0.0))

    
    row[src_col] = 1

    row["__area_name"] = area_name
    rows.append(row)

X_candidates = pd.DataFrame(rows)

area_labels = X_candidates["__area_name"].values
X_candidates_model = X_candidates.drop(columns=["__area_name"])


predicted_demand = model.predict(X_candidates_model)

results_df = pd.DataFrame({
    "Area": area_labels,
    "Predicted demand (relative units)": predicted_demand
}).sort_values(by="Predicted demand (relative units)", ascending=False)

best_row = results_df.iloc[0]

st.write("### Step 2: Predicted demand by area")
st.dataframe(results_df, hide_index=True)

st.write("### Step 3: Recommended spot for the driver (highest predicted demand)")
st.success(
    f"**Recommended area:** `{best_row['Area']}`  \n"
    f"**Predicted demand:** **{best_row['Predicted demand (relative units)']:.2f}**  \n\n"
    "Given your selected time and conditions, this area has the highest predicted demand."
)

st.write("### Demand comparison by area")
chart_df = results_df.set_index("Area")["Predicted demand (relative units)"]
st.bar_chart(chart_df)

